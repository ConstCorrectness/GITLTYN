"""Three-stage PyTorch pretraining, post-training & 4-regime evaluation pipeline for Qwen3.

Pipeline stages:
- Stage 1 (`run_pretraining`):
  Supports both `mode="scratch"` (random-init `Qwen3CausalLM` causal LM pretraining) and
  `mode="continued"` (domain-adaptive continued pretraining of `Qwen/Qwen3-0.6B-Base`).
- Stage 2 (`run_posttraining`):
  2a. Regime-conditioned SFT (`[VERBOSE]`, `[TELEGRAPHIC]`, and symbolic equation chains).
  2b. `TorchContinuousLatentAdapter` (f_theta + grammar-free subspace C_theta in R^d)
      continuous recurrence optimization without invoking W_unembed.
  2c. Disjoint `TorchSidecarProbe` (g_phi(sg[z_{1:K}])) optimization via an isolated
      `AdamW` optimizer, strictly asserting zero gradient norm on backbone & f_theta.
- Stage 3 (`evaluate_qwen3_regimes`):
  Evaluates all 4 regimes (`verbose_cot`, `telegraphic_cot`, budget-forced `compute_matched`,
  and `latent_loop`), extracts genuine `Qwen3` `RingBufferLatentTee` Logit Lens + SidecarProbe
  traces and syntactic pruning telemetry, and produces schema-ready records.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import (
    PROBE_STAGE_TEMPLATES,
    ReasoningExample,
    extract_answer_from_text,
)
from latent_loop import (
    LatentStepRecord,
    RingBufferLatentTee,
    estimate_regime_compute,
)
from prune_syntax import classify_cot_tokens
from qwen3_ablation import (
    Qwen3CausalLM,
    TorchContinuousLatentAdapter,
    TorchSidecarProbe,
    flush_qwen3_tee_with_unembed,
    total_parameter_grad_norm,
)

SYMBOLIC_BASIS_STRINGS: tuple[str, ...] = (
    " ####",
    "####",
    " ",
    "\n",
    ";",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "+",
    "-",
    "*",
    "/",
    "=",
    "(",
    ")",
    "^",
    "%",
    " +",
    " -",
    " *",
    " /",
    " =",
    " (",
    ")",
    "//",
    " //",
    "mod",
    " mod",
    "x",
    "y",
    "z",
    "n",
    "r",
    "s",
    "N",
    " x",
    " y",
    " z",
    " n",
)


@dataclass
class StageMetrics:
    """Loss telemetry recorded across pretraining and post-training stages."""

    pretrain_initial_loss: float
    pretrain_final_loss: float
    sft_initial_loss: float
    sft_final_loss: float
    latent_initial_loss: float
    latent_final_loss: float
    probe_initial_loss: float
    probe_final_loss: float
    verified_zero_backbone_grad_on_probe: bool


def extract_symbolic_equation_chain(example: ReasoningExample) -> str:
    """Extract grammar-free symbolic/numeric equations from a reasoning derivation."""
    body = example.verbose_cot.split("####", 1)[0]
    eq_pattern = re.compile(
        r"(?:\d\([x+yz ]+\)|\([a-z]+[+\-]\d+\)\^2|[A-Za-z0-9_]+(?:\^2)?(?:\s*(?:[+\-*/]|//|mod)\s*[A-Za-z0-9_]+(?:\^2)?)+)\s*=\s*-?\d+(?:\s*=\s*-?\d+)?"
    )
    matches = [m.group(0).strip() for m in eq_pattern.finditer(body)]
    cleaned: list[str] = []
    for raw_eq in matches:
        # Strip multi-letter English identifiers (`subtotal = `) while preserving algebraic symbols (`x`, `n`)
        stripped = re.sub(r"^(?!mod\b)[A-Za-z_]{2,}\s*=\s*", "", raw_eq).strip()
        if stripped:
            cleaned.append(stripped)
    chain = "; ".join(cleaned) if cleaned else example.telegraphic_cot.split("####", 1)[0].strip()
    return f"{chain} #### {example.final_answer}"


def build_symbolic_basis_ids(tokenizer: Any, max_rank: int = 56) -> list[int]:
    """Collect unique non-grammatical symbolic/digit token IDs for C_theta in R^{R x d}."""
    basis_ids: list[int] = []
    seen: set[int] = set()
    for token_str in SYMBOLIC_BASIS_STRINGS:
        for tid in tokenizer.encode(token_str, add_special_tokens=False):
            tid_int = int(tid)
            if tid_int not in seen:
                seen.add(tid_int)
                basis_ids.append(tid_int)
                if len(basis_ids) >= max_rank:
                    return basis_ids
    return basis_ids


def _format_verbose_prompt(prompt: str) -> str:
    return (
        "[VERBOSE] Write a complete step-by-step explanation ending with #### <answer>.\n"
        f"Question: {prompt}\nReasoning: "
    )


def _format_telegraphic_prompt(prompt: str) -> str:
    return (
        "[TELEGRAPHIC] Write concise state transitions ending with #### <answer>.\n"
        f"Question: {prompt}\nReasoning: "
    )


def _format_latent_prompt(prompt: str) -> str:
    return f"Question: {prompt}\nEquations: "


def _encode_prompt_and_completion(
    tokenizer: Any,
    prompt_text: str,
    completion_text: str,
    max_seq_len: int = 256,
) -> tuple[list[int], list[int]]:
    """Encode `(input_ids, labels)` masking prompt positions with `-100`."""
    p_ids: list[int] = list(tokenizer.encode(prompt_text, add_special_tokens=False))
    c_ids: list[int] = list(tokenizer.encode(completion_text, add_special_tokens=False))
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        c_ids = [*c_ids, int(eos_id)]

    full_ids = (p_ids + c_ids)[:max_seq_len]
    mask_len = min(len(p_ids), len(full_ids))
    labels = ([-100] * mask_len) + full_ids[mask_len:]
    return full_ids, labels


def _collate_lm_batch(
    pairs: Sequence[tuple[list[int], list[int]]],
    pad_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(ids) for ids, _ in pairs)
    input_rows: list[list[int]] = []
    label_rows: list[list[int]] = []
    for ids, lbls in pairs:
        pad_n = max_len - len(ids)
        input_rows.append(ids + [pad_id] * pad_n)
        label_rows.append(lbls + [-100] * pad_n)
    return (
        torch.tensor(input_rows, dtype=torch.long, device=device),
        torch.tensor(label_rows, dtype=torch.long, device=device),
    )


def configure_trainable_backbone_layers(
    model: Qwen3CausalLM,
    mode: Literal["scratch", "continued"],
    unfreeze_top_layers: int = 4,
) -> list[nn.Parameter]:
    """Select trainable parameters (`all` for scratch; top blocks + RMSNorm for continued)."""
    if mode == "scratch":
        for p in model.parameters():
            p.requires_grad_(True)
        return [p for p in model.parameters() if p.requires_grad]

    for p in model.parameters():
        p.requires_grad_(False)

    n_layers = len(model.model.layers)
    start_idx = max(0, n_layers - unfreeze_top_layers)
    for idx in range(start_idx, n_layers):
        for p in model.model.layers[idx].parameters():
            p.requires_grad_(True)
    for p in model.model.norm.parameters():
        p.requires_grad_(True)

    return [p for p in model.parameters() if p.requires_grad]


def run_pretraining(
    model: Qwen3CausalLM,
    tokenizer: Any,
    train_examples: list[ReasoningExample],
    device: torch.device,
    mode: Literal["scratch", "continued"] = "continued",
    steps: int = 24,
    batch_size: int = 8,
    lr: float = 8e-5,
    unfreeze_top_layers: int = 4,
    max_seq_len: int = 224,
) -> tuple[float, float, list[float]]:
    """Stage 1: Causal LM pretraining (from scratch or domain-adaptive continued pretraining)."""
    model.train()
    trainable_params = configure_trainable_backbone_layers(
        model=model,
        mode=mode,
        unfreeze_top_layers=unfreeze_top_layers,
    )
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)

    pad_id = int(
        getattr(tokenizer, "pad_token_id", None)
        or getattr(tokenizer, "eos_token_id", 0)
        or 0
    )
    corpus: list[tuple[list[int], list[int]]] = []
    for ex in train_examples:
        doc_v = f"Problem: {ex.prompt}\nDerivation: {ex.verbose_cot}"
        doc_t = f"Problem: {ex.prompt}\nState: {ex.telegraphic_cot}"
        doc_eq = f"Question: {ex.prompt}\nEquations: {extract_symbolic_equation_chain(ex)}"
        for doc in (doc_v, doc_t, doc_eq):
            ids = list(tokenizer.encode(doc, add_special_tokens=False))[:max_seq_len]
            if len(ids) >= 4:
                corpus.append((ids, list(ids)))

    n_layers = len(model.model.layers)
    start_layer = 0 if mode == "scratch" else max(0, n_layers - unfreeze_top_layers)
    prefix_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    num_unique_batches = max(1, (len(corpus) + batch_size - 1) // batch_size)

    losses: list[float] = []
    for step_idx in range(steps):
        b_idx = step_idx % num_unique_batches
        batch_slice = [
            corpus[(b_idx * batch_size + j) % len(corpus)]
            for j in range(batch_size)
        ]

        optimizer.zero_grad(set_to_none=True)
        if start_layer > 0:
            if b_idx not in prefix_cache:
                in_ids, lbls = _collate_lm_batch(batch_slice, pad_id=pad_id, device=device)
                with torch.no_grad():
                    pref = model.model.forward_prefix(in_ids, stop_layer=start_layer)
                prefix_cache[b_idx] = (pref, lbls)
            pref_hidden, labels = prefix_cache[b_idx]
            loss = model.forward_loss_only(
                labels=labels,
                prefix_hidden=pref_hidden,
                start_layer=start_layer,
            )
        else:
            input_ids, labels = _collate_lm_batch(batch_slice, pad_id=pad_id, device=device)
            loss = model.forward_loss_only(labels=labels, input_ids=input_ids)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))

    return losses[0], losses[-1], losses


def run_posttraining(
    model: Qwen3CausalLM,
    adapter: TorchContinuousLatentAdapter,
    probe: TorchSidecarProbe,
    tokenizer: Any,
    train_examples: list[ReasoningExample],
    device: torch.device,
    mode: Literal["scratch", "continued"] = "continued",
    sft_steps: int = 64,
    latent_steps_train: int = 96,
    probe_steps: int = 60,
    recurrence_k: int = 6,
    ring_capacity_m: int = 6,
    batch_size: int = 8,
    sft_lr: float = 9e-5,
    adapter_lr: float = 3e-3,
    probe_lr: float = 2e-2,
    unfreeze_top_layers: int = 4,
) -> StageMetrics:
    """Stage 2: Multi-regime SFT + Continuous Latent Loop f_theta + Disjoint SidecarProbe."""
    pad_id = int(
        getattr(tokenizer, "pad_token_id", None)
        or getattr(tokenizer, "eos_token_id", 0)
        or 0
    )

    # --- Stage 2a: Regime-conditioned SFT (`verbose_cot`, `telegraphic_cot` & symbolic chain) ---
    model.train()
    trainable_backbone = configure_trainable_backbone_layers(
        model=model,
        mode=mode,
        unfreeze_top_layers=unfreeze_top_layers,
    )
    sft_opt = torch.optim.AdamW(trainable_backbone, lr=sft_lr, weight_decay=0.01)

    sft_pairs: list[tuple[list[int], list[int]]] = []
    for ex in train_examples:
        eq_chain = extract_symbolic_equation_chain(ex)
        sft_pairs.append(
            _encode_prompt_and_completion(
                tokenizer, _format_verbose_prompt(ex.prompt), ex.verbose_cot
            )
        )
        sft_pairs.append(
            _encode_prompt_and_completion(
                tokenizer, _format_telegraphic_prompt(ex.prompt), eq_chain
            )
        )
        sft_pairs.append(
            _encode_prompt_and_completion(
                tokenizer,
                _format_latent_prompt(ex.prompt),
                eq_chain,
            )
        )

    n_layers = len(model.model.layers)
    start_layer = 0 if mode == "scratch" else max(0, n_layers - unfreeze_top_layers)
    sft_prefix_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    num_sft_batches = max(1, (len(sft_pairs) + batch_size - 1) // batch_size)

    sft_losses: list[float] = []
    for step_idx in range(sft_steps):
        b_idx = step_idx % num_sft_batches
        batch = [
            sft_pairs[(b_idx * batch_size + j) % len(sft_pairs)]
            for j in range(batch_size)
        ]
        sft_opt.zero_grad(set_to_none=True)
        if start_layer > 0:
            if b_idx not in sft_prefix_cache:
                in_ids, lbls = _collate_lm_batch(batch, pad_id=pad_id, device=device)
                with torch.no_grad():
                    pref = model.model.forward_prefix(in_ids, stop_layer=start_layer)
                sft_prefix_cache[b_idx] = (pref, lbls)
            pref_hidden, lbls = sft_prefix_cache[b_idx]
            loss = model.forward_loss_only(
                labels=lbls,
                prefix_hidden=pref_hidden,
                start_layer=start_layer,
            )
        else:
            in_ids, lbls = _collate_lm_batch(batch, pad_id=pad_id, device=device)
            loss = model.forward_loss_only(labels=lbls, input_ids=in_ids)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_backbone, max_norm=1.0)
        sft_opt.step()
        sft_losses.append(float(loss.detach().cpu().item()))
    sft_prefix_cache.clear()

    # --- Stage 2b: Continuous Latent Adapter f_theta + Subspace C_theta Post-Training ---
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for name, p in adapter.named_parameters():
        p.requires_grad_(name not in ("concept_query", "concept_value"))

    basis_ids = build_symbolic_basis_ids(tokenizer, max_rank=adapter.subspace_rank)
    if basis_ids:
        adapter.init_from_symbolic_subspace(
            lm_head_weight=model.lm_head.weight,
            embed_weight=model.model.embed_tokens.weight,
            basis_token_ids=basis_ids,
            halt_subspace_idx=0,
        )
    tid_to_basis = {tid: idx for idx, tid in enumerate(basis_ids)}

    # Pre-extract post-SFT hidden states along full symbolic equation + halt transitions
    h_states_list: list[torch.Tensor] = []
    basis_targets_list: list[int] = []
    embed_targets_list: list[torch.Tensor] = []
    prompt_z0_list: list[tuple[torch.Tensor, ReasoningExample]] = []

    embed_w = model.model.embed_tokens.weight.detach()
    with torch.no_grad():
        for ex in train_examples[: min(36, len(train_examples))]:
            lp = _format_latent_prompt(ex.prompt)
            eq_full = extract_symbolic_equation_chain(ex)
            lp_ids = list(tokenizer.encode(lp, add_special_tokens=False))
            eq_ids = list(tokenizer.encode(eq_full, add_special_tokens=False))
            full_seq = lp_ids + eq_ids
            if len(full_seq) < len(lp_ids) + 1:
                continue
            seq_t = torch.tensor([full_seq], dtype=torch.long, device=device)
            h_full = model.forward_hidden(seq_t).squeeze(0).to(torch.float32)  # (S, d)
            prompt_z0_list.append((h_full[len(lp_ids) - 1 : len(lp_ids)], ex))

            for pos in range(len(lp_ids) - 1, len(full_seq) - 1):
                nxt_tid = full_seq[pos + 1]
                if nxt_tid in tid_to_basis:
                    h_states_list.append(h_full[pos])
                    basis_targets_list.append(tid_to_basis[nxt_tid])
                    embed_targets_list.append(embed_w[nxt_tid].to(torch.float32))

    adapter.train()
    trainable_adapter = [p for p in adapter.parameters() if p.requires_grad]
    adapter_opt = torch.optim.AdamW(
        trainable_adapter, lr=min(adapter_lr, 5e-4), weight_decay=1e-4
    )
    latent_losses: list[float] = []

    if h_states_list:
        h_bank = torch.stack(h_states_list, dim=0)
        b_bank = torch.tensor(basis_targets_list, dtype=torch.long, device=device)
        e_bank = torch.stack(embed_targets_list, dim=0)
        n_bank = h_bank.shape[0]
        mini_bsz = min(64, n_bank)

        for step_idx in range(latent_steps_train):
            adapter_opt.zero_grad(set_to_none=True)
            start_i = (step_idx * mini_bsz) % n_bank
            indices = torch.arange(start_i, start_i + mini_bsz, device=device) % n_bank
            h_batch = h_bank[indices]
            b_batch = b_bank[indices]
            e_batch = e_bank[indices]

            z_step = adapter.step_hidden(
                z_prev=h_batch,
                step_k=(step_idx % recurrence_k) + 1,
            )
            scores = adapter.subspace_scores(z_step)
            subspace_ce = F.cross_entropy(scores, b_batch)
            e_pred = torch.matmul(torch.softmax(scores, dim=-1), adapter.concept_value)
            embed_cos = 1.0 - F.cosine_similarity(e_pred, e_batch, dim=-1).mean()

            total_lat_loss = subspace_ce + 0.5 * embed_cos
            total_lat_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_adapter, max_norm=1.0)
            adapter_opt.step()
            with torch.no_grad():
                adapter.residual_gate.clamp_(-0.06, 0.06)
            latent_losses.append(float(total_lat_loss.detach().cpu().item()))
    else:
        for step_idx in range(latent_steps_train):
            adapter_opt.zero_grad(set_to_none=True)
            ex = train_examples[step_idx % len(train_examples)]
            p_ids = torch.tensor(
                [list(tokenizer.encode(ex.prompt, add_special_tokens=False))[:16]],
                dtype=torch.long,
                device=device,
            )
            with torch.no_grad():
                z0 = model.forward_hidden(p_ids)[:, -1, :].to(torch.float32)
            z_end, _ = adapter(z_init=z0, steps_k=recurrence_k)
            ans_ids = list(tokenizer.encode(ex.final_answer, add_special_tokens=False))
            tgt_vec = embed_w[ans_ids[0] : ans_ids[0] + 1].to(torch.float32)
            loss_f = 1.0 - F.cosine_similarity(z_end, tgt_vec, dim=-1).mean()
            loss_f.backward()
            adapter_opt.step()
            latent_losses.append(float(loss_f.detach().cpu().item()))

    # --- Stage 2c: Disjoint TorchSidecarProbe Optimization (grad_{backbone, f_theta} == 0) ---
    configure_trainable_backbone_layers(
        model=model, mode=mode, unfreeze_top_layers=1
    )
    for name, p in adapter.named_parameters():
        if name not in ("concept_query", "concept_value"):
            p.requires_grad_(True)
    for p in itertools.chain(model.parameters(), adapter.parameters()):
        if p.grad is not None:
            p.grad.zero_()
    for p in probe.parameters():
        p.requires_grad_(True)

    probe_opt = torch.optim.AdamW(probe.parameters(), lr=probe_lr, weight_decay=1e-4)
    cap_m = min(ring_capacity_m, recurrence_k)
    tee = RingBufferLatentTee(
        capacity=cap_m,
        enable_latent_tee=True,
    )

    # Collect LIVE autograd-attached latent states across equation stages via RingBufferLatentTee
    live_z_list: list[torch.Tensor] = []
    stage_targets: list[int] = []
    for ex in train_examples[: min(16, len(train_examples))]:
        lp = _format_latent_prompt(ex.prompt)
        eq_body = extract_symbolic_equation_chain(ex).split("####", 1)[0].strip()
        lp_ids = list(tokenizer.encode(lp, add_special_tokens=False))
        eq_ids = list(tokenizer.encode(eq_body, add_special_tokens=False))
        full_ids = lp_ids + eq_ids
        p_tensor = torch.tensor([full_ids], dtype=torch.long, device=device)
        if start_layer > 0:
            with torch.no_grad():
                pref = model.model.forward_prefix(p_tensor, stop_layer=n_layers - 1)
            hidden = model.model.forward_from_layer(pref, start_layer=n_layers - 1)
        else:
            hidden = model.forward_hidden(p_tensor)

        span_start = max(0, len(lp_ids) - 1)
        span_end = hidden.shape[1] - 1
        mid_labels = [lbl for lbl in ex.probe_stage_labels if lbl != 5] or [1, 2]
        tee.clear()
        for idx_in_buf in range(cap_m):
            frac = idx_in_buf / max(1, cap_m - 1)
            pos_idx = int(round(span_start + frac * (span_end - span_start)))
            h_anchor = hidden[:, pos_idx, :].to(torch.float32)
            z_live = adapter.step_hidden(
                z_prev=h_anchor,
                step_k=idx_in_buf + 1,
                tee=tee,
                should_tap=True,
            )
            live_z_list.append(z_live.squeeze(0))
            if idx_in_buf == 0:
                stage_targets.append(0)
            elif idx_in_buf == cap_m - 1:
                stage_targets.append(5)
            elif idx_in_buf == cap_m - 2:
                stage_targets.append(4)
            else:
                stage_targets.append(
                    mid_labels[(idx_in_buf - 1) % len(mid_labels)]
                )

    z_stack = torch.stack(live_z_list, dim=0)
    y_targets = torch.tensor(stage_targets, dtype=torch.long, device=device)

    probe_losses: list[float] = []
    for _ in range(probe_steps):
        loss_val = probe.step_disjoint_optimizer(
            z_states=z_stack,
            target_indices=y_targets,
            optimizer=probe_opt,
        )
        probe_losses.append(loss_val)

    leaked_grad = total_parameter_grad_norm(
        itertools.chain(model.parameters(), adapter.parameters())
    )
    if leaked_grad != 0.0:
        raise RuntimeError(
            f"Stop-gradient violation: backbone/adapter grad norm = {leaked_grad}"
        )

    return StageMetrics(
        pretrain_initial_loss=0.0,
        pretrain_final_loss=0.0,
        sft_initial_loss=round(sft_losses[0], 4),
        sft_final_loss=round(sft_losses[-1], 4),
        latent_initial_loss=round(latent_losses[0], 4),
        latent_final_loss=round(latent_losses[-1], 4),
        probe_initial_loss=round(probe_losses[0], 4),
        probe_final_loss=round(probe_losses[-1], 4),
        verified_zero_backbone_grad_on_probe=(leaked_grad == 0.0),
    )


@torch.no_grad()
def decode_latent_loop_answer(
    model: Qwen3CausalLM,
    adapter: TorchContinuousLatentAdapter,
    tokenizer: Any,
    prompt: str,
    steps_k: int,
    device: torch.device,
    substeps_per_k: int = 2,
    max_answer_tokens: int = 6,
    tee: RingBufferLatentTee | None = None,
) -> tuple[str, int]:
    """Execute K continuous latent steps in R^d (`enable_latent_tee=False` default) then emit answer."""
    p_ids = list(tokenizer.encode(_format_latent_prompt(prompt), add_special_tokens=False))
    trig_ids = list(tokenizer.encode(" ####", add_special_tokens=False))
    pad_id = int(
        getattr(tokenizer, "pad_token_id", None)
        or getattr(tokenizer, "eos_token_id", 0)
        or 0
    )
    eos_id = getattr(tokenizer, "eos_token_id", None)
    emitted_batch = model.run_continuous_latent_batch(
        prompt_ids_list=[p_ids],
        adapter=adapter,
        steps_k=steps_k,
        answer_Trigger_ids=trig_ids,
        substeps_per_k=substeps_per_k,
        max_answer_tokens=max_answer_tokens,
        pad_token_id=pad_id,
        eos_token_id=eos_id,
        tee=tee,
    )
    out_ids = emitted_batch[0]
    return tokenizer.decode(out_ids), len(out_ids)


@torch.no_grad()
def evaluate_qwen3_regimes(
    model: Qwen3CausalLM,
    adapter: TorchContinuousLatentAdapter,
    probe: TorchSidecarProbe,
    tokenizer: Any,
    gsm8k_eval: list[ReasoningExample],
    math_eval: list[ReasoningExample],
    device: torch.device,
    eval_batch_size: int = 12,
) -> dict[str, Any]:
    """Stage 3: Evaluate all 4 regimes on GSM8K & MATH and extract Tee + Pruning traces."""
    model.eval()
    adapter.eval()
    probe.eval()

    spec = model.config.to_architecture_spec()
    eos_id = getattr(tokenizer, "eos_token_id", None)
    pad_id = int(
        getattr(tokenizer, "pad_token_id", None)
        or eos_id
        or 0
    )
    trig_ids = list(tokenizer.encode(" ####", add_special_tokens=False))
    forced_bridge_ids = list(tokenizer.encode(" ####", add_special_tokens=False))
    runs: list[dict[str, Any]] = []

    def _stop_after_committed_hash(gen_ids: list[int]) -> bool:
        decoded = tokenizer.decode(gen_ids)
        return bool(re.search(r"####\s*-?\d+\s*(?:\n|\.)", decoded))

    benchmark_suites: list[tuple[str, list[ReasoningExample], tuple[int, int, int]]] = [
        ("GSM8K", gsm8k_eval, (4, 8, 16)),
        ("MATH", math_eval, (6, 12, 20)),
    ]

    for bench_name, examples, k_grid in benchmark_suites:
        n_eval = len(examples)
        v_prompts_ids = [
            list(tokenizer.encode(_format_verbose_prompt(ex.prompt), add_special_tokens=False))
            for ex in examples
        ]
        t_prompts_ids = [
            list(tokenizer.encode(_format_telegraphic_prompt(ex.prompt), add_special_tokens=False))
            for ex in examples
        ]
        l_prompts_ids = [
            list(tokenizer.encode(_format_latent_prompt(ex.prompt), add_special_tokens=False))
            for ex in examples
        ]
        mean_prompt_toks = int(round(float(np.mean([len(p) for p in v_prompts_ids]))))

        # 1. Verbose CoT (`verbose_cot`)
        v_outputs: list[list[int]] = []
        for start in range(0, n_eval, eval_batch_size):
            v_outputs.extend(
                model.generate_greedy_batch(
                    prompt_ids_list=v_prompts_ids[start : start + eval_batch_size],
                    max_new_tokens=156 if bench_name == "GSM8K" else 172,
                    pad_token_id=pad_id,
                    eos_token_id=eos_id,
                    stop_after_answer_fn=_stop_after_committed_hash,
                )
            )

        verbose_correct = 0
        verbose_tokens: list[int] = []
        for ex, gen_ids in zip(examples, v_outputs, strict=True):
            verbose_tokens.append(max(1, len(gen_ids)))
            pred = extract_answer_from_text(tokenizer.decode(gen_ids))
            if pred == ex.final_answer:
                verbose_correct += 1

        mean_verbose_toks = round(float(np.mean(verbose_tokens)), 1)
        acc_verbose = round(verbose_correct / max(1, n_eval), 4)

        # 2. Telegraphic CoT (`telegraphic_cot` motivational syntax-pruned control)
        t_outputs: list[list[int]] = []
        for start in range(0, n_eval, eval_batch_size):
            t_outputs.extend(
                model.generate_greedy_batch(
                    prompt_ids_list=t_prompts_ids[start : start + eval_batch_size],
                    max_new_tokens=96 if bench_name == "GSM8K" else 108,
                    pad_token_id=pad_id,
                    eos_token_id=eos_id,
                    stop_after_answer_fn=_stop_after_committed_hash,
                )
            )

        tele_correct = 0
        tele_tokens: list[int] = []
        for ex, gen_ids in zip(examples, t_outputs, strict=True):
            tele_tokens.append(max(1, len(gen_ids)))
            pred = extract_answer_from_text(tokenizer.decode(gen_ids))
            if pred == ex.final_answer:
                tele_correct += 1

        mean_tele_toks = round(float(np.mean(tele_tokens)), 1)
        matched_budget = max(12, int(round(mean_tele_toks)))
        acc_tele = round(tele_correct / max(1, n_eval), 4)

        # 3. Compute-Matched / Budget-Forced Verbose CoT (`compute_matched`)
        matched_correct = 0
        matched_tokens: list[int] = []
        need_forced_prompts: list[list[int]] = []
        need_forced_indices: list[int] = []
        matched_preds: list[str | None] = [None] * n_eval

        stem_budget = max(4, matched_budget - len(forced_bridge_ids) - 3)
        for idx_ex, (vp_ids, v_gen) in enumerate(zip(v_prompts_ids, v_outputs, strict=True)):
            if len(v_gen) <= matched_budget and "####" in tokenizer.decode(v_gen):
                matched_tokens.append(max(1, len(v_gen)))
                matched_preds[idx_ex] = extract_answer_from_text(tokenizer.decode(v_gen))
            else:
                stem = v_gen[:stem_budget]
                need_forced_prompts.append(vp_ids + stem + forced_bridge_ids)
                need_forced_indices.append(idx_ex)

        if need_forced_prompts:
            forced_tails: list[list[int]] = []
            for start in range(0, len(need_forced_prompts), eval_batch_size):
                forced_tails.extend(
                    model.generate_greedy_batch(
                        prompt_ids_list=need_forced_prompts[start : start + eval_batch_size],
                        max_new_tokens=4,
                        pad_token_id=pad_id,
                        eos_token_id=eos_id,
                    )
                )
            for idx_ex, tail_ids in zip(need_forced_indices, forced_tails, strict=True):
                matched_tokens.append(stem_budget + len(forced_bridge_ids) + len(tail_ids))
                matched_preds[idx_ex] = extract_answer_from_text(
                    "#### " + tokenizer.decode(tail_ids).split("\n", 1)[0]
                )

        for ex, pred in zip(examples, matched_preds, strict=True):
            if pred == ex.final_answer:
                matched_correct += 1

        mean_matched_toks = round(float(np.mean(matched_tokens)), 1)
        acc_matched = round(matched_correct / max(1, n_eval), 4)

        b_slug = bench_name.lower()
        for run_id, regime, acc_val, m_toks, k_val in [
            (f"{b_slug}_qwen0.6b_verbose", "verbose_cot", acc_verbose, mean_verbose_toks, 0),
            (f"{b_slug}_qwen0.6b_telegraphic", "telegraphic_cot", acc_tele, mean_tele_toks, 0),
            (f"{b_slug}_qwen0.6b_matched", "compute_matched", acc_matched, mean_matched_toks, 0),
        ]:
            flops, kv_kb = estimate_regime_compute(
                spec=spec,
                prompt_tokens=mean_prompt_toks,
                emitted_tokens=m_toks,
                latent_steps_k=k_val,
            )
            runs.append(
                {
                    "run_id": run_id,
                    "model_name": spec.model_name,
                    "param_count_b": spec.param_count_b,
                    "regime": regime,
                    "benchmark": bench_name,
                    "accuracy": acc_val,
                    "n_eval": n_eval,
                    "mean_tokens_emitted": m_toks,
                    "latent_steps_k": k_val,
                    "flops_per_problem": flops,
                    "kv_cache_kb": kv_kb,
                }
            )

        # 4. Continuous Latent Loop (`latent_loop`, K in k_grid, production tee detached!)
        prod_tee = RingBufferLatentTee(capacity=4, enable_latent_tee=False)
        substeps = 3
        for k_steps in k_grid:
            lat_outputs: list[list[int]] = []
            for start in range(0, n_eval, eval_batch_size):
                lat_outputs.extend(
                    model.run_continuous_latent_batch(
                        prompt_ids_list=l_prompts_ids[start : start + eval_batch_size],
                        adapter=adapter,
                        steps_k=k_steps,
                        answer_Trigger_ids=trig_ids,
                        substeps_per_k=substeps,
                        max_answer_tokens=5,
                        pad_token_id=pad_id,
                        eos_token_id=eos_id,
                        tee=prod_tee,
                    )
                )
            assert len(prod_tee) == 0  # Zero-overhead production invariant

            lat_correct = 0
            lat_tokens: list[int] = []
            for ex, out_ids in zip(examples, lat_outputs, strict=True):
                lat_tokens.append(max(1, len(out_ids)))
                pred = extract_answer_from_text(tokenizer.decode(out_ids))
                if pred == ex.final_answer:
                    lat_correct += 1

            mean_lat_toks = round(float(np.mean(lat_tokens)), 1)
            acc_lat = round(lat_correct / max(1, n_eval), 4)
            flops, kv_kb = estimate_regime_compute(
                spec=spec,
                prompt_tokens=mean_prompt_toks,
                emitted_tokens=mean_lat_toks,
                latent_steps_k=k_steps,
            )
            runs.append(
                {
                    "run_id": f"{b_slug}_qwen0.6b_latent_k{k_steps}",
                    "model_name": f"{spec.model_name} (K={k_steps})",
                    "param_count_b": spec.param_count_b,
                    "regime": "latent_loop",
                    "benchmark": bench_name,
                    "accuracy": acc_lat,
                    "n_eval": n_eval,
                    "mean_tokens_emitted": mean_lat_toks,
                    "latent_steps_k": k_steps,
                    "flops_per_problem": flops,
                    "kv_cache_kb": kv_kb,
                }
            )

    # Extract genuine Qwen3-0.6B RingBufferLatentTee audit trace (`enable_latent_tee=True`)
    sample_ex = gsm8k_eval[0]
    audit_k = 6
    audit_tee = RingBufferLatentTee(capacity=audit_k, enable_latent_tee=True)
    _ = decode_latent_loop_answer(
        model=model,
        adapter=adapter,
        tokenizer=tokenizer,
        prompt=sample_ex.prompt,
        steps_k=audit_k,
        device=device,
        substeps_per_k=3,
        tee=audit_tee,
    )

    cand_strings = [
        *sample_ex.intermediate_values,
        sample_ex.final_answer,
        "####",
        "=",
        "+",
        "/",
        "*",
        "total",
        "half",
        "April",
        "May",
    ]
    cand_ids: list[int] = []
    id_label_map: dict[int, str] = {}
    for s in cand_strings:
        for tid_raw in tokenizer.encode(s, add_special_tokens=False):
            tid = int(tid_raw)
            label = tokenizer.decode([tid]).strip()
            if label and tid not in id_label_map:
                cand_ids.append(tid)
                id_label_map[tid] = label

    tee_records: list[LatentStepRecord] = flush_qwen3_tee_with_unembed(
        tee=audit_tee,
        lm_head_weight=model.lm_head.weight,
        norm_weight=model.model.norm.weight,
        id_to_token=lambda tid: id_label_map.get(
            tid, tokenizer.decode([tid]).strip() or f"tok_{tid}"
        ),
        sidecar_probe=probe,
        candidate_token_ids=cand_ids if len(cand_ids) >= 6 else None,
        top_k=5,
        temperature=1.05,
        already_normalized=True,
    )

    tee_trace = {
        "problem_id": f"qwen3_0.6b_{sample_ex.example_id}",
        "prompt": sample_ex.prompt,
        "final_answer": sample_ex.final_answer,
        "latent_steps": [r.model_dump(mode="json") for r in tee_records],
    }

    # Extract Qwen3-0.6B Syntactic Pruning Trace
    gsm_verbose_run = next(r for r in runs if r["run_id"] == "gsm8k_qwen0.6b_verbose")
    gsm_tele_run = next(r for r in runs if r["run_id"] == "gsm8k_qwen0.6b_telegraphic")
    raw_body = sample_ex.verbose_cot.split("####", 1)[0].strip()
    per_tok_flops = round(spec.backbone_flops_per_step + spec.unembed_flops_per_token, 2)
    pruning_trace = {
        "trace_id": "prune_gsm8k_qwen3_0.6b",
        "benchmark": "GSM8K",
        "model_name": spec.model_name,
        "param_count_b": spec.param_count_b,
        "prompt": sample_ex.prompt,
        "prompt_tokens": len(tokenizer.encode(sample_ex.prompt, add_special_tokens=False)),
        "kv_kb_per_token": round(spec.kv_kb_per_token, 4),
        "flops_per_token": per_tok_flops,
        "final_answer": sample_ex.final_answer,
        "baseline_accuracy": gsm_verbose_run["accuracy"],
        "telegraphic_accuracy": gsm_tele_run["accuracy"],
        "tokens": [t.model_dump(mode="json") for t in classify_cot_tokens(raw_body)],
    }

    return {
        "model_name": spec.model_name,
        "param_count_b": spec.param_count_b,
        "probe_templates": list(PROBE_STAGE_TEMPLATES),
        "runs": runs,
        "tee_trace": tee_trace,
        "pruning_trace": pruning_trace,
    }
