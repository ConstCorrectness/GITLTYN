"""Unit tests for native PyTorch Qwen3 architecture, Continuous Latent Loop, and Disjoint Probe."""

from __future__ import annotations

import itertools

import torch

from datasets import MicroMathTokenizer, PROBE_STAGE_TEMPLATES, build_gsm8k_examples
from latent_loop import RingBufferLatentTee
from qwen3_ablation import (
    Qwen3CausalLM,
    Qwen3ModelConfig,
    TorchContinuousLatentAdapter,
    TorchSidecarProbe,
    flush_qwen3_tee_with_unembed,
    total_parameter_grad_norm,
)
from train_pipeline import run_posttraining, run_pretraining


def test_qwen3_scratch_pretraining_and_kv_cache_equivalence() -> None:
    """Verify Qwen3CausalLM from-scratch pretraining reduces CE loss & KV-cache matches full forward."""
    torch.manual_seed(17)
    device = torch.device("cpu")
    examples = build_gsm8k_examples(8, seed=17)
    texts = [f"{ex.prompt} {ex.verbose_cot} {ex.telegraphic_cot}" for ex in examples]
    tok = MicroMathTokenizer(texts=texts, max_vocab=128)
    cfg = Qwen3ModelConfig.micro_ablation(vocab_size=tok.vocab_size)

    model = Qwen3CausalLM(cfg).to(device=device, dtype=torch.float32)
    init_loss, final_loss, losses = run_pretraining(
        model=model,
        tokenizer=tok,
        train_examples=examples,
        device=device,
        mode="scratch",
        steps=14,
        batch_size=4,
        lr=4e-3,
    )
    assert len(losses) == 14
    assert final_loss < init_loss

    # Verify KV-cached greedy generation matches step-by-step uncached argmax
    model.eval()
    prompt_ids = torch.tensor(
        [tok.encode(f"Problem: {examples[0].prompt}", add_special_tokens=False)[:16]],
        dtype=torch.long,
    )
    cached_out, kv_kb = model.generate_greedy(prompt_ids, max_new_tokens=6)
    assert kv_kb > 0.0
    assert cached_out.shape == (1, prompt_ids.shape[1] + 6)

    uncached = prompt_ids.clone()
    with torch.no_grad():
        for _ in range(6):
            logits, _ = model(uncached)
            next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            uncached = torch.cat([uncached, next_tok], dim=1)

    assert torch.equal(cached_out, uncached)

    # Verify left-padded generate_greedy_batch matches single-sequence generate_greedy on unequal lengths
    p0 = tok.encode(f"Problem: {examples[0].prompt}", add_special_tokens=False)[:11]
    p1 = tok.encode(f"Problem: {examples[1].prompt}", add_special_tokens=False)[:17]
    single0, _ = model.generate_greedy(torch.tensor([p0], dtype=torch.long), max_new_tokens=5)
    single1, _ = model.generate_greedy(torch.tensor([p1], dtype=torch.long), max_new_tokens=5)
    batched = model.generate_greedy_batch([p0, p1], max_new_tokens=5, pad_token_id=tok.pad_token_id)
    assert batched[0] == single0[0, len(p0) :].tolist()
    assert batched[1] == single1[0, len(p1) :].tolist()

    # Verify forward_prefix + forward_loss_only matches full forward CE loss exactly
    labels = prompt_ids.clone()
    labels[:, :5] = -100
    _, full_ce = model(input_ids=prompt_ids, labels=labels)
    assert full_ce is not None
    pref = model.model.forward_prefix(prompt_ids, stop_layer=1)
    fast_ce = model.forward_loss_only(labels=labels, prefix_hidden=pref, start_layer=1)
    assert torch.allclose(full_ce, fast_ce, atol=1e-5)


def test_torch_latent_adapter_tee_non_interference_and_bounded_memory() -> None:
    """Verify enable_latent_tee=False vs True bit-for-bit equality and M <= K ring buffer bound."""
    torch.manual_seed(29)
    cfg = Qwen3ModelConfig.micro_ablation(vocab_size=96)
    model = Qwen3CausalLM(cfg)
    adapter = TorchContinuousLatentAdapter(hidden_dim=cfg.hidden_dim)
    probe = TorchSidecarProbe(hidden_dim=cfg.hidden_dim, templates=PROBE_STAGE_TEMPLATES)

    z_0 = torch.randn(1, cfg.hidden_dim, dtype=torch.float32)

    prod_tee = RingBufferLatentTee(capacity=4)
    assert prod_tee.enable_latent_tee is False
    z_prod, _ = adapter(z_init=z_0, steps_k=9, tee=prod_tee)
    assert len(prod_tee) == 0

    audit_tee = RingBufferLatentTee(capacity=4, enable_latent_tee=True)
    z_audit, _ = adapter(z_init=z_0, steps_k=9, tee=audit_tee)
    assert torch.equal(z_prod, z_audit)
    assert len(audit_tee) == 4

    records = flush_qwen3_tee_with_unembed(
        tee=audit_tee,
        lm_head_weight=model.lm_head.weight,
        norm_weight=model.model.norm.weight,
        id_to_token=[f"tok_{i}" for i in range(cfg.vocab_size)],
        sidecar_probe=probe,
        candidate_token_ids=list(range(12)),
        top_k=5,
        temperature=1.05,
    )
    assert [r.step_k for r in records] == [6, 7, 8, 9]
    for rec in records:
        assert rec.l2_norm > 0.0
        assert len(rec.logit_lens_top_k) == 5
        assert 0.15 <= rec.logit_lens_top_k[0].prob <= 0.92


def test_torch_sidecar_probe_disjoint_optimizer_zero_backbone_gradient() -> None:
    """Verify optimizing L_probe on live autograd-tracked z_k leaves grad_{backbone, f_theta} == 0."""
    torch.manual_seed(101)
    device = torch.device("cpu")
    examples = build_gsm8k_examples(6, seed=101)
    texts = [f"{ex.prompt} {ex.verbose_cot} {ex.telegraphic_cot}" for ex in examples]
    tok = MicroMathTokenizer(texts=texts, max_vocab=128)
    cfg = Qwen3ModelConfig.micro_ablation(vocab_size=tok.vocab_size)

    model = Qwen3CausalLM(cfg).to(device=device)
    adapter = TorchContinuousLatentAdapter(hidden_dim=cfg.hidden_dim).to(device=device)
    probe = TorchSidecarProbe(
        hidden_dim=cfg.hidden_dim, templates=PROBE_STAGE_TEMPLATES
    ).to(device=device)

    # Zero all backbone & adapter gradients while keeping requires_grad=True
    for p in itertools.chain(model.parameters(), adapter.parameters()):
        p.requires_grad_(True)
        if p.grad is not None:
            p.grad.zero_()

    in_ids = torch.tensor(
        [tok.encode(examples[0].prompt, add_special_tokens=False)[:14]],
        dtype=torch.long,
    )
    hidden = model.forward_hidden(in_ids)
    z_0 = hidden[:, -1, :]
    tee = RingBufferLatentTee(capacity=4, enable_latent_tee=True)
    z_final, traj = adapter(z_init=z_0, steps_k=6, tee=tee)

    # Confirm live trajectory tensors ARE attached to autograd
    assert all(z.requires_grad and z.grad_fn is not None for z in traj)

    live_batch = torch.stack([z.squeeze(0) for z in traj[-4:]], dim=0)
    targets = torch.tensor([0, 1, 2, 5], dtype=torch.long)
    probe_opt = torch.optim.AdamW(probe.parameters(), lr=3e-2)

    init_probe_loss = float(probe.compute_loss(live_batch, targets).item())
    for _ in range(18):
        _ = probe.step_disjoint_optimizer(live_batch, targets, probe_opt)
    final_probe_loss = float(probe.compute_loss(live_batch, targets).item())

    assert final_probe_loss < init_probe_loss
    assert probe.optimizer_steps == 18

    # CRITICAL INVARIANT: Zero gradient norm across Qwen3CausalLM & TorchContinuousLatentAdapter
    leaked = total_parameter_grad_norm(
        itertools.chain(model.parameters(), adapter.parameters())
    )
    assert leaked == 0.0

    # Primary task loss on z_final DOES populate both adapter and backbone gradients
    z_final.pow(2).sum().backward()
    assert total_parameter_grad_norm(adapter.parameters()) > 0.0
    assert total_parameter_grad_norm(model.parameters()) > 0.0

    # Also verify full Stage 2 post-training pipeline assert passes end-to-end
    metrics = run_posttraining(
        model=model,
        adapter=adapter,
        probe=probe,
        tokenizer=tok,
        train_examples=examples,
        device=device,
        mode="scratch",
        sft_steps=4,
        latent_steps_train=4,
        probe_steps=6,
        recurrence_k=4,
        ring_capacity_m=4,
    )
    assert metrics.verified_zero_backbone_grad_on_probe is True
