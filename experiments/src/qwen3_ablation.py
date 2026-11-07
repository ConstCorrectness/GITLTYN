"""Native PyTorch Qwen3-0.6B Causal LM + Continuous Latent Loop + Disjoint Sidecar Probe.

Supports both:
1. From-scratch pretraining (`Qwen3CausalLM(config)` from random initialization), and
2. Continued pretraining & post-training from HuggingFace `Qwen/Qwen3-0.6B-Base` weights
   (`Qwen3CausalLM.from_huggingface("Qwen/Qwen3-0.6B-Base")`).

Enforces all GITLTYN architectural invariants on live `torch.Tensor` autograd graphs:
- `RingBufferLatentTee.push(z_k)` unconditionally calls `z_k.detach()` (sg[z_k]).
- Fixed circular capacity M <= K (`O(1)` trajectory memory w.r.t. recurrence depth K),
  with `enable_latent_tee=False` by default in production.
- `TorchSidecarProbe` optimizes L_probe via a disjoint `torch.optim.Optimizer` strictly
  over probe weights phi so grad_{theta_backbone} L_probe == 0 and grad_{theta_adapter} L_probe == 0.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from latent_loop import (
    LatentStepRecord,
    LogitLensPrediction,
    RingBufferLatentTee,
    SLMArchitectureSpec,
)


@dataclass(frozen=True)
class Qwen3ModelConfig:
    """Architectural hyperparameters matching Qwen3 (GQA + QK-RMSNorm + RoPE + SwiGLU)."""

    model_name: str = "Qwen3-0.6B"
    param_count_b: float = 0.6
    vocab_size: int = 151936
    hidden_dim: int = 1024
    intermediate_dim: int = 3072
    num_layers: int = 28
    num_attention_heads: int = 16
    num_kv_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    tie_word_embeddings: bool = True

    @classmethod
    def qwen3_0_6b(cls) -> Qwen3ModelConfig:
        """Exact HuggingFace `Qwen/Qwen3-0.6B-Base` architectural configuration."""
        return cls()

    @classmethod
    def micro_ablation(cls, vocab_size: int = 256) -> Qwen3ModelConfig:
        """Compact Qwen3 configuration for fast deterministic from-scratch unit tests."""
        return cls(
            model_name="Qwen3-Micro",
            param_count_b=0.004,
            vocab_size=vocab_size,
            hidden_dim=128,
            intermediate_dim=384,
            num_layers=4,
            num_attention_heads=4,
            num_kv_heads=2,
            head_dim=32,
            rms_norm_eps=1e-6,
            rope_theta=10_000.0,
            tie_word_embeddings=True,
        )

    def to_architecture_spec(self) -> SLMArchitectureSpec:
        """Convert into `SLMArchitectureSpec` for exact FLOP & FP16 KV-cache accounting."""
        return SLMArchitectureSpec(
            model_name=self.model_name,
            param_count_b=self.param_count_b,
            num_layers=self.num_layers,
            hidden_dim=self.hidden_dim,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            vocab_size=self.vocab_size,
        )


class Qwen3RMSNorm(nn.Module):
    """Root-Mean-Square Layer Normalization matching Qwen3 precision semantics."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x_fp32 * torch.rsqrt(variance + self.eps)
        return self.weight * x_normed.to(orig_dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


class Qwen3RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) precomputing cos/sin caches dynamically."""

    def __init__(self, head_dim: int, rope_theta: float = 1_000_000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, x: torch.Tensor, seq_len: int, offset: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq: torch.Tensor = self.inv_freq  # type: ignore[assignment]
        positions = torch.arange(
            offset, offset + seq_len, device=x.device, dtype=torch.float32
        )
        freqs = torch.outer(positions, inv_freq.to(x.device))
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors of shape (B, H, T, D)."""
    cos_b = cos.unsqueeze(0).unsqueeze(0)
    sin_b = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos_b) + (_rotate_half(q) * sin_b)
    k_embed = (k * cos_b) + (_rotate_half(k) * sin_b)
    return q_embed, k_embed


class Qwen3Attention(nn.Module):
    """Grouped-Query Self-Attention with Qwen3 per-head QK-RMSNorm."""

    def __init__(self, config: Qwen3ModelConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(
            config.hidden_dim, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_dim, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_dim, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_dim, bias=False
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        bsz, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(
            bsz, seq_len, self.num_kv_heads, self.head_dim
        )
        v = self.v_proj(hidden_states).view(
            bsz, seq_len, self.num_kv_heads, self.head_dim
        )

        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)

        present_kv = (k, v) if use_cache else None

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=(seq_len > 1 and past_kv is None),
            enable_gqa=(self.num_heads != self.num_kv_heads),
        )
        attn_out = (
            attn_out.transpose(1, 2)
            .contiguous()
            .view(bsz, seq_len, self.num_heads * self.head_dim)
        )
        return self.o_proj(attn_out), present_kv


class Qwen3MLP(nn.Module):
    """SwiGLU feed-forward network matching Qwen3 `mlp.{gate_proj,up_proj,down_proj}`."""

    def __init__(self, config: Qwen3ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_dim, config.intermediate_dim, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_dim, config.intermediate_dim, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_dim, config.hidden_dim, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3DecoderLayer(nn.Module):
    """Single pre-RMSNorm Qwen3 transformer block."""

    def __init__(self, config: Qwen3ModelConfig) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(config)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_dim, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        attn_delta, present_kv = self.self_attn(
            self.input_layernorm(hidden_states),
            cos,
            sin,
            past_kv=past_kv,
            use_cache=use_cache,
        )
        hidden_states = hidden_states + attn_delta
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
        return hidden_states, present_kv


class Qwen3Backbone(nn.Module):
    """Qwen3 transformer stack returning pre-unembed top-layer states h_t^{(L)} in R^d."""

    def __init__(self, config: Qwen3ModelConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config) for _ in range(config.num_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(
            head_dim=config.head_dim, rope_theta=config.rope_theta
        )

    def forward_embeds(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]] | None]:
        """Run all L transformer blocks on continuous embeddings in R^{B x S x d}."""
        hidden_states = inputs_embeds
        seq_len = inputs_embeds.shape[1]
        offset = past_key_values[0][0].shape[2] if past_key_values else 0
        cos, sin = self.rotary_emb(hidden_states, seq_len=seq_len, offset=offset)
        next_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = (
            [] if use_cache else None
        )
        for idx, layer in enumerate(self.layers):
            layer_past = past_key_values[idx] if past_key_values is not None else None
            hidden_states, present_kv = layer(
                hidden_states,
                cos,
                sin,
                past_kv=layer_past,
                use_cache=use_cache,
            )
            if next_cache is not None and present_kv is not None:
                next_cache.append(present_kv)
        return self.norm(hidden_states), next_cache

    def forward_prefix(
        self,
        input_ids: torch.Tensor,
        stop_layer: int,
    ) -> torch.Tensor:
        """Run `embed_tokens` + `layers[:stop_layer]` (for caching frozen prefix states)."""
        hidden_states = self.embed_tokens(input_ids)
        seq_len = input_ids.shape[1]
        cos, sin = self.rotary_emb(hidden_states, seq_len=seq_len, offset=0)
        for idx in range(min(stop_layer, len(self.layers))):
            hidden_states, _ = self.layers[idx](
                hidden_states, cos, sin, past_kv=None, use_cache=False
            )
        return hidden_states

    def forward_from_layer(
        self,
        hidden_states: torch.Tensor,
        start_layer: int,
    ) -> torch.Tensor:
        """Resume forward pass from `layers[start_layer:]` + final `RMSNorm`."""
        seq_len = hidden_states.shape[1]
        cos, sin = self.rotary_emb(hidden_states, seq_len=seq_len, offset=0)
        for idx in range(start_layer, len(self.layers)):
            hidden_states, _ = self.layers[idx](
                hidden_states, cos, sin, past_kv=None, use_cache=False
            )
        return self.norm(hidden_states)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]] | None]:
        hidden_states = self.embed_tokens(input_ids)
        return self.forward_embeds(
            inputs_embeds=hidden_states,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )


class Qwen3CausalLM(nn.Module):
    """Native PyTorch Qwen3 Causal LM supporting scratch init & HuggingFace weight loading."""

    def __init__(self, config: Qwen3ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3Backbone(config)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        self._init_weights()
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def _init_weights(self) -> None:
        std = 1.0 / math.sqrt(self.config.hidden_dim)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)

    @classmethod
    def from_huggingface(
        cls,
        repo_id: str = "Qwen/Qwen3-0.6B-Base",
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ) -> Qwen3CausalLM:
        """Load pretrained `Qwen/Qwen3-0.6B-Base` safetensors directly into `Qwen3CausalLM`."""
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from transformers import AutoConfig

        hf_cfg = AutoConfig.from_pretrained(repo_id)
        rope_params = getattr(hf_cfg, "rope_parameters", None) or {}
        rope_theta = float(
            rope_params.get("rope_theta", getattr(hf_cfg, "rope_theta", 1_000_000.0) or 1_000_000.0)
        )
        cfg = Qwen3ModelConfig(
            model_name="Qwen3-0.6B",
            param_count_b=0.6,
            vocab_size=int(hf_cfg.vocab_size),
            hidden_dim=int(hf_cfg.hidden_size),
            intermediate_dim=int(hf_cfg.intermediate_size),
            num_layers=int(hf_cfg.num_hidden_layers),
            num_attention_heads=int(hf_cfg.num_attention_heads),
            num_kv_heads=int(hf_cfg.num_key_value_heads),
            head_dim=int(hf_cfg.head_dim),
            rms_norm_eps=float(hf_cfg.rms_norm_eps),
            rope_theta=rope_theta,
            tie_word_embeddings=bool(getattr(hf_cfg, "tie_word_embeddings", True)),
        )

        # Construct directly on meta/target dtype to avoid redundant memory copies
        prev_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(dtype)
            instance = cls(cfg)
        finally:
            torch.set_default_dtype(prev_dtype)

        weights_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
        state_dict = load_file(weights_path, device="cpu")
        if cfg.tie_word_embeddings and "lm_head.weight" not in state_dict:
            state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]

        missing, unexpected = instance.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"State dict mismatch loading {repo_id}: missing={missing}, unexpected={unexpected}"
            )
        if cfg.tie_word_embeddings:
            instance.lm_head.weight = instance.model.embed_tokens.weight

        return instance.to(device=device, dtype=dtype)

    def forward_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return normalized top-layer hidden states h_t^{(L)} in R^d without unembedding."""
        hidden, _ = self.model(input_ids, use_cache=False)
        return hidden

    def unembed(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states in R^d through W_unembed in R^{d x |V|}."""
        return self.lm_head(hidden_states)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Standard causal LM forward pass returning (logits, optional_ce_loss)."""
        hidden = self.forward_hidden(input_ids)
        logits = self.unembed(hidden)
        loss: torch.Tensor | None = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous().to(torch.float32)
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    def forward_loss_only(
        self,
        labels: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        prefix_hidden: torch.Tensor | None = None,
        start_layer: int = 0,
    ) -> torch.Tensor:
        """Compute exact causal LM cross-entropy projecting only non-masked positions."""
        if prefix_hidden is not None:
            hidden = self.model.forward_from_layer(prefix_hidden, start_layer=start_layer)
        elif input_ids is not None:
            hidden = self.forward_hidden(input_ids)
        else:
            raise ValueError("Either input_ids or prefix_hidden must be provided")

        shift_hidden = hidden[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        active_mask = shift_labels != -100
        active_hidden = shift_hidden[active_mask]
        active_targets = shift_labels[active_mask]
        active_logits = self.unembed(active_hidden).to(torch.float32)
        return F.cross_entropy(active_logits, active_targets)

    @torch.no_grad()
    def generate_greedy_batch(
        self,
        prompt_ids_list: Sequence[Sequence[int]],
        max_new_tokens: int,
        pad_token_id: int = 0,
        eos_token_id: int | None = None,
        stop_after_answer_fn: Callable[[list[int]], bool] | None = None,
    ) -> list[list[int]]:
        """Left-padded batched KV-cached greedy decoding returning generated token IDs per item."""
        if not prompt_ids_list:
            return []
        was_training = self.training
        self.eval()
        device = self.model.embed_tokens.weight.device

        bsz = len(prompt_ids_list)
        max_p = max(len(p) for p in prompt_ids_list)
        padded_rows: list[list[int]] = []
        valid_rows: list[list[bool]] = []
        for p in prompt_ids_list:
            pad_len = max_p - len(p)
            padded_rows.append(([pad_token_id] * pad_len) + list(p))
            valid_rows.append(([False] * pad_len) + ([True] * len(p)))

        input_ids = torch.tensor(padded_rows, dtype=torch.long, device=device)
        valid_mask = torch.tensor(valid_rows, dtype=torch.bool, device=device)

        # Prefill with left-padding-aware position IDs and causal mask
        pos_ids = (valid_mask.to(torch.long).cumsum(dim=-1) - 1).clamp(min=0)
        hidden_states = self.model.embed_tokens(input_ids)
        inv_freq: torch.Tensor = self.model.rotary_emb.inv_freq  # type: ignore[assignment]
        freqs = pos_ids.unsqueeze(-1).to(torch.float32) * inv_freq.to(device)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_b = emb.cos().to(hidden_states.dtype).unsqueeze(1)
        sin_b = emb.sin().to(hidden_states.dtype).unsqueeze(1)

        causal = torch.tril(torch.ones((max_p, max_p), dtype=torch.bool, device=device))
        diag = torch.eye(max_p, dtype=torch.bool, device=device)
        attn_mask = (causal[None, None, :, :] & valid_mask[:, None, None, :]) | diag[None, None, :, :]

        past_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
        h = hidden_states
        for layer in self.model.layers:
            ln_h = layer.input_layernorm(h)
            q = layer.self_attn.q_proj(ln_h).view(bsz, max_p, layer.self_attn.num_heads, layer.self_attn.head_dim)
            k = layer.self_attn.k_proj(ln_h).view(bsz, max_p, layer.self_attn.num_kv_heads, layer.self_attn.head_dim)
            v = layer.self_attn.v_proj(ln_h).view(bsz, max_p, layer.self_attn.num_kv_heads, layer.self_attn.head_dim)
            q = layer.self_attn.q_norm(q).transpose(1, 2)
            k = layer.self_attn.k_norm(k).transpose(1, 2)
            v = v.transpose(1, 2)
            q = (q * cos_b) + (_rotate_half(q) * sin_b)
            k = (k * cos_b) + (_rotate_half(k) * sin_b)
            past_kv.append((k, v))
            attn_out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False,
                enable_gqa=(layer.self_attn.num_heads != layer.self_attn.num_kv_heads),
            )
            attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, max_p, -1)
            h = h + layer.self_attn.o_proj(attn_out)
            h = h + layer.mlp(layer.post_attention_layernorm(h))

        h = self.model.norm(h)
        next_logits = self.unembed(h[:, -1, :])
        next_tok = torch.argmax(next_logits, dim=-1, keepdim=True)  # (B, 1)

        generated: list[list[int]] = [[] for _ in range(bsz)]
        finished = [False] * bsz
        cur_pos = pos_ids[:, -1:] + 1  # (B, 1)

        for i in range(bsz):
            tid = int(next_tok[i, 0].item())
            if eos_token_id is not None and tid == eos_token_id:
                finished[i] = True
            else:
                generated[i].append(tid)
                if stop_after_answer_fn is not None and stop_after_answer_fn(generated[i]):
                    finished[i] = True

        for _ in range(max(0, max_new_tokens - 1)):
            if all(finished):
                break
            valid_mask = torch.cat(
                [valid_mask, torch.ones((bsz, 1), dtype=torch.bool, device=device)], dim=1
            )
            step_mask = valid_mask[:, None, None, :]  # (B, 1, 1, KV_len)
            h = self.model.embed_tokens(next_tok)
            freqs_s = cur_pos.unsqueeze(-1).to(torch.float32) * inv_freq.to(device)
            emb_s = torch.cat((freqs_s, freqs_s), dim=-1)
            cos_s = emb_s.cos().to(h.dtype).unsqueeze(1)
            sin_s = emb_s.sin().to(h.dtype).unsqueeze(1)

            new_past: list[tuple[torch.Tensor, torch.Tensor]] = []
            for idx, layer in enumerate(self.model.layers):
                ln_h = layer.input_layernorm(h)
                q = layer.self_attn.q_proj(ln_h).view(bsz, 1, layer.self_attn.num_heads, layer.self_attn.head_dim)
                k = layer.self_attn.k_proj(ln_h).view(bsz, 1, layer.self_attn.num_kv_heads, layer.self_attn.head_dim)
                v = layer.self_attn.v_proj(ln_h).view(bsz, 1, layer.self_attn.num_kv_heads, layer.self_attn.head_dim)
                q = layer.self_attn.q_norm(q).transpose(1, 2)
                k = layer.self_attn.k_norm(k).transpose(1, 2)
                v = v.transpose(1, 2)
                q = (q * cos_s) + (_rotate_half(q) * sin_s)
                k = (k * cos_s) + (_rotate_half(k) * sin_s)
                k_cat = torch.cat([past_kv[idx][0], k], dim=2)
                v_cat = torch.cat([past_kv[idx][1], v], dim=2)
                new_past.append((k_cat, v_cat))
                attn_out = F.scaled_dot_product_attention(
                    q, k_cat, v_cat, attn_mask=step_mask, dropout_p=0.0, is_causal=False,
                    enable_gqa=(layer.self_attn.num_heads != layer.self_attn.num_kv_heads),
                )
                attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, 1, -1)
                h = h + layer.self_attn.o_proj(attn_out)
                h = h + layer.mlp(layer.post_attention_layernorm(h))

            past_kv = new_past
            cur_pos = cur_pos + 1
            h = self.model.norm(h)
            next_logits = self.unembed(h[:, -1, :])
            next_tok = torch.argmax(next_logits, dim=-1, keepdim=True)

            for i in range(bsz):
                if finished[i]:
                    continue
                tid = int(next_tok[i, 0].item())
                if eos_token_id is not None and tid == eos_token_id:
                    finished[i] = True
                else:
                    generated[i].append(tid)
                    if stop_after_answer_fn is not None and stop_after_answer_fn(generated[i]):
                        finished[i] = True

        if was_training:
            self.train()
        return generated

    @torch.no_grad()
    def generate_greedy(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: int | None = None,
        stop_substrings: Sequence[str] | None = None,
        decode_fn: Any | None = None,
    ) -> tuple[torch.Tensor, float]:
        """Deterministic KV-cached greedy decoding returning `(full_ids, measured_kv_kb)`."""
        was_training = self.training
        self.eval()
        cur = input_ids.clone()
        prompt_len = cur.shape[1]

        hidden, past_kv = self.model(cur, past_key_values=None, use_cache=True)
        next_logits = self.unembed(hidden[:, -1, :])
        next_tok = torch.argmax(next_logits, dim=-1, keepdim=True)
        cur = torch.cat([cur, next_tok], dim=1)

        for _ in range(max(0, max_new_tokens - 1)):
            if eos_token_id is not None and int(next_tok.item()) == eos_token_id:
                break
            if stop_substrings and decode_fn is not None:
                gen_text = decode_fn(cur[0, prompt_len:].tolist())
                if any(s in gen_text for s in stop_substrings):
                    break
            hidden, past_kv = self.model(
                next_tok, past_key_values=past_kv, use_cache=True
            )
            next_logits = self.unembed(hidden[:, -1, :])
            next_tok = torch.argmax(next_logits, dim=-1, keepdim=True)
            cur = torch.cat([cur, next_tok], dim=1)

        # Measure FP16-equivalent KV-cache footprint (2 bytes per scalar across K and V)
        kv_scalars = (
            sum(k_t.numel() + v_t.numel() for k_t, v_t in past_kv)
            if past_kv is not None
            else 0
        )
        measured_kv_kb = (kv_scalars * 2.0) / 1024.0

        if was_training:
            self.train()
        return cur, round(measured_kv_kb, 2)

    @torch.no_grad()
    def run_continuous_latent_batch(
        self,
        prompt_ids_list: Sequence[Sequence[int]],
        adapter: TorchContinuousLatentAdapter,
        steps_k: int,
        answer_Trigger_ids: Sequence[int],
        substeps_per_k: int = 3,
        max_answer_tokens: int = 6,
        pad_token_id: int = 0,
        eos_token_id: int | None = None,
        tee: RingBufferLatentTee | None = None,
    ) -> list[list[int]]:
        """Execute batched continuous latent recurrence in R^d without W_unembed, then emit answers.

        During k = 1..K (`steps_k * substeps_per_k` continuous R^d transitions), `W_unembed`
        (`self.lm_head` in R^{d x |V|}) and discrete `argmax` are NEVER invoked; each transition
        forms a continuous superposition vector `e_cont in R^d` via `adapter.project_continuous_embed`
        and updates KV state directly. `W_unembed` is called strictly after `answer_Trigger_ids`.
        """
        if not prompt_ids_list:
            return []
        was_training = self.training
        self.eval()
        adapter.eval()
        device = self.model.embed_tokens.weight.device
        dtype = self.model.embed_tokens.weight.dtype

        bsz = len(prompt_ids_list)
        max_p = max(len(p) for p in prompt_ids_list)
        padded_rows: list[list[int]] = []
        valid_rows: list[list[bool]] = []
        for p in prompt_ids_list:
            pad_len = max_p - len(p)
            padded_rows.append(([pad_token_id] * pad_len) + list(p))
            valid_rows.append(([False] * pad_len) + ([True] * len(p)))

        input_ids = torch.tensor(padded_rows, dtype=torch.long, device=device)
        valid_mask = torch.tensor(valid_rows, dtype=torch.bool, device=device)
        pos_ids = (valid_mask.to(torch.long).cumsum(dim=-1) - 1).clamp(min=0)

        h = self.model.embed_tokens(input_ids)
        inv_freq: torch.Tensor = self.model.rotary_emb.inv_freq  # type: ignore[assignment]
        freqs = pos_ids.unsqueeze(-1).to(torch.float32) * inv_freq.to(device)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_b = emb.cos().to(dtype).unsqueeze(1)
        sin_b = emb.sin().to(dtype).unsqueeze(1)

        causal = torch.tril(torch.ones((max_p, max_p), dtype=torch.bool, device=device))
        diag = torch.eye(max_p, dtype=torch.bool, device=device)
        attn_mask = (causal[None, None, :, :] & valid_mask[:, None, None, :]) | diag[None, None, :, :]

        past_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer in self.model.layers:
            ln_h = layer.input_layernorm(h)
            q = layer.self_attn.q_proj(ln_h).view(bsz, max_p, layer.self_attn.num_heads, layer.self_attn.head_dim)
            k = layer.self_attn.k_proj(ln_h).view(bsz, max_p, layer.self_attn.num_kv_heads, layer.self_attn.head_dim)
            v = layer.self_attn.v_proj(ln_h).view(bsz, max_p, layer.self_attn.num_kv_heads, layer.self_attn.head_dim)
            q = layer.self_attn.q_norm(q).transpose(1, 2)
            k = layer.self_attn.k_norm(k).transpose(1, 2)
            v = v.transpose(1, 2)
            q = (q * cos_b) + (_rotate_half(q) * sin_b)
            k = (k * cos_b) + (_rotate_half(k) * sin_b)
            past_kv.append((k, v))
            attn_out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False,
                enable_gqa=(layer.self_attn.num_heads != layer.self_attn.num_kv_heads),
            )
            attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, max_p, -1)
            h = h + layer.self_attn.o_proj(attn_out)
            h = h + layer.mlp(layer.post_attention_layernorm(h))

        z_curr = self.model.norm(h)[:, -1, :]  # z_0 = h_t^{(L)} in R^{B x d}
        cur_pos = pos_ids[:, -1:] + 1

        def _step_continuous_embed(e_step: torch.Tensor) -> torch.Tensor:
            nonlocal valid_mask, past_kv, cur_pos
            valid_mask = torch.cat(
                [valid_mask, torch.ones((bsz, 1), dtype=torch.bool, device=device)], dim=1
            )
            step_mask = valid_mask[:, None, None, :]
            freqs_s = cur_pos.unsqueeze(-1).to(torch.float32) * inv_freq.to(device)
            emb_s = torch.cat((freqs_s, freqs_s), dim=-1)
            cos_s = emb_s.cos().to(dtype).unsqueeze(1)
            sin_s = emb_s.sin().to(dtype).unsqueeze(1)

            h_s = e_step.to(dtype)
            new_past: list[tuple[torch.Tensor, torch.Tensor]] = []
            for idx_l, layer in enumerate(self.model.layers):
                ln_h = layer.input_layernorm(h_s)
                q = layer.self_attn.q_proj(ln_h).view(bsz, 1, layer.self_attn.num_heads, layer.self_attn.head_dim)
                k = layer.self_attn.k_proj(ln_h).view(bsz, 1, layer.self_attn.num_kv_heads, layer.self_attn.head_dim)
                v = layer.self_attn.v_proj(ln_h).view(bsz, 1, layer.self_attn.num_kv_heads, layer.self_attn.head_dim)
                q = layer.self_attn.q_norm(q).transpose(1, 2)
                k = layer.self_attn.k_norm(k).transpose(1, 2)
                v = v.transpose(1, 2)
                q = (q * cos_s) + (_rotate_half(q) * sin_s)
                k = (k * cos_s) + (_rotate_half(k) * sin_s)
                k_cat = torch.cat([past_kv[idx_l][0], k], dim=2)
                v_cat = torch.cat([past_kv[idx_l][1], v], dim=2)
                new_past.append((k_cat, v_cat))
                attn_out = F.scaled_dot_product_attention(
                    q, k_cat, v_cat, attn_mask=step_mask, dropout_p=0.0, is_causal=False,
                    enable_gqa=(layer.self_attn.num_heads != layer.self_attn.num_kv_heads),
                )
                attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, 1, -1)
                h_s = h_s + layer.self_attn.o_proj(attn_out)
                h_s = h_s + layer.mlp(layer.post_attention_layernorm(h_s))
            past_kv = new_past
            cur_pos = cur_pos + 1
            return self.model.norm(h_s)[:, -1, :]

        should_tap = tee is not None and tee.enable_latent_tee
        halted = torch.zeros(bsz, dtype=torch.bool, device=device)
        for k_macro in range(1, steps_k + 1):
            for sub_idx in range(substeps_per_k):
                tap_now = should_tap and (sub_idx == substeps_per_k - 1)
                z_refined = adapter.step_hidden(
                    z_prev=z_curr.to(torch.float32),
                    step_k=k_macro,
                    tee=tee if tap_now else None,
                    should_tap=tap_now,
                )
                if adapter.halt_subspace_idx is not None:
                    top_sub = torch.argmax(adapter.subspace_scores(z_refined), dim=-1)
                    halted = halted | (top_sub == adapter.halt_subspace_idx)
                if bool(halted.all()) and not should_tap:
                    break
                e_cont = adapter.project_continuous_embed(z_refined).unsqueeze(1)
                z_step_out = _step_continuous_embed(e_cont)
                z_curr = torch.where(
                    halted.unsqueeze(-1), z_curr, z_step_out
                )
            if bool(halted.all()) and not should_tap:
                break

        # Transition to final scalar answer emission (`W_unembed` invoked strictly here)
        trig_list = list(answer_Trigger_ids)
        for tid in trig_list:
            tok_tensor = torch.full((bsz, 1), int(tid), dtype=torch.long, device=device)
            z_curr = _step_continuous_embed(self.model.embed_tokens(tok_tensor))

        emitted: list[list[int]] = [list(trig_list) for _ in range(bsz)]
        finished = [False] * bsz
        for _ in range(max_answer_tokens):
            if all(finished):
                break
            logits = self.unembed(z_curr.to(dtype))
            nxt = torch.argmax(logits, dim=-1, keepdim=True)
            for i in range(bsz):
                if finished[i]:
                    continue
                tid = int(nxt[i, 0].item())
                if eos_token_id is not None and tid == eos_token_id:
                    finished[i] = True
                else:
                    emitted[i].append(tid)
            z_curr = _step_continuous_embed(self.model.embed_tokens(nxt))

        if was_training:
            self.train()
        return emitted


class TorchContinuousLatentAdapter(nn.Module):
    """Differentiable continuous latent recurrence f_theta : R^d -> R^d in PyTorch.

    Executes z_k = normalize(z_{k-1} + alpha * f_theta(z_{k-1})) for k = 1..K and maps z_k
    through a compact grammar-free concept subspace C_theta in R^{R x d} (R << |V|)
    without invoking W_unembed or discrete argmax, optionally tapping detached snapshots
    into `RingBufferLatentTee` (fixed capacity M <= K, `enable_latent_tee=False` default).
    """

    def __init__(
        self,
        hidden_dim: int,
        expansion_ratio: float = 1.5,
        subspace_rank: int = 56,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.subspace_rank = subspace_rank
        inter_dim = max(64, int(hidden_dim * expansion_ratio))
        self.pre_norm = Qwen3RMSNorm(hidden_dim, eps=rms_norm_eps)
        self.ctx_norm = Qwen3RMSNorm(hidden_dim, eps=rms_norm_eps)
        self.post_norm = Qwen3RMSNorm(hidden_dim, eps=rms_norm_eps)
        self.progress_proj = nn.Linear(4, hidden_dim, bias=False)
        self.ctx_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.ctx_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.ctx_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.ctx_o = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gate_proj = nn.Linear(hidden_dim, inter_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, inter_dim, bias=False)
        self.down_proj = nn.Linear(inter_dim, hidden_dim, bias=False)
        self.residual_gate = nn.Parameter(torch.tensor(0.05))

        # Grammar-free continuous concept subspace C_theta in R^{R x d} (R << |V|)
        self.concept_query = nn.Parameter(
            torch.randn(subspace_rank, hidden_dim) / math.sqrt(hidden_dim),
            requires_grad=False,
        )
        self.concept_value = nn.Parameter(
            torch.randn(subspace_rank, hidden_dim) * 0.03,
            requires_grad=False,
        )
        self.subspace_temp = nn.Parameter(torch.tensor(0.30))
        self.halt_subspace_idx: int | None = None

        nn.init.normal_(self.progress_proj.weight, mean=0.0, std=0.005)
        nn.init.normal_(self.ctx_q.weight, mean=0.0, std=1.0 / math.sqrt(hidden_dim))
        nn.init.normal_(self.ctx_k.weight, mean=0.0, std=1.0 / math.sqrt(hidden_dim))
        nn.init.normal_(self.ctx_v.weight, mean=0.0, std=1.0 / math.sqrt(hidden_dim))
        nn.init.zeros_(self.ctx_o.weight)
        nn.init.normal_(self.gate_proj.weight, mean=0.0, std=1.0 / math.sqrt(hidden_dim))
        nn.init.normal_(self.up_proj.weight, mean=0.0, std=1.0 / math.sqrt(hidden_dim))
        nn.init.normal_(self.down_proj.weight, mean=0.0, std=0.005 / math.sqrt(hidden_dim))

    @torch.no_grad()
    def init_from_symbolic_subspace(
        self,
        lm_head_weight: torch.Tensor,
        embed_weight: torch.Tensor,
        basis_token_ids: Sequence[int],
        halt_subspace_idx: int | None = 0,
    ) -> None:
        """Seed C_theta in R^{R x d} from symbolic/numeric basis vectors (zero grammar words)."""
        chosen = list(basis_token_ids)[: self.subspace_rank]
        if not chosen:
            return
        while len(chosen) < self.subspace_rank:
            chosen.append(chosen[len(chosen) % len(basis_token_ids)])
        idx_t = torch.tensor(chosen, dtype=torch.long, device=lm_head_weight.device)
        self.concept_query.copy_(
            lm_head_weight.detach()[idx_t].to(
                device=self.concept_query.device, dtype=self.concept_query.dtype
            )
        )
        self.concept_value.copy_(
            embed_weight.detach()[idx_t].to(
                device=self.concept_value.device, dtype=self.concept_value.dtype
            )
        )
        self.halt_subspace_idx = halt_subspace_idx

    def subspace_scores(self, z_state: torch.Tensor) -> torch.Tensor:
        """Compute continuous concept similarity logits in R^R (R << |V|)."""
        z_f = z_state.to(self.concept_query.dtype)
        temp = self.subspace_temp.clamp(min=0.15, max=2.0)
        return torch.matmul(z_f, self.concept_query.t()) / temp

    def project_continuous_embed(self, z_state: torch.Tensor) -> torch.Tensor:
        """Form a smooth continuous superposition vector e_cont in R^d without argmax or W_unembed."""
        scores = self.subspace_scores(z_state)
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, self.concept_value)

    def step_hidden(
        self,
        z_prev: torch.Tensor,
        step_k: int,
        tee: RingBufferLatentTee | None = None,
        should_tap: bool = False,
        k_mat: torch.Tensor | None = None,
        v_mat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Execute one geometry-preserving continuous transition in R^d."""
        prog = min(float(step_k), 32.0) / 20.0
        prog_feat = torch.tensor(
            [
                prog,
                1.0 - prog,
                math.sin(math.pi * prog),
                math.cos(math.pi * prog),
            ],
            device=z_prev.device,
            dtype=z_prev.dtype,
        )
        conditioned = self.pre_norm(z_prev + self.progress_proj(prog_feat))

        if k_mat is not None and v_mat is not None:
            q_vec = self.ctx_q(conditioned).unsqueeze(1)
            attn_scores = torch.matmul(q_vec, k_mat.transpose(-2, -1)) / math.sqrt(
                self.hidden_dim
            )
            attn_probs = torch.softmax(attn_scores, dim=-1)
            ctx_read = self.ctx_o(torch.matmul(attn_probs, v_mat).squeeze(1))
            conditioned = self.pre_norm(conditioned + ctx_read)

        delta = self.down_proj(
            F.silu(self.gate_proj(conditioned)) * self.up_proj(conditioned)
        )
        raw_next = z_prev + self.residual_gate * self.post_norm(delta)
        base_norm = z_prev.to(torch.float32).norm(dim=-1, keepdim=True).clamp(min=1e-5)
        next_norm = raw_next.to(torch.float32).norm(dim=-1, keepdim=True).clamp(min=1e-5)
        z_k = raw_next * (base_norm / next_norm).to(z_prev.dtype)

        if should_tap and tee is not None:
            z_tap = z_k[0] if z_k.ndim == 2 else z_k
            tee.push(z_k=z_tap.to(torch.float32), step_k=step_k)

        return z_k

    def forward(
        self,
        z_init: torch.Tensor,
        steps_k: int,
        tee: RingBufferLatentTee | None = None,
        enable_latent_tee: bool | None = None,
        context_hidden: torch.Tensor | None = None,
        return_trajectory: bool = True,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run K continuous latent reasoning transitions starting from z_0 = h_t^{(L)}.

        Returns `(z_K, trajectory)` where `z_K` is bit-for-bit identical whether
        `enable_latent_tee` is `False` (production default) or `True` (audit mode).
        """
        if steps_k < 0:
            raise ValueError("steps_k must be >= 0")

        should_tap = tee is not None and (
            tee.enable_latent_tee if enable_latent_tee is None else enable_latent_tee
        )

        k_mat: torch.Tensor | None = None
        v_mat: torch.Tensor | None = None
        if context_hidden is not None:
            ctx = self.ctx_norm(context_hidden.to(z_init.dtype))
            k_mat = self.ctx_k(ctx)
            v_mat = self.ctx_v(ctx)

        z_k = z_init
        trajectory: list[torch.Tensor] = []

        for k in range(1, steps_k + 1):
            z_k = self.step_hidden(
                z_prev=z_k,
                step_k=k,
                tee=tee,
                should_tap=should_tap,
                k_mat=k_mat,
                v_mat=v_mat,
            )
            if return_trajectory:
                trajectory.append(z_k)

        return z_k, trajectory


class TorchSidecarProbe(nn.Module):
    """Detached PyTorch Sidecar Probe g_phi(sg[z_k]) with isolated optimizer invariant.

    All forward and loss calls detach incoming tensors via `.detach()` (`sg[z_k]`) and
    assert `not z_det.requires_grad` so probe supervision never backpropagates into
    `Qwen3CausalLM` or `TorchContinuousLatentAdapter`.
    """

    def __init__(
        self,
        hidden_dim: int,
        templates: Sequence[str],
        rms_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if len(templates) == 0:
            raise ValueError("templates must be non-empty")
        self.templates = list(templates)
        self.norm = Qwen3RMSNorm(hidden_dim, eps=rms_norm_eps)
        self.classifier = nn.Linear(hidden_dim, len(self.templates), bias=True)
        self.optimizer_steps: int = 0

    def forward_detached(self, z_states: torch.Tensor) -> torch.Tensor:
        """Compute probe class logits strictly on stop-gradient states sg[z]."""
        z_det = z_states.detach()
        if z_det.requires_grad or z_det.grad_fn is not None:
            raise RuntimeError("Stop-gradient invariant violated in TorchSidecarProbe")
        z_cast = z_det.to(
            device=self.classifier.weight.device,
            dtype=self.classifier.weight.dtype,
        )
        return self.classifier(self.norm(z_cast)).to(torch.float32)

    def compute_loss(
        self,
        z_states: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute L_probe(phi; sg[z_{1:K}], y) with zero autograd link to backbone."""
        logits = self.forward_detached(z_states)
        return F.cross_entropy(logits, target_indices.to(logits.device))

    def step_disjoint_optimizer(
        self,
        z_states: torch.Tensor,
        target_indices: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """Run one isolated optimizer step strictly over probe parameters phi."""
        optimizer.zero_grad(set_to_none=True)
        loss = self.compute_loss(z_states, target_indices)
        loss.backward()
        optimizer.step()
        self.optimizer_steps += 1
        return float(loss.detach().cpu().item())

    @torch.no_grad()
    def decode_numpy_state(self, z_vec: np.ndarray, step_k: int) -> str:
        """Callable adapter compatible with `RingBufferLatentTee.flush(...)`."""
        was_training = self.training
        self.eval()
        tensor_in = torch.from_numpy(np.asarray(z_vec, dtype=np.float32)).unsqueeze(0)
        logits = self.forward_detached(tensor_in)
        best_idx = int(torch.argmax(logits[0]).item())
        if was_training:
            self.train()
        return self.templates[best_idx].replace("{step_k}", str(step_k))


def flush_qwen3_tee_with_unembed(
    tee: RingBufferLatentTee,
    lm_head_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    id_to_token: Callable[[int], str] | Sequence[str],
    sidecar_probe: TorchSidecarProbe,
    candidate_token_ids: Sequence[int] | None = None,
    top_k: int = 5,
    temperature: float = 2.0,
    already_normalized: bool = False,
) -> list[LatentStepRecord]:
    """Flush `RingBufferLatentTee` against real `Qwen3CausalLM` unembedding weights.

    Projects buffered detached states `sg[z_k]` through `W_unembed` (optionally scoped to
    candidate domain vocabulary IDs or full 151,936 vocabulary) and `TorchSidecarProbe`.
    """
    peeked = list(tee._buffer)
    tee.clear()
    if not peeked:
        return []

    w_full = lm_head_weight.detach().to(device="cpu", dtype=torch.float32)
    rms_w = norm_weight.detach().to(device="cpu", dtype=torch.float32)

    if candidate_token_ids is not None:
        cand_ids = list(candidate_token_ids)
        w_mat = w_full[cand_ids]  # (V_sub, d)
    else:
        cand_ids = None
        w_mat = w_full  # (|V|, d)

    records: list[LatentStepRecord] = []
    k_eff = min(top_k, w_mat.shape[0])

    for step_k, l2_norm, z_detached in peeked:
        z_t = torch.from_numpy(np.array(z_detached, dtype=np.float32, copy=True))
        z_proj = (
            z_t
            if already_normalized
            else (z_t * torch.rsqrt(z_t.pow(2).mean() + 1e-6) * rms_w)
        )
        raw_logits = torch.mv(w_mat, z_proj)
        if candidate_token_ids is not None and raw_logits.numel() >= 3:
            std = raw_logits.std().clamp(min=1e-5)
            logits = ((raw_logits - raw_logits.mean()) / std) / max(temperature, 1e-5)
        else:
            logits = raw_logits / max(temperature, 1e-5)
        probs = torch.softmax(logits, dim=-1)
        top_vals, top_pos = torch.topk(probs, k=k_eff)

        preds: list[LogitLensPrediction] = []
        for prob_val, pos_idx in zip(top_vals.tolist(), top_pos.tolist(), strict=True):
            tok_id = cand_ids[pos_idx] if cand_ids is not None else pos_idx
            tok_str = (
                id_to_token(tok_id)
                if callable(id_to_token)
                else id_to_token[tok_id]
            )
            preds.append(
                LogitLensPrediction(
                    token=tok_str,
                    prob=round(float(prob_val), 4),
                )
            )

        decoded = sidecar_probe.decode_numpy_state(z_detached, step_k=step_k)
        records.append(
            LatentStepRecord(
                step_k=step_k,
                l2_norm=round(float(l2_norm), 3),
                logit_lens_top_k=preds,
                sidecar_decoded=decoded,
            )
        )

    return records


def total_parameter_grad_norm(params: Iterable[torch.nn.Parameter]) -> float:
    """Compute global L2 norm of populated `.grad` tensors across `params`."""
    sq_sum = 0.0
    for p in params:
        if p.grad is not None:
            sq_sum += float(p.grad.detach().to(torch.float32).pow(2).sum().cpu().item())
    return math.sqrt(sq_sum)
