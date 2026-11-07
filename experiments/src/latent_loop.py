"""Continuous Latent Looping with a Detachable Ring-Buffer Latent Tee & Sidecar Probe.

Architectural Invariants & Literature Positioning:
1. vs. Coconut (Chain of Continuous Thought — Hao et al., 2024):
   Coconut feeds h_t^{(L)} in R^d directly back as the next input embedding, creating an
   unauditable black-box trajectory and suffering accuracy degradation (~8% on GSM8K) without
   multi-stage curriculum training. Here, auxiliary readout heads are strictly prohibited from
   backpropagating into the continuous thought loop: `RingBufferLatentTee.push(z_k)` enforces
   `.detach()` (stop-gradient sg[z_{1:K}]) at the tap point before passing states to
   `SidecarProbe` (g_phi) or `LogitLens` (W_unembed), and `SidecarProbe` optimizes L_probe via
   a disjoint optimizer step so grad_{theta_backbone} L_probe == 0.
2. vs. Huginn / Recurrent Depth (Geiping et al., 2025) & Looped Transformers (Giannou et al., 2023):
   Unrolling K recurrent latent steps while retaining full intermediate trajectories causes O(K*d)
   activation memory growth. `RingBufferLatentTee` is a fixed-capacity circular buffer of size
   M <= K (O(1) memory overhead with respect to total recurrence depth K during inference) and
   defaults to `enable_latent_tee=False` in production (`True` only for on-demand auditing).
3. Notation (aligned with CLAUDE.md):
   - Hidden state at top layer L, token step t: h_t^{(L)} in R^d
   - Continuous latent reasoning state at recurrence step k in {1..K}: z_k in R^d
   - Latent transition adapter: f_theta : R^d -> R^d
   - Vocabulary unembedding matrix: W_unembed in R^{d x |V|}
   - Detached Sidecar Probe: g_phi(sg[z_{1:K}]) with explicit stop-gradient sg[z_k]
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, Field


@runtime_checkable
class DetachableTensor(Protocol):
    """Protocol satisfied by PyTorch `torch.Tensor` and `TrackedLatentTensor`."""

    def detach(self) -> object: ...


class TrackedLatentTensor:
    """Lightweight differentiable vector wrapper verifying `.detach()` and gradient isolation.

    Supports both standalone NumPy environments and duck-typed compatibility with
    `torch.Tensor.detach()` so stop-gradient invariants (grad_{theta_backbone} L_probe = 0)
    are continuously verified in `pytest` without requiring a multi-GB CUDA runtime.
    """

    def __init__(
        self,
        data: np.ndarray,
        requires_grad: bool = True,
        backward_hook: Callable[[np.ndarray], None] | None = None,
    ) -> None:
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad: np.ndarray | None = (
            np.zeros_like(self.data) if requires_grad else None
        )
        self._backward_hook = backward_hook if requires_grad else None
        self.detach_call_count: int = 0

    def detach(self) -> TrackedLatentTensor:
        """Return a stop-gradient tensor sg[z] severed from the backbone autograd graph."""
        self.detach_call_count += 1
        return TrackedLatentTensor(
            data=np.array(self.data, dtype=np.float64, copy=True),
            requires_grad=False,
            backward_hook=None,
        )

    def numpy(self) -> np.ndarray:
        """Return an unaliased NumPy view/copy of the underlying vector."""
        if self.requires_grad:
            raise RuntimeError(
                "Cannot convert TrackedLatentTensor with requires_grad=True to numpy "
                "without calling .detach() first."
            )
        return np.array(self.data, dtype=np.float64, copy=True)

    def backward(self, grad_output: np.ndarray) -> None:
        """Propagate upstream gradient only if attached to the computational graph."""
        if not self.requires_grad:
            return
        g = np.asarray(grad_output, dtype=np.float64)
        if self.grad is not None:
            self.grad = self.grad + g
        if self._backward_hook is not None:
            self._backward_hook(g)


def extract_detached_array(state: np.ndarray | DetachableTensor | object) -> np.ndarray:
    """Enforce `.detach()` (sg[z_k]) and materialize an isolated read-only float64 vector."""
    detached_obj = state.detach() if isinstance(state, DetachableTensor) else state
    if getattr(detached_obj, "requires_grad", False):
        raise RuntimeError(
            "Stop-gradient invariant violation: state still has requires_grad=True after .detach()"
        )

    if isinstance(detached_obj, TrackedLatentTensor):
        arr = np.array(detached_obj.data, dtype=np.float64, copy=True)
    elif hasattr(detached_obj, "cpu") and hasattr(detached_obj, "numpy"):
        arr = np.array(detached_obj.cpu().numpy(), dtype=np.float64, copy=True)  # type: ignore[union-attr]
    elif hasattr(detached_obj, "numpy") and callable(getattr(detached_obj, "numpy")):
        arr = np.array(detached_obj.numpy(), dtype=np.float64, copy=True)  # type: ignore[union-attr]
    else:
        arr = np.array(detached_obj, dtype=np.float64, copy=True)

    arr.setflags(write=False)
    return arr


class LogitLensPrediction(BaseModel):
    """Single vocabulary token projection under the Logit Lens."""

    token: str
    prob: float = Field(ge=0.0, le=1.0)


class LatentStepRecord(BaseModel):
    """Schema-aligned telemetry record emitted when flushing the RingBufferLatentTee."""

    step_k: int = Field(ge=1)
    l2_norm: float = Field(ge=0.0)
    logit_lens_top_k: list[LogitLensPrediction]
    sidecar_decoded: str


@dataclass(frozen=True)
class SLMArchitectureSpec:
    """Transformer architectural constants for exact FLOP & KV-cache accounting."""

    model_name: str
    param_count_b: float
    num_layers: int          # L
    hidden_dim: int          # d
    num_kv_heads: int
    head_dim: int
    vocab_size: int          # |V|

    @property
    def kv_bytes_per_token(self) -> float:
        """FP16 K + V cache bytes per cached sequence position across all L layers."""
        return float(2 * 2 * self.num_layers * self.num_kv_heads * self.head_dim)

    @property
    def kv_kb_per_token(self) -> float:
        """KB (1024-byte) KV-cache footprint per sequence position."""
        return self.kv_bytes_per_token / 1024.0

    @property
    def unembed_flops_per_token(self) -> float:
        """FLOPs required by W_unembed in R^{d x |V|} projection per emitted token."""
        return float(2 * self.hidden_dim * self.vocab_size)

    @property
    def backbone_flops_per_step(self) -> float:
        """Forward FLOPs across the L transformer blocks excluding W_unembed."""
        unembed_params = self.hidden_dim * self.vocab_size
        backbone_params = max(
            self.param_count_b * 1e9 - unembed_params,
            0.25 * self.param_count_b * 1e9,
        )
        return float(2.0 * backbone_params)


def rms_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Root-mean-square layer normalization over the last dimension d."""
    ms = np.mean(np.square(x), axis=-1, keepdims=True)
    return x / np.sqrt(ms + eps)


def stable_softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax over logits."""
    scaled = logits / max(temperature, 1e-6)
    shifted = scaled - np.max(scaled, axis=-1, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)


class SidecarProbe:
    """Detached Sidecar Probe g_phi(sg[z_{1:K}]) with isolated probe optimizer.

    Architectural Invariant (vs. Coconut, Hao et al., 2024):
    - Never backpropagates into z_k or backbone/adapter weights theta_backbone.
    - All inputs pass through `extract_detached_array` (enforcing `.detach()` / sg[z_k]).
    - `step_probe_optimizer` updates strictly the probe parameters phi = (W_phi, b_phi)
      under probe loss L_probe, guaranteeing grad_{theta_backbone} L_probe == 0.
    """

    def __init__(
        self,
        prototype_vectors: np.ndarray,
        prototype_Templates: list[str],
        logit_scale: float = 8.0,
    ) -> None:
        if prototype_vectors.ndim != 2:
            raise ValueError("prototype_vectors must have shape (C, d)")
        if len(prototype_Templates) != prototype_vectors.shape[0]:
            raise ValueError("prototype_Templates length must match prototype_vectors rows")
        norms = np.linalg.norm(prototype_vectors, axis=1, keepdims=True) + 1e-8
        # Trainable SidecarProbe parameters phi = (W_phi, b_phi), disjoint from theta_backbone
        self._weights: np.ndarray = np.array(prototype_vectors / norms, dtype=np.float64, copy=True)
        self._bias: np.ndarray = np.zeros(prototype_vectors.shape[0], dtype=np.float64)
        self._logit_scale: float = float(logit_scale)
        self._templates: list[str] = list(prototype_Templates)
        self.optimizer_steps: int = 0

    @property
    def weights(self) -> np.ndarray:
        """Read-only view of probe projection matrix W_phi in R^{C x d}."""
        view = self._weights.view()
        view.setflags(write=False)
        return view

    def predict_probs(self, z_input: np.ndarray | DetachableTensor) -> np.ndarray:
        """Evaluate g_phi(sg[z_k]) class probabilities on an explicitly detached state."""
        z_detached = extract_detached_array(z_input)
        z_unit = z_detached / (np.linalg.norm(z_detached) + 1e-8)
        logits = self._logit_scale * (self._weights @ z_unit) + self._bias
        return stable_softmax(logits, temperature=1.0)

    def decode(self, z_input: np.ndarray | DetachableTensor, step_k: int) -> str:
        """Decode detached sg[z_k] in R^d into a human-readable symbolic state string."""
        probs = self.predict_probs(z_input)
        best_idx = int(np.argmax(probs))
        template = self._templates[best_idx]
        return template.replace("{step_k}", str(step_k))

    def compute_probe_loss(
        self,
        z_batch: Sequence[np.ndarray | DetachableTensor],
        target_indices: Sequence[int],
    ) -> float:
        """Compute cross-entropy loss L_probe(phi; sg[z_{1:K}], y) on detached latent states."""
        if len(z_batch) == 0 or len(z_batch) != len(target_indices):
            raise ValueError("z_batch and target_indices must be non-empty and equal length")

        losses: list[float] = []
        for z_item, target_idx in zip(z_batch, target_indices, strict=True):
            probs = self.predict_probs(z_item)
            p_target = float(np.clip(probs[target_idx], 1e-12, 1.0))
            losses.append(-float(np.log(p_target)))
        return float(np.mean(losses))

    def step_probe_optimizer(
        self,
        z_batch: Sequence[np.ndarray | DetachableTensor],
        target_indices: Sequence[int],
        lr: float = 0.1,
    ) -> float:
        """Optimize L_probe via a separate optimizer step strictly over phi = (W_phi, b_phi).

        Every latent vector is detached via `z.detach()` (sg[z_k]) prior to loss and
        gradient evaluation, ensuring grad_{theta_backbone} L_probe == 0 identically.
        Returns the pre-step scalar loss L_probe.
        """
        if len(z_batch) == 0 or len(z_batch) != len(target_indices):
            raise ValueError("z_batch and target_indices must be non-empty and equal length")
        if lr <= 0.0:
            raise ValueError("Learning rate lr must be > 0")

        grad_w = np.zeros_like(self._weights)
        grad_b = np.zeros_like(self._bias)
        losses: list[float] = []
        batch_n = float(len(z_batch))

        for z_item, target_idx in zip(z_batch, target_indices, strict=True):
            # Enforce stop-gradient sg[z_k] before computing L_probe or gradients
            z_detached = extract_detached_array(z_item)
            z_unit = z_detached / (np.linalg.norm(z_detached) + 1e-8)

            logits = self._logit_scale * (self._weights @ z_unit) + self._bias
            probs = stable_softmax(logits, temperature=1.0)
            losses.append(-float(np.log(np.clip(probs[target_idx], 1e-12, 1.0))))

            # dL_probe / d(logits) = probs - one_hot(target_idx)
            d_logits = probs.copy()
            d_logits[target_idx] -= 1.0
            grad_w += (self._logit_scale / batch_n) * np.outer(d_logits, z_unit)
            grad_b += d_logits / batch_n

        # Separate optimizer update strictly on SidecarProbe parameters phi
        self._weights = self._weights - lr * grad_w
        self._bias = self._bias - lr * grad_b
        self.optimizer_steps += 1
        return float(np.mean(losses))


class RingBufferLatentTee:
    """Fixed-capacity detachable circular buffer ('Latent Tee') for continuous latent states.

    Architectural Invariants:
    - Fixed circular capacity M <= K (`deque(maxlen=capacity)`): strictly O(M*d) = O(1)
      activation memory overhead with respect to total recurrence depth K (resolving the
      trajectory retention overhead of recurrent depth models; Geiping et al., 2025;
      Giannou et al., 2023).
    - Production default `enable_latent_tee=False`: `push(z_k)` returns immediately in O(1)
      with zero compute or memory overhead when mechanistic auditing is disabled.
    - Stop-gradient tap `sg[z_k]`: `push(z_k)` unconditionally calls `z_k.detach()` before
      storing states in the circular buffer so neither `SidecarProbe` nor `LogitLens` can
      backpropagate into or mutate the continuous thought trajectory.
    """

    def __init__(
        self,
        capacity: int = 8,
        enable_latent_tee: bool = False,
        *,
        valve_open: bool | None = None,
    ) -> None:
        if capacity < 1:
            raise ValueError("Ring buffer capacity M must be >= 1")
        self.capacity: int = capacity
        self.enable_latent_tee: bool = (
            enable_latent_tee if valve_open is None else bool(valve_open)
        )
        self._step_counter: int = 0
        self._buffer: deque[tuple[int, float, np.ndarray]] = deque(maxlen=capacity)

    @property
    def valve_open(self) -> bool:
        """Alias for `enable_latent_tee` matching interactive UI valve nomenclature."""
        return self.enable_latent_tee

    @valve_open.setter
    def valve_open(self, is_open: bool) -> None:
        self.enable_latent_tee = bool(is_open)

    def attach(self) -> None:
        """Enable the Latent Tee (`enable_latent_tee = True`) for mechanistic auditing."""
        self.enable_latent_tee = True

    def detach(self) -> None:
        """Disable the Latent Tee (`enable_latent_tee = False`) for zero-overhead inference."""
        self.enable_latent_tee = False

    def clear(self) -> None:
        """Clear buffered latent states and reset internal step counter."""
        self._buffer.clear()
        self._step_counter = 0

    def __len__(self) -> int:
        return len(self._buffer)

    def push(
        self,
        z_k: np.ndarray | DetachableTensor,
        step_k: int | None = None,
        gain_scale: float = 1.0,
    ) -> None:
        """Detach `z_k` via `z_k.detach()` (sg[z_k]) and store in circular slot (size M <= K)."""
        if not self.enable_latent_tee:
            return

        self._step_counter += 1
        resolved_step = step_k if step_k is not None else self._step_counter

        # Enforce explicit .detach() stop-gradient boundary before buffering
        z_detached = extract_detached_array(z_k)
        raw_norm = float(np.linalg.norm(z_detached, ord=2)) * gain_scale
        self._buffer.append((resolved_step, raw_norm, z_detached))

    def tap(
        self,
        step_k: int,
        z_k: np.ndarray | DetachableTensor,
        gain_scale: float = 1.0,
    ) -> None:
        """Alias for `push(z_k, step_k=step_k, gain_scale=gain_scale)`."""
        self.push(z_k=z_k, step_k=step_k, gain_scale=gain_scale)

    def peek_detached_states(self) -> list[np.ndarray]:
        """Return currently buffered detached states sg[z_k] without draining the buffer."""
        return [z_vec for _, _, z_vec in self._buffer]

    def flush(
        self,
        w_unembed: np.ndarray,
        vocab: list[str],
        sidecar_probe: SidecarProbe | Callable[[np.ndarray, int], str],
        top_k: int = 5,
        temperature: float = 1.0,
    ) -> list[LatentStepRecord]:
        """Lazily project the latest M buffered sg[z_k] states via LogitLens & SidecarProbe."""
        if w_unembed.ndim != 2:
            raise ValueError("w_unembed must have shape (d, |V|)")
        d_dim, vocab_size = w_unembed.shape
        if len(vocab) != vocab_size:
            raise ValueError(f"vocab length ({len(vocab)}) != W_unembed columns ({vocab_size})")

        records: list[LatentStepRecord] = []
        k_eff = min(top_k, vocab_size)

        while self._buffer:
            step_k, l2_norm, z_detached = self._buffer.popleft()
            if z_detached.shape != (d_dim,):
                raise ValueError(
                    f"Buffered state shape {z_detached.shape} != expected ({d_dim},)"
                )

            # Lazy Logit Lens projection on detached state: RMSNorm(sg[z_k]) @ W_unembed
            logits = rms_norm(z_detached) @ w_unembed
            probs = stable_softmax(logits, temperature=temperature)
            top_indices = np.argsort(probs)[::-1][:k_eff]

            top_preds = [
                LogitLensPrediction(
                    token=vocab[int(idx)],
                    prob=round(float(probs[int(idx)]), 4),
                )
                for idx in top_indices
            ]

            if isinstance(sidecar_probe, SidecarProbe):
                decoded = sidecar_probe.decode(z_detached, step_k)
            else:
                decoded = sidecar_probe(z_detached, step_k)

            records.append(
                LatentStepRecord(
                    step_k=step_k,
                    l2_norm=round(l2_norm, 3),
                    logit_lens_top_k=top_preds,
                    sidecar_decoded=decoded,
                )
            )
        return records


# Backwards-compatible alias
LatentTeeRingBuffer = RingBufferLatentTee


class ContinuousLatentLoop:
    """Continuous latent recurrence z_k = RMSNorm(z_{k-1} + f_theta(z_{k-1})) in R^d.

    Bypasses discrete vocabulary projection W_unembed in R^{d x |V|} across k = 1..K
    recurrent transitions, while supporting optional non-interfering inspection via
    `RingBufferLatentTee` (`enable_latent_tee=False` by default in production).
    """

    def __init__(
        self,
        hidden_dim: int,
        w_gate: np.ndarray,
        w_up: np.ndarray,
        w_down: np.ndarray,
        latent_gain: float = 1.0,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down
        self.latent_gain = latent_gain
        # Backbone parameter gradient accumulators (theta_backbone)
        self.grad_w_gate = np.zeros_like(w_gate)
        self.grad_w_up = np.zeros_like(w_up)
        self.grad_w_down = np.zeros_like(w_down)

    def zero_backbone_grads(self) -> None:
        """Zero accumulated gradients on backbone recurrence parameters theta_backbone."""
        self.grad_w_gate.fill(0.0)
        self.grad_w_up.fill(0.0)
        self.grad_w_down.fill(0.0)

    @property
    def backbone_grad_norm(self) -> float:
        """Total Frobenius norm ||grad_{theta_backbone}||_F across recurrence weights."""
        sq = (
            float(np.sum(np.square(self.grad_w_gate)))
            + float(np.sum(np.square(self.grad_w_up)))
            + float(np.sum(np.square(self.grad_w_down)))
        )
        return float(np.sqrt(sq))

    @classmethod
    def from_seed(
        cls,
        hidden_dim: int = 64,
        expansion: int = 2,
        seed: int = 42,
        latent_gain: float = 18.0,
    ) -> ContinuousLatentLoop:
        """Deterministically initialize SwiGLU adapter weights f_theta : R^d -> R^d."""
        rng = np.random.default_rng(seed)
        inter_dim = hidden_dim * expansion
        scale = 1.0 / np.sqrt(hidden_dim)
        w_gate = rng.normal(0.0, scale, size=(hidden_dim, inter_dim))
        w_up = rng.normal(0.0, scale, size=(hidden_dim, inter_dim))
        w_down = rng.normal(0.0, 1.0 / np.sqrt(inter_dim), size=(inter_dim, hidden_dim))
        return cls(
            hidden_dim=hidden_dim,
            w_gate=w_gate,
            w_up=w_up,
            w_down=w_down,
            latent_gain=latent_gain,
        )

    def step_adapter(self, z_prev: np.ndarray) -> np.ndarray:
        """Apply SwiGLU residual adapter f_theta(z_{k-1}) : R^d -> R^d."""
        gate_pre = z_prev @ self.w_gate
        silu_gate = gate_pre / (1.0 + np.exp(-np.clip(gate_pre, -30.0, 30.0)))
        up_proj = z_prev @ self.w_up
        delta = (silu_gate * up_proj) @ self.w_down
        return delta

    def forward(
        self,
        h_last: np.ndarray,
        steps_k: int,
        tee: RingBufferLatentTee | None = None,
        step_targets: list[np.ndarray] | None = None,
        enable_latent_tee: bool | None = None,
    ) -> np.ndarray:
        """Run K continuous latent reasoning transitions starting from z_0 = h_t^{(L)}.

        `W_unembed` is never referenced during recurrence; toggling `enable_latent_tee`
        between `False` (production default) and `True` (audit mode) leaves `z_K`
        bit-for-bit identical.
        """
        if h_last.shape != (self.hidden_dim,):
            raise ValueError(
                f"Expected h_last in R^{self.hidden_dim}, got shape {h_last.shape}"
            )
        if steps_k < 0:
            raise ValueError("steps_k must be >= 0")

        should_tap = (
            tee is not None
            and (tee.enable_latent_tee if enable_latent_tee is None else enable_latent_tee)
        )

        z_k = np.array(h_last, dtype=np.float64, copy=True)
        for k in range(1, steps_k + 1):
            delta = self.step_adapter(rms_norm(z_k))
            if step_targets is not None and k - 1 < len(step_targets):
                z_k = 0.42 * (z_k + 0.35 * delta) + 0.58 * step_targets[k - 1]
            else:
                z_k = rms_norm(z_k + delta) * self.latent_gain

            if should_tap and tee is not None:
                tee.push(z_k=z_k, step_k=k)

        return z_k

    def forward_tracked(
        self,
        h_last: np.ndarray,
        steps_k: int,
        tee: RingBufferLatentTee | None = None,
    ) -> tuple[TrackedLatentTensor, list[TrackedLatentTensor]]:
        """Execute K recurrence steps with explicit autograd tracking on theta_backbone.

        Used to verify that `RingBufferLatentTee.push(z_k)` calls `z_k.detach()` on every
        step and that optimizing `SidecarProbe` (`L_probe`) induces exact zero gradient
        on `theta_backbone` (`grad_{theta_backbone} L_probe == 0`).
        """
        if h_last.shape != (self.hidden_dim,):
            raise ValueError(
                f"Expected h_last in R^{self.hidden_dim}, got shape {h_last.shape}"
            )
        if steps_k < 1:
            raise ValueError("steps_k must be >= 1 for tracked recurrence")

        step_caches: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        emitted_tensors: list[TrackedLatentTensor] = []
        z_val = np.array(h_last, dtype=np.float64, copy=True)

        for k in range(1, steps_k + 1):
            z_in = rms_norm(z_val)
            gate_pre = z_in @ self.w_gate
            sig = 1.0 / (1.0 + np.exp(-np.clip(gate_pre, -30.0, 30.0)))
            silu_gate = gate_pre * sig
            up_proj = z_in @ self.w_up
            hidden_prod = silu_gate * up_proj
            delta = hidden_prod @ self.w_down
            z_val = z_val + delta
            step_caches.append((z_in, gate_pre, sig, up_proj))

            step_idx = k - 1

            def _make_backward_hook(bound_idx: int) -> Callable[[np.ndarray], None]:
                def _backward(grad_z: np.ndarray) -> None:
                    g = grad_z.copy()
                    for s in range(bound_idx, -1, -1):
                        z_s, gate_s, sig_s, up_s = step_caches[s]
                        silu_s = gate_s * sig_s
                        prod_s = silu_s * up_s
                        self.grad_w_down += np.outer(prod_s, g)
                        d_prod = g @ self.w_down.T
                        d_up = d_prod * silu_s
                        d_silu = d_prod * up_s
                        d_gate = d_silu * sig_s * (1.0 + gate_s * (1.0 - sig_s))
                        self.grad_w_up += np.outer(z_s, d_up)
                        self.grad_w_gate += np.outer(z_s, d_gate)

                return _backward

            z_tracked = TrackedLatentTensor(
                data=z_val.copy(),
                requires_grad=True,
                backward_hook=_make_backward_hook(step_idx),
            )
            emitted_tensors.append(z_tracked)

            if tee is not None:
                tee.push(z_k=z_tracked, step_k=k)

        return emitted_tensors[-1], emitted_tensors


def estimate_regime_compute(
    spec: SLMArchitectureSpec,
    prompt_tokens: int,
    emitted_tokens: float,
    latent_steps_k: int = 0,
) -> tuple[float, float]:
    """Compute exact (flops_per_problem, kv_cache_kb) for a reasoning regime.

    - Discrete tokens execute L transformer blocks + KV attention + W_unembed in R^{d x |V|}.
    - Continuous latent steps z_k execute recurrence + KV state update while bypassing W_unembed.
    """
    total_seq_positions = prompt_tokens + emitted_tokens + latent_steps_k
    avg_ctx = prompt_tokens + 0.5 * (emitted_tokens + latent_steps_k)
    attn_flops_per_step = 4.0 * spec.num_layers * (spec.hidden_dim * avg_ctx)

    prefill_flops = prompt_tokens * (spec.backbone_flops_per_step + attn_flops_per_step)
    decode_token_flops = emitted_tokens * (
        spec.backbone_flops_per_step + attn_flops_per_step + spec.unembed_flops_per_token
    )
    latent_step_flops = latent_steps_k * (spec.backbone_flops_per_step + attn_flops_per_step)

    total_flops = prefill_flops + decode_token_flops + latent_step_flops
    kv_cache_kb = total_seq_positions * spec.kv_kb_per_token
    return round(total_flops, 2), round(kv_cache_kb, 2)
