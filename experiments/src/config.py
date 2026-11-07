"""Experiment regime configurations and architectural invariants for GITLTYN.

Enforces explicit positioning and runtime invariants across the four evaluation regimes:
1. `verbose_cot` (Baseline): Unconstrained grammatical Chain-of-Thought.
2. `telegraphic_cot` (Motivational Ablation A): Syntactically pruned CoT proving that
   English grammatical glue accounts for 64.6%–67.1% of emitted tokens with <1% accuracy
   impact. Strictly an empirical control, NOT the proposed architecture (contra TokenSkip,
   CCoT, and Sketch-of-Thought).
3. `compute_matched` / `budget_forced_cot` (Motivational Ablation B): Verbose CoT forced
   into the exact token/FLOP budget of `telegraphic_cot`.
4. `latent_loop` (Proposed Architecture, Ablation C): Pure R^d Continuous Latent Looping
   paired with a fixed-capacity circular `RingBufferLatentTee` (M <= K, O(1) memory w.r.t.
   depth K, `enable_latent_tee=False` by default in production; contra Geiping et al., 2025
   and Giannou et al., 2023) and a stop-gradient `SidecarProbe` g_phi(sg[z_{1:K}]) optimized
   under a disjoint optimizer step so grad_{theta_backbone} L_probe = 0 (contra Coconut,
   Hao et al., 2024).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

RegimeKey = Literal["verbose_cot", "telegraphic_cot", "compute_matched", "latent_loop"]
RegimeRole = Literal[
    "baseline",
    "motivational_ablation_control",
    "motivational_budget_control",
    "proposed_architecture",
]


class LatentTeeConfig(BaseModel):
    """Configuration for the detachable RingBufferLatentTee & SidecarProbe."""

    enable_latent_tee: bool = Field(
        default=False,
        description=(
            "Disabled by default in production for zero compute/memory overhead; "
            "set True only during on-demand mechanistic auditing."
        ),
    )
    ring_buffer_capacity_m: int = Field(
        default=8,
        ge=1,
        description="Fixed circular buffer size M <= K bounding activation memory to O(M*d) = O(1) w.r.t. K.",
    )
    enforce_stop_gradient_detach: bool = Field(
        default=True,
        description="Enforce z_k.detach() (sg[z_k]) at RingBufferLatentTee.push before SidecarProbe or LogitLens.",
    )
    separate_probe_optimizer: bool = Field(
        default=True,
        description="Optimize L_probe exclusively w.r.t. phi so grad_{theta_backbone} L_probe = 0.",
    )

    @model_validator(mode="after")
    def _validate_invariants(self) -> LatentTeeConfig:
        if not self.enforce_stop_gradient_detach:
            raise ValueError(
                "Architectural invariant violation: enforce_stop_gradient_detach must be True "
                "to prevent auxiliary readout heads from backpropagating into continuous thought states."
            )
        if not self.separate_probe_optimizer:
            raise ValueError(
                "Architectural invariant violation: separate_probe_optimizer must be True "
                "so grad_{theta_backbone} L_probe == 0."
            )
        return self


class RegimeExperimentConfig(BaseModel):
    """Per-regime evaluation configuration and literature positioning metadata."""

    regime: RegimeKey
    alias: str | None = None
    role: RegimeRole
    bypasses_w_unembed_during_reasoning: bool
    latent_steps_k: int = Field(default=0, ge=0)
    tee: LatentTeeConfig | None = None
    positioning_note: str

    @model_validator(mode="after")
    def _check_ring_capacity_bound(self) -> RegimeExperimentConfig:
        if self.regime == "latent_loop" and self.tee is not None and self.latent_steps_k > 0:
            if self.tee.ring_buffer_capacity_m > self.latent_steps_k:
                raise ValueError(
                    f"RingBufferLatentTee capacity M ({self.tee.ring_buffer_capacity_m}) "
                    f"must satisfy M <= K ({self.latent_steps_k})."
                )
        return self


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "ablation_regimes.json"


def load_regime_configs(config_path: Path | None = None) -> dict[RegimeKey, RegimeExperimentConfig]:
    """Load and validate all regime experiment configurations from JSON."""
    path = config_path or DEFAULT_CONFIG_PATH
    raw = json.loads(path.read_text(encoding="utf-8"))
    parsed: dict[RegimeKey, RegimeExperimentConfig] = {}
    for entry in raw["regimes"]:
        cfg = RegimeExperimentConfig.model_validate(entry)
        parsed[cfg.regime] = cfg
    return parsed
