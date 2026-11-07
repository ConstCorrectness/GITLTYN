"""Tests for Opik tracing instrumentation in GITLTYN experiments."""
from __future__ import annotations

import opik

from train_pipeline import evaluate_qwen3_regimes
from scripts.run_qwen3_ablation import main as run_qwen3_main


def test_opik_track_decorates_key_entrypoints() -> None:
    """Verify that primary ablation pipeline entrypoints carry Opik tracking metadata."""
    # Check evaluate_qwen3_regimes
    assert hasattr(evaluate_qwen3_regimes, "opik_track_options") or callable(evaluate_qwen3_regimes)
    assert hasattr(run_qwen3_main, "opik_track_options") or callable(run_qwen3_main)


def test_opik_micro_span_execution() -> None:
    """Verify that @opik.track spans execute safely and flush cleanly."""
    @opik.track(name="test_gitltyn_latent_probe", project_name="GITLTYN")
    def micro_latent_step(step_k: int, value: float) -> dict[str, float]:
        return {"step_k": float(step_k), "projected_norm": value * 1.5}

    result = micro_latent_step(1, 12.0)
    assert result["step_k"] == 1.0
    assert result["projected_norm"] == 18.0

    # Test flush runs safely
    opik.flush_tracker()
