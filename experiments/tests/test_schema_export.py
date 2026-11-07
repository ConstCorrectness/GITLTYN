"""Tests for continuous latent loop invariants, stop-gradient Tee, pruner, and schema export."""

from __future__ import annotations

import json
import math
from pathlib import Path

import jsonschema
import numpy as np
import pytest

from config import LatentTeeConfig, RegimeExperimentConfig, load_regime_configs
from latent_loop import (
    ContinuousLatentLoop,
    RingBufferLatentTee,
    SidecarProbe,
    TrackedLatentTensor,
)
from prune_syntax import prune_cot_trace
from scripts.export_web_artifacts import (
    compute_accuracy_logit,
    compute_wilson_ci_95,
    export_to_web,
    generate_artifact,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "shared" / "schema.json"


def test_ring_buffer_latent_tee_production_default_and_bounded_memory() -> None:
    """Verify M <= K O(1) circular buffer capacity and enable_latent_tee=False default."""
    d_dim = 32
    vocab = [f"tok_{i}" for i in range(20)]
    rng = np.random.default_rng(7)
    w_unembed = rng.normal(0.0, 0.2, size=(d_dim, len(vocab)))
    h_init = rng.normal(0.0, 1.0, size=(d_dim,))

    loop = ContinuousLatentLoop.from_seed(hidden_dim=d_dim, seed=7)

    # 1. Production default: enable_latent_tee=False -> strict O(1) no-op
    default_tee = RingBufferLatentTee(capacity=4)
    assert default_tee.enable_latent_tee is False
    z_prod = loop.forward(h_init, steps_k=10, tee=default_tee)
    assert len(default_tee) == 0

    # 2. Mechanistic audit mode: enable_latent_tee=True with fixed M = 4 <= K = 10
    audit_tee = RingBufferLatentTee(capacity=4, enable_latent_tee=True)
    z_audit = loop.forward(h_init, steps_k=10, tee=audit_tee)

    # Bit-for-bit forward non-interference
    assert np.array_equal(z_prod, z_audit)
    # Strictly bounded circular memory: retains only the latest M = 4 transitions (7..10)
    assert len(audit_tee) == 4

    probe = SidecarProbe(
        prototype_vectors=rng.normal(0.0, 1.0, size=(3, d_dim)),
        prototype_Templates=["state_a@{step_k}", "state_b@{step_k}", "state_c@{step_k}"],
    )
    records = audit_tee.flush(w_unembed=w_unembed, vocab=vocab, sidecar_probe=probe, top_k=5)
    assert [r.step_k for r in records] == [7, 8, 9, 10]
    for rec in records:
        assert rec.l2_norm > 0.0
        assert len(rec.logit_lens_top_k) == 5
        probs = [entry.prob for entry in rec.logit_lens_top_k]
        assert probs == sorted(probs, reverse=True)
        assert "@" in rec.sidecar_decoded


def test_push_calls_detach_and_sidecar_probe_has_zero_backbone_gradient() -> None:
    """Verify RingBufferLatentTee.push(z_k) calls z_k.detach() and grad_{theta_backbone} L_probe == 0."""
    d_dim = 24
    steps_k = 6
    ring_m = 4
    rng = np.random.default_rng(19)
    h_init = rng.normal(0.0, 1.0, size=(d_dim,))

    loop = ContinuousLatentLoop.from_seed(hidden_dim=d_dim, seed=19)
    loop.zero_backbone_grads()

    tee = RingBufferLatentTee(capacity=ring_m, enable_latent_tee=True)
    z_final, tracked_steps = loop.forward_tracked(h_init, steps_k=steps_k, tee=tee)

    # Action Item 1: Every z_k passed to RingBufferLatentTee.push(z_k) had .detach() called
    assert len(tracked_steps) == steps_k
    for z_step in tracked_steps:
        assert isinstance(z_step, TrackedLatentTensor)
        assert z_step.detach_call_count == 1
        assert z_step.grad is not None and np.all(z_step.grad == 0.0)

    # Buffered states inside RingBufferLatentTee are detached, read-only float64 snapshots
    buffered_states = tee.peek_detached_states()
    assert len(buffered_states) == ring_m
    assert all(not arr.flags.writeable for arr in buffered_states)

    # Action Item 2: Optimize SidecarProbe loss L_probe via separate optimizer step
    probe = SidecarProbe(
        prototype_vectors=rng.normal(0.0, 1.0, size=(4, d_dim)),
        prototype_Templates=["s0@{step_k}", "s1@{step_k}", "s2@{step_k}", "s3@{step_k}"],
    )
    targets = [0, 1, 2, 3]

    # Even if live tracked_steps (with requires_grad=True) are passed directly to SidecarProbe,
    # SidecarProbe detaches them and updates strictly phi, leaving grad_{theta_backbone} == 0.
    initial_loss = probe.compute_probe_loss(tracked_steps[-ring_m:], targets)
    for _ in range(12):
        _ = probe.step_probe_optimizer(tracked_steps[-ring_m:], targets, lr=0.25)
    final_loss = probe.compute_probe_loss(tracked_steps[-ring_m:], targets)

    assert final_loss < initial_loss
    assert probe.optimizer_steps == 12
    assert loop.backbone_grad_norm == 0.0
    assert np.all(loop.grad_w_gate == 0.0)
    assert np.all(loop.grad_w_up == 0.0)
    assert np.all(loop.grad_w_down == 0.0)

    # Sanity check: primary task loss on z_final DOES populate theta_backbone gradients
    z_final.backward(np.ones(d_dim, dtype=np.float64))
    assert loop.backbone_grad_norm > 0.0


def test_regime_experiment_configs_enforce_invariants() -> None:
    """Verify experiment configs enforce production defaults and baseline roles."""
    configs = load_regime_configs()
    assert set(configs.keys()) == {
        "verbose_cot",
        "telegraphic_cot",
        "compute_matched",
        "latent_loop",
    }

    assert configs["telegraphic_cot"].role == "motivational_ablation_control"
    assert configs["compute_matched"].alias == "budget_forced_cot"
    assert configs["latent_loop"].role == "proposed_architecture"

    latent_tee_cfg = configs["latent_loop"].tee
    assert latent_tee_cfg is not None
    assert latent_tee_cfg.enable_latent_tee is False
    assert latent_tee_cfg.ring_buffer_capacity_m <= configs["latent_loop"].latent_steps_k
    assert latent_tee_cfg.enforce_stop_gradient_detach is True
    assert latent_tee_cfg.separate_probe_optimizer is True

    # Disabling stop-gradient or separate probe optimizer must raise ValidationError
    with pytest.raises(ValueError):
        LatentTeeConfig(enforce_stop_gradient_detach=False)
    with pytest.raises(ValueError):
        LatentTeeConfig(separate_probe_optimizer=False)
    with pytest.raises(ValueError):
        RegimeExperimentConfig(
            regime="latent_loop",
            role="proposed_architecture",
            bypasses_w_unembed_during_reasoning=True,
            latent_steps_k=4,
            tee=LatentTeeConfig(ring_buffer_capacity_m=8),
            positioning_note="invalid M > K",
        )


def test_syntactic_pruner_preserves_state_transitions() -> None:
    """100% syntactic pruning removes grammatical glue while preserving equations & numbers."""
    trace = (
        "Let's think step by step to find the total. "
        "First, we know that apr = 48 and may = 48 / 2 = 24. "
        "Therefore, the final sum is total = 48 + 24 = 72."
    )
    zero_pruned = prune_cot_trace(trace, ratio=0.0)
    half_pruned = prune_cot_trace(trace, ratio=0.5)
    full_pruned = prune_cot_trace(trace, ratio=1.0)

    assert zero_pruned.retained_token_count == zero_pruned.original_token_count
    assert full_pruned.retained_token_count < half_pruned.retained_token_count < zero_pruned.retained_token_count
    assert full_pruned.token_reduction_ratio > 0.45

    for critical_symbol in ("48", "24", "72", "+"):
        assert critical_symbol in full_pruned.retained_text

    for glue_word in ("Let's", "Therefore,", "the"):
        assert glue_word not in full_pruned.retained_text.split()

    with pytest.raises(ValueError):
        prune_cot_trace(trace, ratio=1.2)


def test_logit_and_confidence_interval_math() -> None:
    """Verify logit transform and Wilson 95% confidence interval invariants."""
    assert compute_accuracy_logit(0.5) == pytest.approx(0.0, abs=1e-6)
    assert compute_accuracy_logit(0.8) == pytest.approx(math.log(4.0), abs=1e-4)

    low, high = compute_wilson_ci_95(0.75, 1000)
    assert 0.70 < low < 0.75 < high < 0.80


def test_exported_artifact_matches_shared_json_schema(tmp_path: Path) -> None:
    """Generated artifact must strictly validate against shared/schema.json across all 4 regimes."""
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    payload = generate_artifact()

    jsonschema.validate(
        instance=payload,
        schema=schema,
        format_checker=jsonschema.Draft202012Validator.FORMAT_CHECKER,
    )

    runs = payload["runs"]
    assert isinstance(runs, list)
    regimes = {r["regime"] for r in runs}
    assert regimes == {"verbose_cot", "telegraphic_cot", "compute_matched", "latent_loop"}

    benchmarks = {r["benchmark"] for r in runs}
    assert {"GSM8K", "MATH"}.issubset(benchmarks)

    param_scales = {r["param_count_b"] for r in runs}
    assert {0.6, 1.5, 3.0}.issubset(param_scales)

    for bench in ("GSM8K", "MATH"):
        bench_runs = [r for r in runs if r["benchmark"] == bench]
        pareto_runs = [r for r in bench_runs if r["is_pareto_frontier"]]
        assert len(pareto_runs) >= 2
        for r in bench_runs:
            lo, hi = r["ci_95"]
            assert 0.0 <= lo <= r["accuracy"] <= hi <= 1.0

    for trace in payload["tee_traces"]:
        assert len(trace["latent_steps"]) >= 5
        for step in trace["latent_steps"]:
            top_probs = [item["prob"] for item in step["logit_lens_top_k"]]
            assert 0.20 <= top_probs[0] <= 0.90
            assert top_probs[2] > 0.01

    out_path = tmp_path / "ablation_frontier.json"
    written = export_to_web(output_path=out_path)
    assert written.exists()
    reloaded = json.loads(written.read_text(encoding="utf-8"))
    assert reloaded == payload
