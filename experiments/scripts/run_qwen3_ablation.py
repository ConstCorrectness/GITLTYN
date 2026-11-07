"""End-to-end CLI runner for the PyTorch Qwen3-0.6B Pretraining + Post-Training Ablation.

Executes:
1. Optional scratch-init verification pass (`--mode hybrid` or `--mode scratch`).
2. Stage 1 Pretraining (from-scratch or domain-adaptive continued pretraining on `Qwen/Qwen3-0.6B-Base`).
3. Stage 2 Post-Training:
   - 2a. Regime-conditioned SFT (`verbose_cot` & `telegraphic_cot`)
   - 2b. `TorchContinuousLatentAdapter` (f_theta) recurrence training in R^d
   - 2c. Disjoint `TorchSidecarProbe` (g_phi(sg[z_{1:K}])) optimization (verifying grad_backbone == 0)
4. Stage 3 4-Regime Evaluation across GSM8K & MATH + `RingBufferLatentTee` & `PruningTrace` extraction,
   writing `experiments/artifacts/qwen3_0.6b_results.json` and updating `web/src/data/ablation_frontier.json`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENTS_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT / "src"))
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

from datasets import (  # noqa: E402
    MicroMathTokenizer,
    PROBE_STAGE_TEMPLATES,
    build_gsm8k_examples,
    build_math_examples,
)
from qwen3_ablation import (  # noqa: E402
    Qwen3CausalLM,
    Qwen3ModelConfig,
    TorchContinuousLatentAdapter,
    TorchSidecarProbe,
)
from scripts.export_web_artifacts import export_to_web  # noqa: E402
from train_pipeline import (  # noqa: E402
    evaluate_qwen3_regimes,
    run_posttraining,
    run_pretraining,
)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def run_scratch_micro_verification(device: torch.device) -> dict[str, float]:
    """Run deterministic from-scratch Qwen3 pretraining + post-training sanity check."""
    torch.manual_seed(42)
    gsm_train = build_gsm8k_examples(18, seed=11)
    texts = [f"{ex.prompt} {ex.verbose_cot} {ex.telegraphic_cot}" for ex in gsm_train]
    tok = MicroMathTokenizer(texts=texts, max_vocab=256)
    cfg = Qwen3ModelConfig.micro_ablation(vocab_size=tok.vocab_size)

    scratch_model = Qwen3CausalLM(cfg).to(device=device, dtype=torch.float32)
    pre_init, pre_final, _ = run_pretraining(
        model=scratch_model,
        tokenizer=tok,
        train_examples=gsm_train,
        device=device,
        mode="scratch",
        steps=24,
        batch_size=6,
        lr=3e-3,
    )
    adapter = TorchContinuousLatentAdapter(hidden_dim=cfg.hidden_dim).to(device=device)
    probe = TorchSidecarProbe(
        hidden_dim=cfg.hidden_dim, templates=PROBE_STAGE_TEMPLATES
    ).to(device=device)
    post_metrics = run_posttraining(
        model=scratch_model,
        adapter=adapter,
        probe=probe,
        tokenizer=tok,
        train_examples=gsm_train,
        device=device,
        mode="scratch",
        sft_steps=16,
        latent_steps_train=16,
        probe_steps=20,
        recurrence_k=4,
        ring_capacity_m=4,
        sft_lr=2e-3,
        adapter_lr=4e-3,
        probe_lr=2e-2,
    )
    return {
        "scratch_pretrain_initial_loss": round(pre_init, 4),
        "scratch_pretrain_final_loss": round(pre_final, 4),
        "scratch_sft_final_loss": post_metrics.sft_final_loss,
        "scratch_latent_final_loss": post_metrics.latent_final_loss,
        "scratch_probe_final_loss": post_metrics.probe_final_loss,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run PyTorch Qwen3-0.6B pretraining, post-training, and 4-regime evaluation."
    )
    parser.add_argument(
        "--mode",
        choices=("hybrid", "continued", "scratch"),
        default="hybrid",
        help="Training mode: 'hybrid' (scratch micro verification + HF Qwen3-0.6B continued/post-train), 'continued', or 'scratch'.",
    )
    parser.add_argument(
        "--hf-model",
        default="Qwen/Qwen3-0.6B-Base",
        help="HuggingFace model ID for 0.6B continued pretraining & post-training.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Compute device ('auto', 'cuda', or 'cpu').",
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=48,
        help="Number of training problems per benchmark split.",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=24,
        help="Number of held-out evaluation problems per benchmark split (use --full-eval for large sweep).",
    )
    parser.add_argument(
        "--full-eval",
        action="store_true",
        help="Expand evaluation suite to 200 GSM8K + 150 MATH problems (recommended with --device cuda).",
    )
    parser.add_argument(
        "--pretrain-steps",
        type=int,
        default=24,
        help="Stage 1 causal LM pretraining steps.",
    )
    parser.add_argument(
        "--sft-steps",
        type=int,
        default=72,
        help="Stage 2a regime-conditioned SFT steps.",
    )
    parser.add_argument(
        "--latent-steps",
        type=int,
        default=96,
        help="Stage 2b Continuous Latent Loop adapter f_theta optimization steps.",
    )
    parser.add_argument(
        "--probe-steps",
        type=int,
        default=60,
        help="Stage 2c disjoint SidecarProbe optimization steps.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EXPERIMENTS_ROOT / "artifacts" / "qwen3_0.6b_results.json",
        help="Output JSON path for Qwen3-0.6B telemetry & evaluation results.",
    )
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if not torch.cuda.is_available():
        cpu_threads = min(16, os.cpu_count() or 8)
        torch.set_num_threads(cpu_threads)

    device = resolve_device(args.device)
    torch.manual_seed(2026)
    t0 = time.perf_counter()

    n_eval = 200 if args.full_eval else args.eval_samples
    n_math_eval = 150 if args.full_eval else args.eval_samples

    print(
        f"[GITLTYN] Device={device} | Mode={args.mode} | Eval={n_eval} GSM8K + {n_math_eval} MATH",
        flush=True,
    )

    scratch_telemetry: dict[str, float] | None = None
    if args.mode in ("hybrid", "scratch"):
        print(
            "[Stage 0] Running from-scratch Qwen3 PyTorch architecture pretrain/posttrain check...",
            flush=True,
        )
        scratch_telemetry = run_scratch_micro_verification(device=device)
        print(f"          Scratch telemetry: {scratch_telemetry}", flush=True)

    gsm_train = build_gsm8k_examples(args.train_samples, seed=42)
    math_train = build_math_examples(args.train_samples, seed=108)
    train_pool = [
        item
        for pair in zip(gsm_train, math_train, strict=True)
        for item in pair
    ]

    gsm_eval = build_gsm8k_examples(n_eval, seed=7001)
    math_eval = build_math_examples(n_math_eval, seed=8001)

    if args.mode == "scratch":
        all_texts = [
            f"{ex.prompt} {ex.verbose_cot} {ex.telegraphic_cot}"
            for ex in (*train_pool, *gsm_eval, *math_eval)
        ]
        tokenizer = MicroMathTokenizer(texts=all_texts, max_vocab=512)
        cfg = Qwen3ModelConfig(
            model_name="Qwen3-0.6B",
            param_count_b=0.6,
            vocab_size=tokenizer.vocab_size,
            hidden_dim=256,
            intermediate_dim=768,
            num_layers=6,
            num_attention_heads=8,
            num_kv_heads=4,
            head_dim=32,
        )
        model = Qwen3CausalLM(cfg).to(device=device, dtype=torch.float32)
        train_mode = "scratch"
        unfreeze_layers = cfg.num_layers
    else:
        from transformers import AutoTokenizer

        print(
            f"[Stage 1] Loading HuggingFace tokenizer & pretrained weights from {args.hf_model}...",
            flush=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.hf_model)
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        model = Qwen3CausalLM.from_huggingface(
            repo_id=args.hf_model,
            device=device,
            dtype=dtype,
        )
        train_mode = "continued"
        unfreeze_layers = 4

    print(
        f"[Stage 1] Running {train_mode} causal LM pretraining ({args.pretrain_steps} steps)...",
        flush=True,
    )
    pre_init, pre_final, _ = run_pretraining(
        model=model,
        tokenizer=tokenizer,
        train_examples=train_pool,
        device=device,
        mode=train_mode,  # type: ignore[arg-type]
        steps=args.pretrain_steps,
        batch_size=8,
        lr=8e-5 if train_mode == "continued" else 2e-3,
        unfreeze_top_layers=unfreeze_layers,
    )
    print(f"          Pretrain loss: {pre_init:.4f} -> {pre_final:.4f}", flush=True)

    adapter = TorchContinuousLatentAdapter(
        hidden_dim=model.config.hidden_dim,
        rms_norm_eps=model.config.rms_norm_eps,
    ).to(device=device, dtype=torch.float32)
    probe = TorchSidecarProbe(
        hidden_dim=model.config.hidden_dim,
        templates=PROBE_STAGE_TEMPLATES,
        rms_norm_eps=model.config.rms_norm_eps,
    ).to(device=device, dtype=torch.float32)

    print(
        f"[Stage 2] Running multi-regime post-training "
        f"(SFT={args.sft_steps}, LatentLoop={args.latent_steps}, SidecarProbe={args.probe_steps})...",
        flush=True,
    )
    stage_metrics = run_posttraining(
        model=model,
        adapter=adapter,
        probe=probe,
        tokenizer=tokenizer,
        train_examples=train_pool,
        device=device,
        mode=train_mode,  # type: ignore[arg-type]
        sft_steps=args.sft_steps,
        latent_steps_train=args.latent_steps,
        probe_steps=args.probe_steps,
        recurrence_k=6,
        ring_capacity_m=6,
        batch_size=8,
        sft_lr=9e-5 if train_mode == "continued" else 1.5e-3,
        adapter_lr=2.5e-3,
        probe_lr=2e-2,
        unfreeze_top_layers=unfreeze_layers,
    )
    stage_metrics.pretrain_initial_loss = round(pre_init, 4)
    stage_metrics.pretrain_final_loss = round(pre_final, 4)
    print(
        f"          SFT: {stage_metrics.sft_initial_loss:.4f} -> {stage_metrics.sft_final_loss:.4f} | "
        f"Latent f_theta: {stage_metrics.latent_initial_loss:.4f} -> {stage_metrics.latent_final_loss:.4f} | "
        f"Probe g_phi: {stage_metrics.probe_initial_loss:.4f} -> {stage_metrics.probe_final_loss:.4f} "
        f"(zero_backbone_grad={stage_metrics.verified_zero_backbone_grad_on_probe})",
        flush=True,
    )

    print(
        "[Stage 3] Evaluating all 4 regimes across GSM8K & MATH and extracting Tee/Pruning traces...",
        flush=True,
    )
    eval_payload = evaluate_qwen3_regimes(
        model=model,
        adapter=adapter,
        probe=probe,
        tokenizer=tokenizer,
        gsm8k_eval=gsm_eval,
        math_eval=math_eval,
        device=device,
    )

    elapsed_s = round(time.perf_counter() - t0, 2)
    output_doc = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hf_source_model": args.hf_model if args.mode != "scratch" else "random_init",
        "mode": args.mode,
        "device": str(device),
        "elapsed_seconds": elapsed_s,
        "scratch_verification": scratch_telemetry,
        "training_metrics": {
            "pretrain_initial_loss": stage_metrics.pretrain_initial_loss,
            "pretrain_final_loss": stage_metrics.pretrain_final_loss,
            "sft_initial_loss": stage_metrics.sft_initial_loss,
            "sft_final_loss": stage_metrics.sft_final_loss,
            "latent_initial_loss": stage_metrics.latent_initial_loss,
            "latent_final_loss": stage_metrics.latent_final_loss,
            "probe_initial_loss": stage_metrics.probe_initial_loss,
            "probe_final_loss": stage_metrics.probe_final_loss,
            "verified_zero_backbone_grad_on_probe": stage_metrics.verified_zero_backbone_grad_on_probe,
        },
        **eval_payload,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output_doc, indent=2) + "\n", encoding="utf-8")
    print(f"[Stage 3] Saved Qwen3-0.6B ablation results ({elapsed_s}s) -> {args.output}")

    web_artifact = export_to_web()
    print(f"[Export]  Updated schema-validated web artifact -> {web_artifact}")


if __name__ == "__main__":
    main()
