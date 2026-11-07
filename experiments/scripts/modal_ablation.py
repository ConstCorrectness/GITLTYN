"""Modal serverless GPU execution pipeline for GITLTYN Qwen3-0.6B ablation suite.

Allows scaling the ablation sweeps across serverless NVIDIA GPUs (T4, L4, A10G, A100)
while streaming traces into Opik Cloud and updating local web artifacts.

Usage:
  cd experiments && uv run modal run scripts/modal_ablation.py
  cd experiments && uv run modal run scripts/modal_ablation.py --gpu A10G --mode continued --full-eval
"""

from __future__ import annotations

import os
from pathlib import Path

# Load local .env if present (supporting OPIK_API_KEY, MODAL_TOKEN_ID, MODAL_TOKEN_SECRET)
try:
    from dotenv import load_dotenv

    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

import modal

APP_NAME = "gitltyn-ablation"
EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = EXPERIMENTS_DIR.parent

app = modal.App(APP_NAME)

# Build CUDA-accelerated container image with all GITLTYN ablation dependencies
ablation_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.6.0",
        "transformers>=4.51.0",
        "tokenizers>=0.21.0",
        "safetensors>=0.5.0",
        "pydantic>=2.9.0",
        "jsonschema>=4.23.0",
        "numpy>=2.0.0",
        "opik>=2.2.0",
        "python-dotenv>=1.0.0",
    )
    .add_local_dir(str(EXPERIMENTS_DIR / "src"), remote_path="/root/gitltyn/src")
    .add_local_dir(str(ROOT_DIR / "shared"), remote_path="/root/gitltyn/shared")
    .add_local_file(
        str(EXPERIMENTS_DIR / "scripts" / "export_web_artifacts.py"),
        remote_path="/root/gitltyn/scripts/export_web_artifacts.py",
    )
)

# Optional Opik secrets passed from environment to remote container
env_secrets = {}
if os.getenv("OPIK_API_KEY"):
    env_secrets["OPIK_API_KEY"] = os.getenv("OPIK_API_KEY", "")
if os.getenv("OPIK_WORKSPACE"):
    env_secrets["OPIK_WORKSPACE"] = os.getenv("OPIK_WORKSPACE", "constcorrectness")
if os.getenv("OPIK_PROJECT_NAME"):
    env_secrets["OPIK_PROJECT_NAME"] = os.getenv("OPIK_PROJECT_NAME", "GITLTYN")


@app.function(
    image=ablation_image,
    gpu="L4",
    timeout=3600,
    secrets=[modal.Secret.from_dict(env_secrets)] if env_secrets else [],
)
def run_gpu_ablation(
    mode: str = "hybrid",
    hf_model: str = "Qwen/Qwen3-0.6B-Base",
    train_samples: int = 48,
    eval_samples: int = 24,
    full_eval: bool = False,
    pretrain_steps: int = 24,
    sft_steps: int = 72,
    latent_steps: int = 96,
    probe_steps: int = 60,
) -> dict:
    """Execute Qwen3-0.6B pretraining, SFT, latent adapter recurrence, and 4-regime evaluation on GPU."""
    import sys
    import time
    from datetime import datetime, timezone
    import torch
    import opik

    sys.path.insert(0, "/root/gitltyn/src")
    sys.path.insert(0, "/root/gitltyn")

    from datasets import (
        MicroMathTokenizer,
        build_gsm8k_examples,
        build_math_examples,
    )
    from qwen3_ablation import (
        Qwen3CausalLM,
        Qwen3ModelConfig,
        TorchContinuousLatentAdapter,
        TorchSidecarProbe,
    )
    from train_pipeline import (
        evaluate_qwen3_regimes,
        run_posttraining,
        run_pretraining,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[Modal Remote] Device={device} ({gpu_name}) | Mode={mode}", flush=True)

    t0 = time.perf_counter()
    n_eval = 200 if full_eval else eval_samples
    n_math_eval = 150 if full_eval else eval_samples

    # Stage 0: Scratch verification if hybrid/scratch
    scratch_telemetry = None
    if mode in ("hybrid", "scratch"):
        print("[Modal Remote] Running scratch verification...", flush=True)
        # Scratch pretraining check
        torch.manual_seed(42)
        cfg = Qwen3ModelConfig(
            model_name="Qwen3-0.6B-ScratchMicro",
            hidden_size=256,
            intermediate_size=768,
            num_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=256,
            max_position_embeddings=512,
        )
        micro_model = Qwen3CausalLM(cfg).to(device)
        micro_adapter = TorchContinuousLatentAdapter(hidden_size=256).to(device)
        micro_probe = TorchSidecarProbe(hidden_size=256, num_stages=6).to(device)
        micro_tokenizer = MicroMathTokenizer(
            ["1 + 1 = 2", "x = 4", "Natalia sold clips"], max_vocab=256
        )
        scratch_metrics = run_posttraining(
            model=micro_model,
            adapter=micro_adapter,
            probe=micro_probe,
            tokenizer=micro_tokenizer,
            train_examples=build_gsm8k_examples(12, seed=42),
            device=device,
            sft_steps=12,
            latent_steps=16,
            probe_steps=12,
            batch_size=4,
        )
        scratch_telemetry = {
            "scratch_sft_final_loss": scratch_metrics.sft_final_loss,
            "scratch_latent_final_loss": scratch_metrics.latent_final_loss,
            "scratch_probe_final_loss": scratch_metrics.probe_final_loss,
        }
        del micro_model, micro_adapter, micro_probe

    # Prepare datasets
    gsm_train = build_gsm8k_examples(train_samples, seed=42)
    math_train = build_math_examples(train_samples, seed=108)
    train_pool = [
        item for pair in zip(gsm_train, math_train, strict=True) for item in pair
    ]
    gsm_eval = build_gsm8k_examples(n_eval, seed=7001)
    math_eval = build_math_examples(n_math_eval, seed=8001)

    if mode == "scratch":
        all_texts = [
            f"{ex.prompt} {ex.verbose_cot} {ex.telegraphic_cot}"
            for ex in (*train_pool, *gsm_eval, *math_eval)
        ]
        tokenizer = MicroMathTokenizer(texts=all_texts, max_vocab=512)
        cfg = Qwen3ModelConfig(
            model_name="Qwen3-0.6B",
            hidden_size=1024,
            intermediate_size=3072,
            num_layers=16,
            num_attention_heads=16,
            num_key_value_heads=8,
            vocab_size=max(512, len(tokenizer.vocab)),
            max_position_embeddings=2048,
        )
        model = Qwen3CausalLM(cfg).to(device=device, dtype=torch.float32)
        unfreeze_layers = cfg.num_layers
    else:
        from transformers import AutoTokenizer

        print(f"[Modal Remote] Loading {hf_model} weights...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(hf_model)
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        model = Qwen3CausalLM.from_huggingface(
            repo_id=hf_model,
            device=device,
            dtype=dtype,
        )
        unfreeze_layers = 4

    adapter = TorchContinuousLatentAdapter(hidden_size=model.config.hidden_size).to(
        device=device, dtype=model.lm_head.weight.dtype
    )
    adapter.init_from_symbolic_subspace(
        unembed_weight=model.lm_head.weight.data,
        operator_token_ids=[
            getattr(tokenizer, "encode")(op, add_special_tokens=False)[0]
            for op in ("+", "-", "*", "/", "=", "^")
            if getattr(tokenizer, "encode")(op, add_special_tokens=False)
        ],
    )
    probe = TorchSidecarProbe(
        hidden_size=model.config.hidden_size, num_stages=6
    ).to(device=device, dtype=torch.float32)

    # Stage 1: Pretraining
    print("[Modal Remote] Stage 1: Domain-adaptive pretraining...", flush=True)
    pretrain_initial, pretrain_final = run_pretraining(
        model=model,
        tokenizer=tokenizer,
        train_examples=train_pool,
        device=device,
        steps=pretrain_steps,
        batch_size=8,
        lr=3e-5,
        unfreeze_last_n_layers=unfreeze_layers,
    )

    # Stage 2: Post-Training
    print("[Modal Remote] Stage 2: Multi-regime SFT + Latent recurrence...", flush=True)
    stage_metrics = run_posttraining(
        model=model,
        adapter=adapter,
        probe=probe,
        tokenizer=tokenizer,
        train_examples=train_pool,
        device=device,
        sft_steps=sft_steps,
        latent_steps=latent_steps,
        probe_steps=probe_steps,
        batch_size=8,
        sft_lr=2e-5,
        adapter_lr=1e-4,
        probe_lr=1e-3,
        unfreeze_last_n_layers=unfreeze_layers,
    )

    # Stage 3: Evaluation
    print("[Modal Remote] Stage 3: 4-Regime Evaluation...", flush=True)
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
        "hf_source_model": hf_model if mode != "scratch" else "random_init",
        "mode": mode,
        "device": str(device),
        "gpu_name": gpu_name,
        "elapsed_seconds": elapsed_s,
        "scratch_verification": scratch_telemetry,
        "training_metrics": {
            "pretrain_initial_loss": round(pretrain_initial, 4),
            "pretrain_final_loss": round(pretrain_final, 4),
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

    opik.flush_tracker()
    return output_doc


@app.local_entrypoint()
def main(
    mode: str = "hybrid",
    hf_model: str = "Qwen/Qwen3-0.6B-Base",
    train_samples: int = 48,
    eval_samples: int = 24,
    full_eval: bool = False,
    gpu: str = "L4",
):
    """Local CLI entrypoint invoking remote Modal GPU function."""
    import json
    from scripts.export_web_artifacts import export_to_web

    print(f"[Modal Local] Launching remote GITLTYN ablation on GPU ({gpu})...")
    result = run_gpu_ablation.remote(
        mode=mode,
        hf_model=hf_model,
        train_samples=train_samples,
        eval_samples=eval_samples,
        full_eval=full_eval,
    )

    out_file = EXPERIMENTS_DIR / "artifacts" / "qwen3_0.6b_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"[Modal Local] Saved GPU results to {out_file}")

    web_path = export_to_web()
    print(f"[Modal Local] Exported to web artifact: {web_path}")
