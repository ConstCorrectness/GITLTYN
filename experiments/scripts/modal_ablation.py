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
import modal

CURR_FILE = Path(__file__).resolve()

# Guard directory resolution: handle local repo workspace vs remote Modal container (/root/modal_ablation.py)
if len(CURR_FILE.parents) >= 3:
    EXPERIMENTS_DIR = CURR_FILE.parents[1]
    ROOT_DIR = CURR_FILE.parents[2]
elif len(CURR_FILE.parents) >= 2:
    EXPERIMENTS_DIR = CURR_FILE.parents[0]
    ROOT_DIR = CURR_FILE.parents[1]
else:
    EXPERIMENTS_DIR = Path("/root/gitltyn")
    ROOT_DIR = Path("/root/gitltyn")

# Load local .env only on host machine when running locally
if modal.is_local():
    try:
        from dotenv import load_dotenv

        env_path = ROOT_DIR / ".env"
        if env_path.exists():
            load_dotenv(env_path)
    except Exception:
        pass

APP_NAME = "gitltyn-ablation"
app = modal.App(APP_NAME)

# Build CUDA-accelerated container image with all GITLTYN ablation dependencies
if modal.is_local():
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
        .add_local_dir(str(EXPERIMENTS_DIR / "configs"), remote_path="/root/gitltyn/configs")
        .add_local_dir(str(EXPERIMENTS_DIR / "scripts"), remote_path="/root/gitltyn/scripts")
        .add_local_dir(str(ROOT_DIR / "shared"), remote_path="/root/gitltyn/shared")
    )
else:
    ablation_image = modal.Image.debian_slim(python_version="3.11")

# Optional Opik secrets passed from environment to remote container
env_secrets = {}
if os.getenv("OPIK_API_KEY"):
    env_secrets["OPIK_API_KEY"] = os.getenv("OPIK_API_KEY", "")
if os.getenv("OPIK_WORKSPACE"):
    env_secrets["OPIK_WORKSPACE"] = os.getenv("OPIK_WORKSPACE", "constcorrectness")
if os.getenv("OPIK_PROJECT_NAME"):
    env_secrets["OPIK_PROJECT_NAME"] = os.getenv("OPIK_PROJECT_NAME", "GITLTYN")

# GPU configuration: prioritize A10G with automatic A100 fallback to eliminate queue wait times
gpu_env = os.getenv("MODAL_GPU", "").strip()
if gpu_env:
    SELECTED_GPU = [g.strip() for g in gpu_env.split(",")] if "," in gpu_env else gpu_env
else:
    SELECTED_GPU = ["A10G", "A100"]


@app.function(
    image=ablation_image,
    gpu=SELECTED_GPU,
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
        PROBE_STAGE_TEMPLATES,
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
        torch.manual_seed(42)
        gsm_train_micro = build_gsm8k_examples(18, seed=11)
        micro_texts = [
            f"{ex.prompt} {ex.verbose_cot} {ex.telegraphic_cot}"
            for ex in gsm_train_micro
        ]
        micro_tok = MicroMathTokenizer(texts=micro_texts, max_vocab=256)
        micro_cfg = Qwen3ModelConfig.micro_ablation(vocab_size=micro_tok.vocab_size)

        micro_model = Qwen3CausalLM(micro_cfg).to(device=device, dtype=torch.float32)
        pre_init_scratch, pre_final_scratch, _ = run_pretraining(
            model=micro_model,
            tokenizer=micro_tok,
            train_examples=gsm_train_micro,
            device=device,
            mode="scratch",
            steps=24,
            batch_size=6,
            lr=3e-3,
        )
        micro_adapter = TorchContinuousLatentAdapter(hidden_dim=micro_cfg.hidden_dim).to(device=device)
        micro_probe = TorchSidecarProbe(
            hidden_dim=micro_cfg.hidden_dim, templates=PROBE_STAGE_TEMPLATES
        ).to(device=device)
        scratch_metrics = run_posttraining(
            model=micro_model,
            adapter=micro_adapter,
            probe=micro_probe,
            tokenizer=micro_tok,
            train_examples=gsm_train_micro,
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
        scratch_telemetry = {
            "scratch_pretrain_initial_loss": round(pre_init_scratch, 4),
            "scratch_pretrain_final_loss": round(pre_final_scratch, 4),
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

        print(f"[Modal Remote] Loading {hf_model} weights...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(hf_model)
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        model = Qwen3CausalLM.from_huggingface(
            repo_id=hf_model,
            device=device,
            dtype=dtype,
        )
        train_mode = "continued"
        unfreeze_layers = 4

    adapter = TorchContinuousLatentAdapter(
        hidden_dim=model.config.hidden_dim,
        rms_norm_eps=model.config.rms_norm_eps,
    ).to(device=device, dtype=torch.float32)
    probe = TorchSidecarProbe(
        hidden_dim=model.config.hidden_dim,
        templates=PROBE_STAGE_TEMPLATES,
        rms_norm_eps=model.config.rms_norm_eps,
    ).to(device=device, dtype=torch.float32)

    # Stage 1: Pretraining
    print(f"[Modal Remote] Stage 1: Running {train_mode} causal LM pretraining ({pretrain_steps} steps)...", flush=True)
    pre_init, pre_final, _ = run_pretraining(
        model=model,
        tokenizer=tokenizer,
        train_examples=train_pool,
        device=device,
        mode=train_mode,
        steps=pretrain_steps,
        batch_size=8,
        lr=8e-5 if train_mode == "continued" else 2e-3,
        unfreeze_top_layers=unfreeze_layers,
    )

    # Stage 2: Post-Training
    print(f"[Modal Remote] Stage 2: Multi-regime SFT + Latent recurrence (SFT={sft_steps}, Latent={latent_steps}, Probe={probe_steps})...", flush=True)
    stage_metrics = run_posttraining(
        model=model,
        adapter=adapter,
        probe=probe,
        tokenizer=tokenizer,
        train_examples=train_pool,
        device=device,
        mode=train_mode,
        sft_steps=sft_steps,
        latent_steps_train=latent_steps,
        probe_steps=probe_steps,
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

    opik.flush_tracker()
    return output_doc


@app.local_entrypoint()
def main(
    mode: str = "hybrid",
    hf_model: str = "Qwen/Qwen3-0.6B-Base",
    train_samples: int = 48,
    eval_samples: int = 24,
    full_eval: bool = False,
    gpu: str = "",
    pretrain_steps: int = 24,
    sft_steps: int = 72,
    latent_steps: int = 96,
    probe_steps: int = 60,
):
    """Local CLI entrypoint invoking remote Modal GPU function."""
    import json
    from scripts.export_web_artifacts import export_to_web

    fn = run_gpu_ablation
    active_gpu = SELECTED_GPU
    if gpu:
        # with_options supports a single GPU name (e.g. "A10G", "A100", "L4", "T4")
        chosen = gpu.split(",")[0].strip()
        fn = run_gpu_ablation.with_options(gpu=chosen)
        active_gpu = chosen

    print(f"[Modal Local] Launching remote GITLTYN ablation on GPU ({active_gpu})...")
    result = fn.remote(
        mode=mode,
        hf_model=hf_model,
        train_samples=train_samples,
        eval_samples=eval_samples,
        full_eval=full_eval,
        pretrain_steps=pretrain_steps,
        sft_steps=sft_steps,
        latent_steps=latent_steps,
        probe_steps=probe_steps,
    )

    out_file = EXPERIMENTS_DIR / "artifacts" / "qwen3_0.6b_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"[Modal Local] Saved GPU results to {out_file}")

    web_path = export_to_web()
    print(f"[Modal Local] Exported to web artifact: {web_path}")
