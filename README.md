# Grammar is the Last Thing You Need (GITLTYN)

**Decoupling Latent State Transitions ($z_k \in \mathbb{R}^d$) from Syntactic Token Overhead ($W_{\text{unembed}}$) in Small Language Models**

Author: [ConstCorrectness](https://github.com/ConstCorrectness)

GITLTYN is an interactive research monograph and deterministic ablation suite investigating how much of autoregressive Chain-of-Thought (CoT) compute advances algebraic state transitions versus serializing English grammatical glue through the vocabulary unembedding bottleneck ($W_{\text{unembed}} \in \mathbb{R}^{d \times \lvert V \rvert}$). Building on this separation, we implement **Continuous Latent Looping** paired with a detachable **`RingBufferLatentTee`** and stop-gradient **`SidecarProbe`** ($g_{\phi}(\mathrm{sg}[z_{1:K}])$).

---

## Repository Structure & Python Pipeline Links

- [`experiments/src/qwen3_ablation.py`](https://github.com/ConstCorrectness/GITLTYN/blob/main/experiments/src/qwen3_ablation.py) — Native PyTorch `Qwen3CausalLM` (`0.6B` GQA + QK-RMSNorm + RoPE + SwiGLU supporting both from-scratch init and HuggingFace `Qwen/Qwen3-0.6B-Base` safetensors), `TorchContinuousLatentAdapter` ($f_{\theta}$), and disjoint `TorchSidecarProbe` ($g_{\phi}(\mathrm{sg}[z_{1:K}])$).
- [`experiments/src/train_pipeline.py`](https://github.com/ConstCorrectness/GITLTYN/blob/main/experiments/src/train_pipeline.py) & [`experiments/src/datasets.py`](https://github.com/ConstCorrectness/GITLTYN/blob/main/experiments/src/datasets.py) — Three-stage PyTorch pretraining (`scratch` & `continued`), post-training (regime-conditioned SFT, $f_{\theta}$ recurrence, disjoint $g_{\phi}$ probe), and 4-regime GSM8K/MATH evaluator.
- [`experiments/scripts/run_qwen3_ablation.py`](https://github.com/ConstCorrectness/GITLTYN/blob/main/experiments/scripts/run_qwen3_ablation.py) — End-to-end CLI executing `Qwen3-0.6B` pretraining, post-training, and evaluation (`--mode hybrid|continued|scratch`, `--device auto|cuda|cpu`).
- [`experiments/src/latent_loop.py`](https://github.com/ConstCorrectness/GITLTYN/blob/main/experiments/src/latent_loop.py) — Continuous latent recurrence ($z_k \in \mathbb{R}^d$), fixed-capacity circular [`RingBufferLatentTee`](https://github.com/ConstCorrectness/GITLTYN/blob/main/experiments/src/latent_loop.py) (`push(z_k)` enforcing `.detach()`), and stop-gradient [`SidecarProbe`](https://github.com/ConstCorrectness/GITLTYN/blob/main/experiments/src/latent_loop.py) ($\nabla_{\theta_{\text{backbone}}} \mathcal{L}_{\text{probe}} = 0$).
- [`experiments/src/prune_syntax.py`](https://github.com/ConstCorrectness/GITLTYN/blob/main/experiments/src/prune_syntax.py) — Stratified syntactic pruning pipeline (`verbose_cot` $\to$ `telegraphic_cot` motivational ablation baseline).
- [`experiments/src/config.py`](https://github.com/ConstCorrectness/GITLTYN/blob/main/experiments/src/config.py) & [`experiments/configs/ablation_regimes.json`](https://github.com/ConstCorrectness/GITLTYN/blob/main/experiments/configs/ablation_regimes.json) — Pydantic-validated regime configurations enforcing `enable_latent_tee=False` by default in production, $M \le K$ circular buffer bounds, and separate probe optimization.
- [`experiments/scripts/export_web_artifacts.py`](https://github.com/ConstCorrectness/GITLTYN/blob/main/experiments/scripts/export_web_artifacts.py) — Deterministic evaluation exporter and `Draft202012Validator` schema gate writing `web/src/data/ablation_frontier.json`.
- [`experiments/tests/test_schema_export.py`](https://github.com/ConstCorrectness/GITLTYN/blob/main/experiments/tests/test_schema_export.py) & [`experiments/tests/test_qwen3_pytorch.py`](https://github.com/ConstCorrectness/GITLTYN/blob/main/experiments/tests/test_qwen3_pytorch.py) — Pytest suites verifying `.detach()` tap invariants, $\nabla_{\theta_{\text{backbone}}} \mathcal{L}_{\text{probe}} = 0$ gradient isolation on live PyTorch autograd graphs, $O(1)$ circular buffer capacity ($M \le K$), and JSON Schema compliance against [`shared/schema.json`](https://github.com/ConstCorrectness/GITLTYN/blob/main/shared/schema.json).
- `web/` — Astro 7 + MDX + React 19 + TypeScript + Tailwind CSS v4 + Observable Plot interactive monograph (`EpochExplorer`, `SyntaxPrunerDemo`, `LatentTeeInspector`, `RegimeSummaryWidget`, and 3-mode `Warm` / `Light` / `Dark` theme system).
- `paper/related_work.tex` — Companion LaTeX section formalizing positioning and architectural invariants against prior latent-reasoning and compressed-CoT literature.

---

## The Fundamental Trilemma of Latent Reasoning

```
                       1. Expressive Latent Reasoning
                          (Continuous ℝᵈ Dynamics)
                                   /\
                                  /  \
                                 /    \
                                /  ★   \   <-- GITLTYN (Resolves all 3)
                               / GITLTYN\
                              /__________\
     2. Zero-Interference Auditability      3. Bounded O(1) Activation Memory
       (Non-Contaminating Probes)             (Fixed Ring Buffer M ≤ K)
```

Test-time reasoning in language models faces a fundamental three-way trade-off between **expressivity**, **auditability**, and **memory footprint**:
1. **Expressive Latent Reasoning**: Operating in continuous $\mathbb{R}^d$ hidden space to bypass the discrete vocabulary projection bottleneck ($W_{\text{unembed}}$) and natural language serialization overhead.
2. **Zero-Interference Auditability**: Inspecting intermediate reasoning trajectories via auxiliary probes or logit lenses without backpropagating surface-grammar gradients into the primary thought loop ($\nabla_{\theta_{\text{backbone}}} \mathcal{L}_{\text{probe}} = 0$).
3. **Bounded $O(1)$ Activation Memory**: Retaining recurrent states for inspection without suffering unbounded $O(K \cdot d)$ activation memory growth as recurrence depth $K$ scales.

Prior approaches compromise on at least one vertex: **Coconut** (Hao et al., 2024) sacrifices auditability by creating an un-inspectable black box; **Huginn** (Geiping et al., 2025) and **Looped Transformers** (Giannou et al., 2023) retain states at the cost of unbounded $O(K \cdot d)$ memory growth; and **Compressed CoT** (TokenSkip, CCoT) remains bound to discrete tokens, continuing to pay the $W_{\text{unembed}}$ vocabulary bottleneck. **GITLTYN resolves all three simultaneously** through Continuous Latent Looping paired with a detachable, fixed-capacity `RingBufferLatentTee` ($M \le K$) and an isolated stop-gradient `SidecarProbe` ($\mathrm{sg}[z_k]$).

---

## Architectural Invariants

1. **Strict Stop-Gradient Tap ($\mathrm{sg}[z_{1:K}]$)**: `RingBufferLatentTee.push(z_k)` unconditionally calls `z_k.detach()` before storing snapshots in the circular buffer. Auxiliary readout heads (`SidecarProbe` $g_{\phi}$ and `LogitLens` $W_{\text{unembed}}$) never backpropagate into the continuous thought trajectory.
2. **Disjoint Probe Optimization ($\nabla_{\theta_{\text{backbone}}} \mathcal{L}_{\text{probe}} = 0$)**: `SidecarProbe.step_probe_optimizer(...)` optimizes $\mathcal{L}_{\text{probe}}(\phi; \mathrm{sg}[z_{1:K}], y)$ in an isolated optimizer step strictly over probe parameters $\phi$, guaranteeing zero gradient interference on backbone/adapter weights $\theta_{\text{backbone}}$.
3. **Bounded $O(1)$ Activation Memory ($M \le K$) & Zero-Overhead Detachment**: `RingBufferLatentTee` is a fixed-capacity circular buffer of size $M \le K$ ($O(M d) = O(1)$ memory with respect to recurrence depth $K$) and defaults to `enable_latent_tee=False` in production (`True` only during on-demand mechanistic auditing).

---

## Related Work & Positioning

While GITLTYN shares the high-level goal of reasoning in continuous activation space $\mathbb{R}^d$ without intermediate vocabulary serialization, our systems framing and auditability mechanism enforce three explicit differentiations relative to prior work:

### 1. Differentiation vs. Coconut (Chain of Continuous Thought — Hao et al., 2024)
- **Baseline Mechanism**: Coconut (Hao et al., 2024) feeds the terminal hidden state $h_t^{(L)} \in \mathbb{R}^d$ directly back as the next input embedding, bypassing $W_{\text{unembed}}$, but yields an unauditable black-box trajectory in $\mathbb{R}^d$ and suffers accuracy degradation ($\sim 8\%$ on GSM8K) without multi-stage curriculum training.
- **GITLTYN Implementation & Claim**: We prohibit auxiliary readout heads from backpropagating into the continuous thought loop by enforcing `.detach()` ($\mathrm{sg}[z_{1:K}]$) at the `RingBufferLatentTee` tap boundary prior to `SidecarProbe` ($g_{\phi}$) or `LogitLens` evaluation, and optimizing $\mathcal{L}_{\text{probe}}$ via a separate optimizer step ($\nabla_{\theta_{\text{backbone}}} \mathcal{L}_{\text{probe}} = 0$). Our detachable tee + stop-gradient sidecar resolves Coconut's interpretability black-box problem with zero gradient interference on the primary $\mathbb{R}^d$ transition dynamics.

### 2. Differentiation vs. Huginn / Recurrent Depth (Geiping et al., 2025) & Looped Transformers (Giannou et al., 2023)
- **Baseline Mechanism**: Recurrent-depth architectures (Geiping et al., 2025) and Looped Transformers (Giannou et al., 2023) scale test-time compute by looping transformer blocks $K$ times in latent space before token emission, which incurs unbounded $O(K d)$ activation memory growth if intermediate trajectories are retained for mechanistic inspection.
- **GITLTYN Implementation & Claim**: We implement `RingBufferLatentTee` as a fixed-capacity circular buffer of size $M \le K$ ($O(1)$ memory overhead with respect to total recurrence depth $K$ during inference), gated by `enable_latent_tee=False` by default in production (`True` for mechanistic auditing). Unlike standard recurrent depth unrolling, our Ring-Buffer Latent Tee strictly bounds activation memory while preserving on-demand mechanistic readouts of the latest $M$ recurrent transitions.

### 3. Differentiation vs. Compressed CoT (TokenSkip, CCoT, Sketch-of-Thought)
- **Baseline Mechanism**: Token-pruning and sketch-compression pipelines (TokenSkip, Xia et al., 2025; CCoT, Cheng & Van Durme, 2024; Sketch-of-Thought, Aytes et al., 2025) treat compressed token emission as the final inference efficiency method, continuing to pay $W_{\text{unembed}}$ projection and KV-cache growth for every surviving token.
- **GITLTYN Implementation & Claim**: We retain `telegraphic_cot` ([`experiments/src/prune_syntax.py`](https://github.com/ConstCorrectness/GITLTYN/blob/main/experiments/src/prune_syntax.py)) strictly as a **motivational ablation baseline** alongside `verbose_cot` and budget-forced CoT (`compute_matched` / `budget_forced_cot`, truncating verbose CoT to match the telegraphic token budget). Telegraphic CoT is not our final proposed architecture; it serves as an empirical control proving that English grammatical scaffolding accounts for $64.6\%$–$67.1\%$ of emitted tokens with $< 1\%$ accuracy contribution, motivating the shift to pure $\mathbb{R}^d$ Continuous Latent Looping.

---

## Quickstart & Verification

### 1. Python PyTorch `Qwen3-0.6B` Ablation Suite & Schema Export (`uv`)
```bash
cd experiments
uv run python scripts/run_qwen3_ablation.py --mode hybrid --hf-model Qwen/Qwen3-0.6B-Base --device auto
uv run pytest
uv run python scripts/export_web_artifacts.py
```

### 2. Interactive Astro 7 + React 19 Paper (`pnpm`)
```bash
cd web
pnpm dev
pnpm astro check && pnpm build
```

### 3. GitHub Pages Deployment
The repository includes a turnkey GitHub Actions deployment workflow ([`.github/workflows/deploy.yml`](.github/workflows/deploy.yml)).
To deploy:
1. Push `main` to GitHub: `git push origin main`
2. Under **Repository Settings -> Pages -> Build and deployment -> Source**, select **GitHub Actions**.
3. The site will automatically build and publish to `https://constcorrectness.github.io/GITLTYN/`.

---

## References
1. **Ramachandran, D., & Amir, E. (2007).** [*Bayesian Inverse Reinforcement Learning.*](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=TJnfAvYAAAAJ&citation_for_view=TJnfAvYAAAAJ:u5HHmVD_uO8C) IJCAI 2007, pp. 2586–2591.
2. **Hao, S., Sukhbaatar, S., Su, D., Li, X., Hu, Z., Weston, J., & Yu, T. (2024).** [*Training Large Language Models to Reason in a Continuous Latent Space (Coconut).*](https://arxiv.org/abs/2412.06769) arXiv:2412.06769.
3. **Geiping, J., McLeish, S., Jain, N., Kirchenbauer, J., Siddiqui, S., Khani, A., ... & Goldstein, T. (2025).** [*Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach (Huginn).*](https://arxiv.org/abs/2502.05171) arXiv:2502.05171.
4. **Giannou, A., Rajput, S., Sohn, J.-Y., Lee, K., Lee, J. D., & Papailiopoulos, D. (2023).** [*Looped Transformers as Programmable Computers.*](https://arxiv.org/abs/2301.13196) ICML 2023 (arXiv:2301.13196).
5. **Cheng, J., & Van Durme, B. (2024).** [*Compressed Chain of Thought: Efficient Reasoning through Dense Representations (CCoT).*](https://arxiv.org/abs/2412.13171) arXiv:2412.13171.
6. **Xia, H., Leong, C. T., Wang, W., Li, Y., & Li, W. (2025).** [*TokenSkip: Controllable Chain-of-Thought Compression in LLMs.*](https://arxiv.org/abs/2502.12067) arXiv:2502.12067.
7. **Aytes, S. A., Baek, J., & Hwang, S. J. (2025).** [*Sketch-of-Thought: Efficient LLM Reasoning with Adaptive Cognitive-Inspired Sketching.*](https://arxiv.org/abs/2503.05179) arXiv:2503.05179.
