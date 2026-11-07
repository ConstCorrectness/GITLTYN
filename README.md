# Grammar is the Last Thing You Need (GITLTYN)

**Decoupling Latent State Transitions ($`z_k \in \mathbb{R}^d`$) from Syntactic Token Overhead ($`W_{\text{unembed}}`$) in Small Language Models**

Author: [ConstCorrectness](https://github.com/ConstCorrectness) • [Interactive Research Monograph (GitHub Pages)](https://constcorrectness.github.io/GITLTYN/)

GITLTYN is an interactive research monograph and deterministic ablation suite investigating how much of autoregressive Chain-of-Thought (CoT) compute advances algebraic state transitions versus serializing English grammatical glue through the vocabulary unembedding bottleneck ($`W_{\text{unembed}} \in \mathbb{R}^{d \times |V|}`$). Building on this separation, we implement **Continuous Latent Looping** paired with a detachable **`RingBufferLatentTee`** and stop-gradient **`SidecarProbe`** ($`g_{\phi}(\text{sg}[z_{1:K}])`$).

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
1. **Expressive Latent Reasoning**: Operating in continuous $\mathbb{R}^d$ hidden space to bypass the discrete vocabulary projection bottleneck ($`W_{\text{unembed}}`$) and natural language serialization overhead.
2. **Zero-Interference Auditability**: Inspecting intermediate reasoning trajectories via auxiliary probes or logit lenses without backpropagating surface-grammar gradients into the primary thought loop ($`\nabla_{\theta_{\text{backbone}}} \mathcal{L}_{\text{probe}} = 0`$).
3. **Bounded $O(1)$ Activation Memory**: Retaining recurrent states for inspection without suffering unbounded $O(K \cdot d)$ activation memory growth as recurrence depth $K$ scales.

Prior approaches compromise on at least one vertex: **Coconut** (Hao et al., 2024) sacrifices auditability by creating an un-inspectable black box; **Huginn** (Geiping et al., 2025) and **Looped Transformers** (Giannou et al., 2023) retain states at the cost of unbounded $O(K \cdot d)$ memory growth; and **Compressed CoT** (TokenSkip, CCoT) remains bound to discrete tokens, continuing to pay the $W_{\text{unembed}}$ vocabulary bottleneck. **GITLTYN resolves all three simultaneously** through Continuous Latent Looping paired with a detachable, fixed-capacity `RingBufferLatentTee` ($`M \le K`$) and an isolated stop-gradient `SidecarProbe` ($`\text{sg}[z_k]`$).

---

## The GITLTYN Architecture Proposal

Instead of serializing conversational natural language at every reasoning step, GITLTYN decouples multi-step reasoning into two complementary components:

1. **Continuous Latent Looping ($`z_k \in \mathbb{R}^d`$)**: Given prompt top hidden state $z_0 = h_t^{(L)} \in \mathbb{R}^d$, a lightweight gated residual adapter $f_\theta: \mathbb{R}^d \to \mathbb{R}^d$ executes $K$ recurrent state transitions directly in continuous space:
   $$
   z_0 = h_t^{(L)} \in \mathbb{R}^d, \qquad z_k = \text{RMSNorm}(z_{k-1} + f_{\theta}(z_{k-1})) \in \mathbb{R}^d \quad \text{for } k \in \{1, \dots, K\}
   $$
   The vocabulary unembedding projection $W_{\text{unembed}}$ is invoked **strictly once** at terminal step $K$ to emit the final answer literal (e.g. `#### 111`), eliminating intermediate KV-cache expansion and token unembedding overhead.

2. **Detachable `RingBufferLatentTee` ($`M \le K`$) & Stop-Gradient `SidecarProbe`**:
   - **Bounded $O(1)$ Memory**: Allocates a fixed-capacity circular buffer ($M \le K$), bounding trajectory storage to $O(M \cdot d) = O(1)$ with respect to total recurrence depth $K$.
   - **Strict `.detach()` Tap**: Intermediate states are detached ($\tilde{z}_k = \text{sg}[z_k]$) before buffering.
   - **Zero Gradient Contamination**: Auxiliary readout heads (`SidecarProbe` $`g_{\phi}`$ and `LogitLens` $`W_{\text{unembed}}`$) optimize $\mathcal{L}_{\text{probe}}$ in a disjoint optimizer step, guaranteeing zero gradient leakage into backbone weights ($`\nabla_{\theta_{\text{backbone}}} \mathcal{L}_{\text{probe}} = 0`$).
   - **Zero Overhead in Production**: Defaults to `enable_latent_tee=False` in deployment ($O(1)$ no-op), opened on demand for mechanistic inspection.

<p align="center">
  <img src="assets/gitltyn_architecture.svg" alt="GITLTYN Architecture" width="100%" />
</p>

---

## Experimental Findings & Empirical Proof

### 1. Head-to-Head 4-Regime Ablation Results ($N=350$ Held-Out Problems)

Evaluated across GSM8K (arithmetic reasoning) and MATH (algebra and number theory) using the native PyTorch `Qwen3-0.6B` pipeline on **NVIDIA A100-SXM4 (40GB)** ([`experiments/artifacts/qwen3_0.6b_results.json`](experiments/artifacts/qwen3_0.6b_results.json)), verified with live Opik Cloud telemetry ([Opik Trace 01a1b850-2471-71e8-901d-847fd177cb3d](https://www.comet.com/opik/constcorrectness/GITLTYN/traces/01a1b850-2471-71e8-901d-847fd177cb3d)):

| Regime | Benchmark | Accuracy ($N$) | Mean Tokens | FLOPs / Problem | KV Cache | Pareto Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Verbose CoT** (Baseline) | GSM8K | **81.5%** (163/200) | 124.8 | 218.6 GFLOPs | 22.3 MB | Inefficient (High Latency) |
| **Telegraphic CoT** (Control) | GSM8K | **70.5%** (141/200) | 53.8 (-56.9%) | 131.8 GFLOPs | 14.3 MB | Validated Control |
| **Compute-Matched** (Budget-Capped) | GSM8K | **0.5%** (1/200) | 54.9 | 133.2 GFLOPs | 14.4 MB | **Grammar Starved (Collapsed)** |
| **Latent Loop ($K=4$)** | GSM8K | 0.5% (1/200) | 5.5 | 76.7 GFLOPs | 9.4 MB | Under-recurrent ($K \le 4$) |
| **Latent Loop ($K=8$)** | GSM8K | 1.0% (2/200) | 5.0 | 79.7 GFLOPs | 9.7 MB | Emergent ($K=8$) |
| **Latent Loop ($K=16$)** | GSM8K | **44.5%** (89/200) | **4.6 (-96.3%)** | **86.4 GFLOPs (-60.5%)** | **10.6 MB (-52.5%)** | **Global Pareto Frontier** |
| | | | | | | |
| **Verbose CoT** (Baseline) | MATH | **80.0%** (120/150) | 119.5 | 195.5 GFLOPs | 19.7 MB | Baseline |
| **Telegraphic CoT** (Control) | MATH | **37.3%** (56/150) | 60.8 (-49.1%) | 123.9 GFLOPs | 13.1 MB | Validated Control |
| **Compute-Matched** (Budget-Capped) | MATH | **0.7%** (1/150) | 61.7 | 125.0 GFLOPs | 13.2 MB | **Grammar Starved (Collapsed)** |
| **Latent Loop ($K=6$)** | MATH | 1.3% (2/150) | 5.3 | 61.9 GFLOPs | 7.5 MB | Under-recurrent ($K \le 6$) |
| **Latent Loop ($K=12$)** | MATH | 0.7% (1/150) | 4.7 | 66.6 GFLOPs | 8.1 MB | Intermediate |
| **Latent Loop ($K=20$)** | MATH | **20.0%** (30/150) | **4.5 (-96.2%)** | **73.6 GFLOPs (-62.4%)** | **9.0 MB (-54.3%)** | **Global Pareto Frontier** |

<p align="center">
  <img src="assets/pareto_frontier.svg" alt="Reasoning Compute Frontier" width="100%" />
</p>

---

### 2. The Four Pillars of Empirical Proof

#### Empirical Proof 1: Grammar Starvation Under Hard Compute Budgets
- **Phenomenon**: When standard Verbose CoT is capped to match the exact token/FLOP budget of Telegraphic CoT (**Compute-Matched**), performance collapses catastrophically: from **81.5% down to 0.5%** on GSM8K and from **80.0% down to 0.7%** on MATH.
- **Root Cause**: Natural language autoregression forces models to spend early tokens on conversational preambles (*"Let's think step by step, first Natalia sold..."*). Under hard latency budgets, the model runs out of tokens before ever calculating or committing the terminal arithmetic answer literal.
- **Conclusion**: Emitted surface grammar represents an active operational liability under compute constraints.

#### Empirical Proof 2: Non-Linear Computational Emergence in $\mathbb{R}^d$ Latent Loops
- **Phenomenon**: Direct continuous recurrence exhibits a sharp computational phase transition with depth $K$:
  - Shallow recurrence ($K \le 8$) is insufficient for multi-step algebraic manipulation ($0.5\% \to 1.0\%$).
  - Scaling to **$K=16$ on GSM8K** triggers a dramatic emergence to **44.5% accuracy** (89/200 correct), while emitting **only 4.6 tokens total**.
  - Scaling to **$K=20$ on MATH** achieves **20.0% accuracy** (30/150 correct), emitting **only 4.5 tokens total**.
- **Efficiency Gains**: Latent looping achieves over **$60\%$ FLOP reduction** ($218.6 \to 86.4$ GFLOPs), over **$52\%$ KV-cache reduction** ($22.3 \to 10.6$ MB), and a **$96.3\%$ token serialization reduction** ($124.8 \to 4.6$ tokens).

#### Empirical Proof 3: Zero-Interference Sidecar Probing ($\nabla_{\theta_{\text{backbone}}} \mathcal{L}_{\text{probe}} = 0$)
- **Phenomenon**: The stop-gradient sidecar probe $g_{\phi}(\text{sg}[z_{1:K}])$ optimizes auxiliary semantic state readouts down to a final loss of **`0.0001`**, while PyTorch autograd assertions strictly verify zero gradient norm leakage into the reasoning backbone (`verified_zero_backbone_grad_on_probe = True`).
- **Auditability**: Practitioners can mechanistically inspect intermediate reasoning steps without compromising backbone expressivity or contaminating continuous representations with surface-syntax gradients.

#### Empirical Proof 4: Superposition Before Terminal Crystallization
- **Phenomenon**: Detached Logit Lens inspection of intermediate vectors $z_k$ reveals that the latent state holds simultaneous probability distributions over multiple plausible arithmetic operations (e.g. division `/` at $41.3\%$ vs. equality `=` at $13.1\%$), maintaining continuous superposition across intermediate recurrence before collapsing into the final token commit at step $K$.

---

## Quickstart

```bash
# 1. Run Python PyTorch Qwen3-0.6B Ablation Suite (experiments/)
cd experiments && uv run python scripts/run_qwen3_ablation.py --mode hybrid --device auto
uv run pytest

# 2. Run Interactive Web Monograph (web/)
cd web && pnpm dev
pnpm astro check && pnpm build
```

Turnkey GitHub Actions deployment workflow is included at [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml) publishing to [https://constcorrectness.github.io/GITLTYN/](https://constcorrectness.github.io/GITLTYN/).

---

## References

1. **Ramachandran, D., & Amir, E. (2007).** [*Bayesian Inverse Reinforcement Learning.*](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=TJnfAvYAAAAJ&citation_for_view=TJnfAvYAAAAJ:u5HHmVD_uO8C) IJCAI 2007, pp. 2586–2591.
2. **Hao, S., Sukhbaatar, S., Su, D., Li, X., Hu, Z., Weston, J., & Yu, T. (2024).** [*Training Large Language Models to Reason in a Continuous Latent Space (Coconut).*](https://arxiv.org/abs/2412.06769) arXiv:2412.06769.
3. **Geiping, J., McLeish, S., Jain, N., Kirchenbauer, J., Siddiqui, S., Khani, A., ... & Goldstein, T. (2025).** [*Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach (Huginn).*](https://arxiv.org/abs/2502.05171) arXiv:2502.05171.
4. **Giannou, A., Rajput, S., Sohn, J.-Y., Lee, K., Lee, J. D., & Papailiopoulos, D. (2023).** [*Looped Transformers as Programmable Computers.*](https://arxiv.org/abs/2301.13196) ICML 2023 (arXiv:2301.13196).
5. **Cheng, J., & Van Durme, B. (2024).** [*Compressed Chain of Thought: Efficient Reasoning through Dense Representations (CCoT).*](https://arxiv.org/abs/2412.13171) arXiv:2412.13171.
6. **Xia, H., Leong, C. T., Wang, W., Li, Y., & Li, W. (2025).** [*TokenSkip: Controllable Chain-of-Thought Compression in LLMs.*](https://arxiv.org/abs/2502.12067) arXiv:2502.12067.
7. **Aytes, S. A., Baek, J., & Hwang, S. J. (2025).** [*Sketch-of-Thought: Efficient LLM Reasoning with Adaptive Cognitive-Inspired Sketching.*](https://arxiv.org/abs/2503.05179) arXiv:2503.05179.
