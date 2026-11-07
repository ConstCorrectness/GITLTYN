---
description: Validate raw PyTorch ablation logs, compute frontier metrics, and export schema-compliant JSON to the Astro web app.
---

Synchronize experimental ablation outputs from `experiments/` into `web/src/data/` by executing the following pipeline. Do NOT manually edit or hand-write JSON files inside `web/src/data/`.

### Target Scope
Optional filter argument: `$ARGUMENTS` (e.g., `gsm8k`, `math`, `latent-tee`, or empty for all suites).

### Execution Steps

1. **Inspect Raw Evaluation Logs**
   - Scan `experiments/outputs/` (or the specific run directory matching `$ARGUMENTS`) for the latest evaluation artifacts across all 4 regimes:
     - `verbose_cot` (Baseline)
     - `telegraphic_cot` (Ablation A: Syntactic Pruning)
     - `compute_matched` (Ablation B: Fixed FLOP/Step Budget)
     - `latent_loop` (Ablation C: Continuous Latent State Transitions + Latent Tee)
   - Verify that every run log records: `model_id`, `regime`, `task`, `accuracy`, `ci_95_low`, `ci_95_high`, `mean_tokens_emitted`, `latent_steps_k`, `kv_cache_bytes`, and `wall_clock_ms`.

2. **Update & Run the Python Exporter**
   - If new metrics or trace fields were added in `experiments/src/`, update `experiments/scripts/export_web_artifacts.py` and `shared/schema.json` first.
   - Ensure the exporter pre-computes the **Epoch-style transformations** in Python so the browser doesn't do heavy math at runtime:
     - Logit transformation of accuracy: $\text{logit}(p) = \ln\left(\frac{p}{1 - p}\right)$ (clamped to $[\epsilon, 1-\epsilon]$).
     - Pareto frontier boolean flags (`is_pareto_frontier: bool`) for Accuracy vs. FLOPs and Accuracy vs. KV-Cache bytes.
     - Top-$k$ Logit Lens vocabulary projections and detached Sidecar Probe strings for the `latent_tee_traces.json` artifact.
   - Run the exporter and schema validator:
     `cd experiments && uv run pytest tests/test_schema_export.py && uv run python scripts/export_web_artifacts.py $ARGUMENTS`

3. **Verify Frontend Type-Safety & Build**
   - Confirm that the exported JSON files in `web/src/data/` match the TypeScript interfaces in `web/src/types/ablation.ts`. If the schema evolved, update `ablation.ts`.
   - Run the Astro typechecker to ensure no React 19 (.tsx) chart islands broke:
     `cd web && pnpm astro check`

4. **Report Diff Summary**
   - Output a concise Markdown table comparing the previous vs. updated frontier metrics (Accuracy, Token Reduction %, and KV-Cache Savings %) across the 4 ablation regimes so the author can verify the update at a glance.

