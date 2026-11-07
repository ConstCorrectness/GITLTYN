---
description: Audit paper.mdx for academic tone, LaTeX notation consistency, and strict alignment with exported JSON data and PyTorch code.
---

Perform a rigorous academic and technical audit of `web/src/content/paper.mdx` (or the specific section specified in `$ARGUMENTS`).

### Audit Checklist

1. **Zero Hardcoded Metric Drift (Cross-Check with `web/src/data/`)**
   - Extract every quantitative claim, percentage, token reduction ratio, accuracy score, and latency figure mentioned in the prose of `paper.mdx`.
   - Cross-reference each number against the ground-truth JSON artifacts in `web/src/data/ablation_frontier.json` and `web/src/data/latent_tee_traces.json`.
   - Flag any prose claim that deviates from the exported JSON data by even 0.1%, or replace hardcoded numbers with dynamic MDX data bindings where appropriate.

2. **LaTeX & Mathematical Notation Consistency**
   - Verify that every equation and inline math symbol strictly obeys the canonical notation in `CLAUDE.md` and matches the tensor shapes in `experiments/src/latent_loop.py`:
     - Transformer layer count: $L$, hidden dimension: $d$, vocabulary size: $\vert{}V\vert{}$
     - Hidden state at layer $l$, step $t$: $h_t^{(l)} \in \mathbb{R}^d$
     - Continuous latent thought vector at recurrent step $k \in \{1 \dots K\}$: $z_k \in \mathbb{R}^d$
     - Latent projection adapter: $f_{\theta}: \mathbb{R}^d \to \mathbb{R}^d$
     - Vocabulary unembedding matrix: $W_{\text{unembed}} \in \mathbb{R}^{d \times \vert{}V\vert{}}$
     - Detached Sidecar Tee decoder: $g_{\phi}(z_{1:K})$ (must explicitly indicate stop-gradient $\text{sg}[z_k]$ when describing training).
   - Check that no raw `<` or `>` characters appear outside math blocks or code fences in `.mdx` (which will crash the MDX/JSX compiler).

3. **Academic Tone & Precision (Distill / NeurIPS Bar)**
   - Eliminate vague, anthropomorphic, or hype-driven phrasing (e.g., "the model magically understands", "incredibly fast", "revolutionary"). Replace with mechanistic, falsifiable statements (e.g., "bypassing $W_{\text{unembed}}$ during the $K$ state-transition steps reduces memory-bandwidth bound operations by $68.4\%$").
   - Ensure the distinction between **syntactic fluency tokens** (determiners, conjunctions, conversational scaffolding) and **state-transition tokens** (entities, operators, intermediate values) is maintained consistently.

4. **Interactive Scrollytelling & Margin Hygiene**
   - Verify that every `<EpochExplorer />`, `<SyntaxPrunerDemo />`, and `<LatentTeeInspector />` component embedded in `paper.mdx` is explicitly referenced in the surrounding prose with a concrete takeaway explaining what the reader should observe when toggling the controls.
   - Move tangential implementation details, hyperparameter tables, or long derivations out of the main `65ch` prose flow and into `<Sidenote>` components so narrative momentum stays tight.
   - Verify all `[@citation_key]` references exist in `web/src/content/references.bib`.

### Output Format
First, present a structured **Audit Report** grouped by `[DATA DRIFT]`, `[NOTATION]`, `[TONE]`, and `[LAYOUT]` with exact line numbers and proposed diffs. Ask for confirmation before applying edits to `web/src/content/paper.mdx`.

