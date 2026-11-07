---
description: Scaffold a new Epoch-style interactive React 19 + D3/Observable Plot visualization island in TypeScript.
---

Create a new interactive React 19 + TypeScript + D3/Observable Plot component in `web/src/components/interactive/$ARGUMENTS.tsx`.

### Requirements
1. **React 19 + Strict TypeScript**:
   - Define an explicit `Props` interface and type all state/data using interfaces imported from `@/types/ablation` (including `REGIME_META` for consistent semantic colors).
   - Use standard React hooks (`useState`, `useMemo`, `useRef`, `useEffect`) — never use `any` or untyped JSON objects.
2. **Epoch AI Control Header**:
   - Include clean UI controls at the top of the card:
     - Metric selector dropdown (e.g., `mean_tokens_emitted`, `flops_per_problem`, `kv_cache_kb`).
     - Scale toggles (`Linear` vs. `Log` X-axis; `Linear` vs. `Logit` Y-axis).
     - Filter toggle (`Pareto Frontier Only` checkbox) and view switcher (`Graph` vs. sortable `Table`).
3. **Observable Plot / D3 Rendering**:
   - Mount the Observable Plot / D3 SVG inside a `useRef<HTMLDivElement>(null)` container inside `useEffect`, and always call `plot.remove()` in the cleanup return function to prevent duplicate SVGs on re-render.
   - Support responsive resizing via `ResizeObserver` and render crisp SVG axes with academic typography and 95% confidence interval (`ci_95`) error bars/bands.
4. **Data Contract & Export**:
   - Import data exclusively from `@/data/` (never hardcode benchmark numbers).
   - Include a toolbar button to download the current view as SVG or filtered CSV.
5. **Verification**:
   - After creating the component, run `cd web && pnpm astro check` to verify zero TypeScript errors.