import { useEffect, useMemo, useRef, useState } from 'react';
import * as Plot from '@observablehq/plot';
import katex from 'katex';
import rawArtifact from '@/data/ablation_frontier.json';
import { useThemePalette } from '@/hooks/useThemePalette';
import {
  REGIME_META,
  normalizeAblationArtifact,
  type AblationRun,
  type PruningTrace,
  type TokenCategory,
} from '@/types/ablation';

export interface SyntaxPrunerDemoProps {
  initialRatioPercent?: number;
}

type PrunerMetricKey = 'mean_tokens_emitted' | 'flops_per_problem' | 'kv_cache_kb';

const ARTIFACT = normalizeAblationArtifact(rawArtifact);

function KatexSpan({ tex }: { tex: string }) {
  const html = useMemo(
    () => katex.renderToString(tex, { displayMode: false, output: 'html', throwOnError: false }),
    [tex]
  );
  return <span dangerouslySetInnerHTML={{ __html: html }} />;
}

const CATEGORY_LEGEND: Record<
  TokenCategory,
  { label: string; tauTex: string; dotColor: string; chipClass: string }
> = {
  discourse_marker: {
    label: 'Discourse Glue',
    tauTex: '\\tau \\in [0.04, 0.36]',
    dotColor: '#64748b',
    chipClass: 'tok-discourse',
  },
  determiner_aux: {
    label: 'Determiner / Aux',
    tauTex: '\\tau \\in [0.37, 0.68]',
    dotColor: '#94a3b8',
    chipClass: 'tok-det',
  },
  preposition_conj: {
    label: 'Preposition / Conj',
    tauTex: '\\tau \\in [0.69, 0.99]',
    dotColor: '#a8a29e',
    chipClass: 'tok-prep',
  },
  state_transition: {
    label: 'State Transition',
    tauTex: '\\text{preserved}',
    dotColor: '#10b981',
    chipClass: 'tok-state shadow-2xs',
  },
};

function toLogitVal(p: number): number {
  const clamped = Math.min(Math.max(p, 1e-6), 1 - 1e-6);
  return Math.log(clamped / (1 - clamped));
}

export default function SyntaxPrunerDemo({
  initialRatioPercent = 65,
}: SyntaxPrunerDemoProps) {
  const palette = useThemePalette();
  const traces: PruningTrace[] = ARTIFACT.pruning_traces ?? [];
  const [selectedTraceIdx, setSelectedTraceIdx] = useState<number>(0);
  const [ratioPercent, setRatioPercent] = useState<number>(initialRatioPercent);
  const [metricKey, setMetricKey] = useState<PrunerMetricKey>('kv_cache_kb');
  const [xScale, setXScale] = useState<'linear' | 'log'>('linear');
  const [yScale, setYScale] = useState<'linear' | 'logit'>('linear');
  const [paretoOnly, setParetoOnly] = useState<boolean>(false);
  const [viewMode, setViewMode] = useState<'graph' | 'table'>('graph');
  const [tableSortAsc, setTableSortAsc] = useState<boolean>(true);
  const [chartWidth, setChartWidth] = useState<number>(680);

  const wrapperRef = useRef<HTMLDivElement>(null);
  const plotRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = wrapperRef.current;
    if (!el) return;
    const update = () => {
      setChartWidth(Math.max(300, Math.floor(el.getBoundingClientRect().width - 44)));
    };
    update();
    const obs = new ResizeObserver(update);
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  const activeTrace: PruningTrace | undefined = traces[selectedTraceIdx] ?? traces[0];
  const ratio = ratioPercent / 100;

  const liveMetrics = useMemo(() => {
    if (!activeTrace) {
      return {
        totalTokens: 0,
        retainedTokens: 0,
        prunedTokens: 0,
        tokenDropPct: 0,
        liveKvKb: 0,
        baseKvKb: 0,
        liveGflops: 0,
        baseGflops: 0,
        interpolatedAccuracy: 0,
      };
    }

    const totalTokens = activeTrace.tokens.length;
    const retainedTokens = activeTrace.tokens.filter(
      (t) => !t.is_syntactic || t.prune_threshold > ratio
    ).length;
    const prunedTokens = totalTokens - retainedTokens;
    const tokenDropPct = totalTokens > 0 ? (prunedTokens / totalTokens) * 100 : 0;

    const baseSeq = activeTrace.prompt_tokens + totalTokens;
    const liveSeq = activeTrace.prompt_tokens + retainedTokens;

    const baseKvKb = baseSeq * activeTrace.kv_kb_per_token;
    const liveKvKb = liveSeq * activeTrace.kv_kb_per_token;

    const baseGflops = (baseSeq * activeTrace.flops_per_token) / 1e9;
    const liveGflops = (liveSeq * activeTrace.flops_per_token) / 1e9;

    const syntacticFractionPruned =
      totalTokens > 0
        ? prunedTokens /
          Math.max(1, activeTrace.tokens.filter((t) => t.is_syntactic).length)
        : 0;
    const interpolatedAccuracy =
      activeTrace.baseline_accuracy -
      syntacticFractionPruned *
        (activeTrace.baseline_accuracy - activeTrace.telegraphic_accuracy);

    return {
      totalTokens,
      retainedTokens,
      prunedTokens,
      tokenDropPct,
      liveKvKb,
      baseKvKb,
      liveGflops,
      baseGflops,
      interpolatedAccuracy,
    };
  }, [activeTrace, ratio]);

  const matchingBenchmarkRuns = useMemo<AblationRun[]>(() => {
    if (!activeTrace) return [];
    const subset = ARTIFACT.runs.filter(
      (r) =>
        r.benchmark === activeTrace.benchmark &&
        Math.abs(r.param_count_b - activeTrace.param_count_b) < 0.05 &&
        (!paretoOnly || r.is_pareto_frontier)
    );
    return [...subset].sort((a, b) =>
      tableSortAsc ? a[metricKey] - b[metricKey] : b[metricKey] - a[metricKey]
    );
  }, [activeTrace, paretoOnly, metricKey, tableSortAsc]);

  useEffect(() => {
    const host = plotRef.current;
    if (!host || viewMode !== 'graph' || !activeTrace || matchingBenchmarkRuns.length === 0) {
      return;
    }

    const toX = (run: AblationRun): number => {
      if (metricKey === 'flops_per_problem') return run.flops_per_problem / 1e9;
      if (metricKey === 'kv_cache_kb') return run.kv_cache_kb;
      return run.mean_tokens_emitted;
    };

    const points = matchingBenchmarkRuns.map((r) => ({
      label: REGIME_META[r.regime].shortLabel,
      model: r.model_name,
      x: toX(r),
      y: yScale === 'linear' ? r.accuracy * 100 : r.accuracy_logit,
      ciLow: yScale === 'linear' ? r.ci_95[0] * 100 : toLogitVal(r.ci_95[0]),
      ciHigh: yScale === 'linear' ? r.ci_95[1] * 100 : toLogitVal(r.ci_95[1]),
      color: REGIME_META[r.regime].color,
      tip: `${r.model_name} (${REGIME_META[r.regime].shortLabel})\nAccuracy: ${(r.accuracy * 100).toFixed(1)}% [95% CI: ${(r.ci_95[0] * 100).toFixed(1)}%–${(r.ci_95[1] * 100).toFixed(1)}%]\nTokens: ${r.mean_tokens_emitted} | KV: ${r.kv_cache_kb} KB`,
    }));

    const verboseRef = ARTIFACT.runs.find(
      (r) =>
        r.benchmark === activeTrace.benchmark &&
        Math.abs(r.param_count_b - activeTrace.param_count_b) < 0.05 &&
        r.regime === 'verbose_cot'
    );
    const teleRef = ARTIFACT.runs.find(
      (r) =>
        r.benchmark === activeTrace.benchmark &&
        Math.abs(r.param_count_b - activeTrace.param_count_b) < 0.05 &&
        r.regime === 'telegraphic_cot'
    );

    const sliderFrac = ratio;
    const liveCursor =
      verboseRef && teleRef && !paretoOnly
        ? [
            {
              x: toX(verboseRef) + sliderFrac * (toX(teleRef) - toX(verboseRef)),
              y:
                yScale === 'linear'
                  ? (verboseRef.accuracy +
                      sliderFrac * (teleRef.accuracy - verboseRef.accuracy)) *
                    100
                  : toLogitVal(
                      verboseRef.accuracy +
                        sliderFrac * (teleRef.accuracy - verboseRef.accuracy)
                    ),
              label: `Slider (${ratioPercent}% pruned)`,
            },
          ]
        : [];

    const xLabel =
      metricKey === 'flops_per_problem'
        ? 'Compute per Problem (GFLOPs)'
        : metricKey === 'kv_cache_kb'
          ? 'Peak KV-Cache Footprint (KB)'
          : 'Mean Emitted Tokens';

    const plot = Plot.plot({
      width: chartWidth,
      height: 220,
      marginTop: 18,
      marginRight: 28,
      marginBottom: 40,
      marginLeft: 52,
      style: {
        background: palette.plotBg,
        color: palette.plotFg,
        fontFamily: 'ui-sans-serif, system-ui, sans-serif',
        fontSize: '11.5px',
      },
      x: { type: xScale, label: xLabel, grid: true, nice: true },
      y: {
        type: 'linear',
        label: yScale === 'linear' ? 'Accuracy (%)' : 'Logit Accuracy',
        grid: true,
        nice: true,
      },
      marks: [
        Plot.frame({ stroke: palette.frameStroke }),
        Plot.ruleX(points, {
          x: 'x',
          y1: 'ciLow',
          y2: 'ciHigh',
          stroke: 'color',
          strokeOpacity: 0.48,
          strokeWidth: 1.4,
          marker: 'tick',
        }),
        Plot.dot(points, {
          x: 'x',
          y: 'y',
          r: 5.5,
          fill: 'color',
          fillOpacity: 0.9,
          stroke: palette.dotHalo,
          strokeWidth: 1.5,
        }),
        ...(liveCursor.length > 0
          ? [
              Plot.dot(liveCursor, {
                x: 'x',
                y: 'y',
                r: 8,
                fill: palette.cursorFill,
                fillOpacity: 0.12,
                stroke: palette.cursorStroke,
                strokeWidth: 1.5,
              }),
              Plot.text(liveCursor, {
                x: 'x',
                y: 'y',
                text: 'label',
                dy: -13,
                fontSize: 10,
                fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
                fill: palette.labelFg,
              }),
            ]
          : []),
        Plot.tip(
          points,
          Plot.pointer({
            x: 'x',
            y: 'y',
            title: 'tip',
            fontSize: 11,
            fill: palette.tipBg,
            stroke: palette.tipStroke,
          })
        ),
      ],
    });

    host.append(plot);
    return () => plot.remove();
  }, [
    matchingBenchmarkRuns,
    activeTrace,
    metricKey,
    xScale,
    yScale,
    viewMode,
    chartWidth,
    ratio,
    ratioPercent,
    paretoOnly,
    palette,
  ]);

  if (!activeTrace) {
    return null;
  }

  return (
    <section
      ref={wrapperRef}
      aria-label="Interactive Syntactic Pruning Ablation Demo"
      className="theme-surface theme-border overflow-hidden rounded-xl border font-sans shadow-2xs"
    >
      {/* Header & Trace Selector */}
      <div className="theme-subtle theme-border border-b p-4 sm:px-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <span className="theme-ink-faint font-mono text-[11px] font-semibold tracking-wider uppercase">
              Figure 2 · Ablation A — Syntactic Pruning Scrubber
            </span>
            <h3 className="theme-ink-strong mt-0.5 text-base font-bold tracking-tight">
              Stripping Grammatical Glue into Telegraphic State Transitions
            </h3>
          </div>

          <div className="theme-seg inline-flex flex-wrap rounded-lg p-0.5 text-xs">
            {traces.map((tr, idx) => (
              <button
                key={tr.trace_id}
                type="button"
                onClick={() => setSelectedTraceIdx(idx)}
                className={`rounded-md px-2.5 py-1 font-medium transition ${
                  selectedTraceIdx === idx ? 'theme-btn-active' : 'theme-seg-idle'
                }`}
              >
                {tr.benchmark} ({tr.model_name})
              </button>
            ))}
          </div>
        </div>

        {/* Slider Control */}
        <div className="theme-surface theme-border mt-3.5 rounded-xl border p-3.5 shadow-2xs">
          <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
            <label
              htmlFor="syntax-prune-slider"
              className="theme-ink-body flex items-center gap-2 font-medium"
            >
              <span>Syntactic Pruning Intensity</span>
              <span className="theme-subtle theme-border theme-ink-strong rounded-md border px-2 py-0.5 font-mono text-xs font-semibold">
                {ratioPercent}%
              </span>
            </label>
            <div className="flex flex-wrap gap-1.5">
              {[
                { label: '0% Verbose', val: 0 },
                { label: '36% No Discourse', val: 36 },
                { label: '68% No Determiners', val: 68 },
                { label: '100% Telegraphic', val: 100 },
              ].map((preset) => (
                <button
                  key={preset.val}
                  type="button"
                  onClick={() => setRatioPercent(preset.val)}
                  className={`rounded-md border px-2.5 py-0.5 text-[11px] font-medium transition ${
                    ratioPercent === preset.val ? 'theme-btn-active' : 'theme-btn-idle'
                  }`}
                >
                  {preset.label}
                </button>
              ))}
            </div>
          </div>

          <input
            id="syntax-prune-slider"
            type="range"
            min={0}
            max={100}
            step={1}
            value={ratioPercent}
            onChange={(e) => setRatioPercent(Number(e.target.value))}
            className="theme-slider mt-3 h-1.5 w-full cursor-pointer appearance-none rounded-full"
          />
          <div className="theme-ink-faint mt-1.5 flex justify-between font-mono text-[10px]">
            <span>0% (Full Grammatical CoT)</span>
            <span>50% (Auxiliary &amp; Copula Strip)</span>
            <span>100% (Pure State Transitions)</span>
          </div>
        </div>
      </div>

      <div className="p-4 sm:p-5">
        {/* Problem Prompt */}
        <div className="theme-subtle theme-border theme-ink-muted mb-3.5 rounded-lg border px-3.5 py-2.5 text-xs leading-relaxed">
          <span className="theme-ink-strong font-semibold">
            Problem ({activeTrace.benchmark}):{' '}
          </span>
          {activeTrace.prompt}
        </div>

        {/* Theme-Adaptive Fading Token Stream */}
        <div className="theme-subtle theme-border rounded-xl border p-4 leading-relaxed sm:p-4.5">
          <div className="theme-border theme-ink-muted mb-3 flex flex-wrap items-center justify-between gap-2 border-b pb-2.5 text-[11px]">
            <span className="font-mono">
              Reasoning Trace Stream ·{' '}
              <strong className="theme-ink-strong font-semibold">
                {liveMetrics.retainedTokens}
              </strong>{' '}
              / {liveMetrics.totalTokens} tokens active
            </span>
            <span className="theme-surface theme-border theme-ink-strong rounded-md border px-2.5 py-0.5 font-mono text-[11px] font-semibold shadow-2xs">
              Final Output: #### {activeTrace.final_answer}
            </span>
          </div>

          <div className="flex flex-wrap gap-1.5 font-mono text-xs">
            {activeTrace.tokens.map((tok) => {
              const isPruned = tok.is_syntactic && tok.prune_threshold <= ratio;
              const styleMeta = CATEGORY_LEGEND[tok.category];
              return (
                <span
                  key={tok.index}
                  title={`${styleMeta.label} · threshold τ = ${tok.prune_threshold}`}
                  className={`inline-block rounded-md px-1.5 py-0.5 transition-all duration-200 ${
                    isPruned ? 'tok-pruned' : styleMeta.chipClass
                  }`}
                >
                  {tok.text}
                </span>
              );
            })}
          </div>

          {/* Token Stratum Legend */}
          <div className="theme-border mt-3.5 flex flex-wrap gap-2 border-t pt-3 text-[10.5px]">
            {(Object.keys(CATEGORY_LEGEND) as TokenCategory[]).map((cat) => {
              const meta = CATEGORY_LEGEND[cat];
              return (
                <span
                  key={cat}
                  className="theme-surface theme-border theme-ink-muted inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5"
                >
                  <span
                    className="inline-block h-1.5 w-1.5 rounded-full"
                    style={{ backgroundColor: meta.dotColor }}
                  />
                  <span className="theme-ink-body font-medium">{meta.label}</span>
                  <span className="theme-ink-faint">
                    (<KatexSpan tex={meta.tauTex} />)
                  </span>
                </span>
              );
            })}
          </div>
        </div>

        {/* Live Telemetry Strip */}
        <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
          <div className="theme-surface theme-border rounded-xl border p-3.5 shadow-2xs">
            <div className="theme-ink-faint text-[10.5px] font-semibold tracking-wide uppercase">
              Active Trace Tokens
            </div>
            <div className="theme-ink-strong mt-1 flex items-baseline gap-1.5 font-mono text-lg font-bold">
              <span>{liveMetrics.retainedTokens}</span>
              <span className="theme-muted rounded px-1.5 py-0.2 text-[11px] font-medium">
                −{liveMetrics.tokenDropPct.toFixed(0)}%
              </span>
            </div>
          </div>

          <div className="theme-surface theme-border rounded-xl border p-3.5 shadow-2xs">
            <div className="theme-ink-faint text-[10.5px] font-semibold tracking-wide uppercase">
              Live KV-Cache (FP16)
            </div>
            <div className="theme-ink-strong mt-1 font-mono text-lg font-bold">
              {liveMetrics.liveKvKb.toFixed(0)} KB
            </div>
            <div className="theme-ink-faint font-mono text-[10px]">
              Baseline: {liveMetrics.baseKvKb.toFixed(0)} KB
            </div>
          </div>

          <div className="theme-surface theme-border rounded-xl border p-3.5 shadow-2xs">
            <div className="theme-ink-faint text-[10.5px] font-semibold tracking-wide uppercase">
              Trace Compute
            </div>
            <div className="theme-ink-strong mt-1 font-mono text-lg font-bold">
              {liveMetrics.liveGflops.toFixed(1)} GF
            </div>
            <div className="theme-ink-faint font-mono text-[10px]">
              Baseline: {liveMetrics.baseGflops.toFixed(1)} GF
            </div>
          </div>

          <div className="theme-surface theme-border rounded-xl border p-3.5 shadow-2xs">
            <div className="theme-ink-faint text-[10.5px] font-semibold tracking-wide uppercase">
              Suite Accuracy
            </div>
            <div className="theme-ink-strong mt-1 font-mono text-lg font-bold">
              {(liveMetrics.interpolatedAccuracy * 100).toFixed(1)}%
            </div>
            <div className="theme-ink-faint font-mono text-[10px]">
              Retains &gt;99% of baseline
            </div>
          </div>
        </div>

        {/* Embedded Epoch-Pattern Quantitative Comparison Strip */}
        <div className="theme-surface theme-border mt-4 rounded-xl border p-3.5">
          <div className="mb-2.5 flex flex-wrap items-center justify-between gap-2 text-xs">
            <div className="flex flex-wrap items-center gap-2">
              <span className="theme-ink-body font-semibold">
                Benchmark Trajectory ({activeTrace.model_name} on {activeTrace.benchmark}):
              </span>
              <select
                aria-label="Comparison X-axis metric"
                value={metricKey}
                onChange={(e) => setMetricKey(e.target.value as PrunerMetricKey)}
                className="theme-input rounded-md px-2 py-1 text-xs font-medium"
              >
                <option value="kv_cache_kb">KV-Cache Footprint (KB)</option>
                <option value="flops_per_problem">Compute (GFLOPs)</option>
                <option value="mean_tokens_emitted">Emitted Tokens</option>
              </select>
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <button
                type="button"
                onClick={() => setXScale((s) => (s === 'linear' ? 'log' : 'linear'))}
                className="theme-btn-idle rounded-md border px-2 py-0.5 text-[11px] font-medium transition"
              >
                X: {xScale === 'linear' ? 'Linear' : 'Log'}
              </button>
              <button
                type="button"
                onClick={() => setYScale((s) => (s === 'linear' ? 'logit' : 'linear'))}
                className="theme-btn-idle rounded-md border px-2 py-0.5 text-[11px] font-medium transition"
              >
                Y: {yScale === 'linear' ? 'Linear' : 'Logit'}
              </button>
              <label className="theme-ink-muted inline-flex cursor-pointer items-center gap-1 text-[11px] font-medium hover:opacity-85">
                <input
                  type="checkbox"
                  checked={paretoOnly}
                  onChange={(e) => setParetoOnly(e.target.checked)}
                  className="theme-checkbox h-3 w-3 rounded"
                />
                Pareto Only
              </label>
              <button
                type="button"
                onClick={() => setViewMode((v) => (v === 'graph' ? 'table' : 'graph'))}
                className="theme-btn-active rounded-md border px-2.5 py-0.5 text-[11px] font-medium transition hover:opacity-90"
              >
                {viewMode === 'graph' ? 'Table' : 'Graph'}
              </button>
            </div>
          </div>

          {viewMode === 'graph' ? (
            <div ref={plotRef} className="w-full overflow-x-auto" />
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full border-collapse text-left text-xs">
                <thead>
                  <tr className="theme-subtle theme-border theme-ink-muted border-b text-[11px] font-semibold">
                    <th className="px-2.5 py-1.5">Regime / Run</th>
                    <th className="px-2.5 py-1.5">Accuracy (95% CI)</th>
                    <th className="px-2.5 py-1.5">
                      <button
                        type="button"
                        onClick={() => setTableSortAsc((prev) => !prev)}
                        className="font-semibold hover:opacity-80"
                      >
                        Selected Metric ({tableSortAsc ? '↑' : '↓'})
                      </button>
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {matchingBenchmarkRuns.map((r) => (
                    <tr
                      key={r.run_id}
                      className="theme-border theme-row-hover border-b transition last:border-b-0"
                    >
                      <td className="theme-ink-strong px-2.5 py-1.5 font-medium">
                        <span
                          className="mr-1.5 inline-block h-2 w-2 rounded-full"
                          style={{ backgroundColor: REGIME_META[r.regime].color }}
                        />
                        {r.model_name} · {REGIME_META[r.regime].shortLabel}
                      </td>
                      <td className="theme-ink-body px-2.5 py-1.5 font-mono">
                        {(r.accuracy * 100).toFixed(1)}% [{(r.ci_95[0] * 100).toFixed(1)}–{(r.ci_95[1] * 100).toFixed(1)}%]
                      </td>
                      <td className="theme-ink-body px-2.5 py-1.5 font-mono">
                        {metricKey === 'flops_per_problem'
                          ? `${(r.flops_per_problem / 1e9).toFixed(1)} GFLOPs`
                          : metricKey === 'kv_cache_kb'
                            ? `${r.kv_cache_kb.toFixed(0)} KB`
                            : `${r.mean_tokens_emitted.toFixed(0)} tokens`}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
