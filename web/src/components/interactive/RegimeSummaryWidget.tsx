import { useEffect, useMemo, useRef, useState } from 'react';
import * as Plot from '@observablehq/plot';
import katex from 'katex';
import rawArtifact from '@/data/ablation_frontier.json';
import { useThemePalette } from '@/hooks/useThemePalette';
import {
  REGIME_META,
  normalizeAblationArtifact,
  type AblationRun,
  type Regime,
} from '@/types/ablation';

type ScaleSelection = 'ALL' | 0.6 | 1.5 | 3.0;
type ResourceBenchmark = 'GSM8K' | 'MATH';
type ResourceMetric = 'flops_per_problem' | 'mean_tokens_emitted' | 'kv_cache_kb';
type SortColumn = 'order' | 'gsm8k' | 'math' | 'tokens' | 'flops' | 'kv';

interface SummaryRow {
  id: string;
  order: number;
  regime: Regime;
  title: string;
  kTex: string | null;
  roleLabel: string;
  auditTex: string;
  auditSuffix: string;
  gsm06?: AblationRun;
  gsm15?: AblationRun;
  gsm30?: AblationRun;
  math06?: AblationRun;
  math15?: AblationRun;
  math30?: AblationRun;
  isPareto: boolean;
}

const ARTIFACT = normalizeAblationArtifact(rawArtifact);

function toLogit(p: number): number {
  const c = Math.min(Math.max(p, 1e-6), 1 - 1e-6);
  return Math.log(c / (1 - c));
}

function KatexSpan({ tex }: { tex: string }) {
  const html = useMemo(
    () => katex.renderToString(tex, { displayMode: false, output: 'html', throwOnError: false }),
    [tex]
  );
  return <span dangerouslySetInnerHTML={{ __html: html }} />;
}

function findRun(
  runs: AblationRun[],
  benchmark: 'GSM8K' | 'MATH',
  paramB: 0.6 | 1.5 | 3.0,
  regime: Regime,
  rankIndex = 0
): AblationRun | undefined {
  const matches = runs
    .filter(
      (r) =>
        r.benchmark === benchmark &&
        Math.abs(r.param_count_b - paramB) < 0.05 &&
        r.regime === regime
    )
    .sort((a, b) => a.latent_steps_k - b.latent_steps_k);
  return matches[rankIndex];
}

export default function RegimeSummaryWidget() {
  const palette = useThemePalette();
  const [scaleFilter, setScaleFilter] = useState<ScaleSelection>('ALL');
  const [resourceBench, setResourceBench] = useState<ResourceBenchmark>('GSM8K');
  const [yScale, setYScale] = useState<'linear' | 'logit'>('linear');
  const [xScale, setXScale] = useState<'linear' | 'log'>('linear');
  const [graphMetric, setGraphMetric] = useState<ResourceMetric>('flops_per_problem');
  const [showCi, setShowCi] = useState<boolean>(false);
  const [paretoOnly, setParetoOnly] = useState<boolean>(false);
  const [viewMode, setViewMode] = useState<'table' | 'graph'>('table');
  const [sortCol, setSortCol] = useState<SortColumn>('order');
  const [sortAsc, setSortAsc] = useState<boolean>(true);
  const [chartWidth, setChartWidth] = useState<number>(820);

  const wrapperRef = useRef<HTMLDivElement>(null);
  const plotRef = useRef<HTMLDivElement>(null);

  const has06B = useMemo(
    () => ARTIFACT.runs.some((r) => Math.abs(r.param_count_b - 0.6) < 0.05),
    []
  );

  useEffect(() => {
    const el = wrapperRef.current;
    if (!el) return;
    const measure = () => {
      setChartWidth(Math.max(320, Math.floor(el.getBoundingClientRect().width - 40)));
    };
    measure();
    const obs = new ResizeObserver(measure);
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  const baseRows = useMemo<SummaryRow[]>(() => {
    const runs = ARTIFACT.runs;
    const defs: Array<{
      id: string;
      order: number;
      regime: Regime;
      rank: number;
      title: string;
      kTex: string | null;
      roleLabel: string;
      auditTex: string;
      auditSuffix: string;
    }> = [
      {
        id: 'verbose_cot',
        order: 1,
        regime: 'verbose_cot',
        rank: 0,
        title: 'Verbose CoT',
        kTex: null,
        roleLabel: 'Baseline',
        auditTex: 'W_{\\text{unembed}}',
        auditSuffix: 'inline prose (high syntax tax)',
      },
      {
        id: 'telegraphic_cot',
        order: 2,
        regime: 'telegraphic_cot',
        rank: 0,
        title: 'Telegraphic CoT',
        kTex: null,
        roleLabel: 'Control A (Syntax)',
        auditTex: '\\mathcal{S}',
        auditSuffix: 'symbolic transitions only',
      },
      {
        id: 'compute_matched',
        order: 3,
        regime: 'compute_matched',
        rank: 0,
        title: 'Compute-Matched',
        kTex: null,
        roleLabel: 'Control B (Budget)',
        auditTex: '\\text{halt}',
        auditSuffix: 'truncated verbose trace',
      },
      {
        id: 'latent_loop_low',
        order: 4,
        regime: 'latent_loop',
        rank: 0,
        title: 'Latent Loop + Tee',
        kTex: 'K = 4 / 6',
        roleLabel: 'Proposed (M ≤ K)',
        auditTex: 'g_{\\phi}(\\operatorname{sg}[z_k])',
        auditSuffix: 'O(1) ring buffer',
      },
      {
        id: 'latent_loop_mid',
        order: 5,
        regime: 'latent_loop',
        rank: 1,
        title: 'Latent Loop + Tee',
        kTex: 'K = 8 / 12',
        roleLabel: 'Proposed (M ≤ K)',
        auditTex: 'g_{\\phi}(\\operatorname{sg}[z_k])',
        auditSuffix: 'O(1) ring buffer',
      },
      {
        id: 'latent_loop_high',
        order: 6,
        regime: 'latent_loop',
        rank: 2,
        title: 'Latent Loop + Tee',
        kTex: 'K = 16 / 20',
        roleLabel: 'Proposed (M ≤ K)',
        auditTex: 'g_{\\phi}(\\operatorname{sg}[z_k])',
        auditSuffix: 'O(1) ring buffer',
      },
    ];

    return defs.map((d) => {
      const gsm06 = findRun(runs, 'GSM8K', 0.6, d.regime, d.rank);
      const gsm15 = findRun(runs, 'GSM8K', 1.5, d.regime, d.rank);
      const gsm30 = findRun(runs, 'GSM8K', 3.0, d.regime, d.rank);
      const math06 = findRun(runs, 'MATH', 0.6, d.regime, d.rank);
      const math15 = findRun(runs, 'MATH', 1.5, d.regime, d.rank);
      const math30 = findRun(runs, 'MATH', 3.0, d.regime, d.rank);
      const isPareto = Boolean(
        gsm06?.is_pareto_frontier ||
          gsm15?.is_pareto_frontier ||
          gsm30?.is_pareto_frontier ||
          math06?.is_pareto_frontier ||
          math15?.is_pareto_frontier ||
          math30?.is_pareto_frontier
      );
      return {
        id: d.id,
        order: d.order,
        regime: d.regime,
        title: d.title,
        kTex: d.kTex,
        roleLabel: d.roleLabel,
        auditTex: d.auditTex,
        auditSuffix: d.auditSuffix,
        gsm06,
        gsm15,
        gsm30,
        math06,
        math15,
        math30,
        isPareto,
      };
    });
  }, []);

  const pickResourceRun = (row: SummaryRow): AblationRun | undefined => {
    const preferParam: 0.6 | 1.5 | 3.0 = scaleFilter === 'ALL' ? 3.0 : scaleFilter;
    if (resourceBench === 'GSM8K') {
      if (preferParam === 0.6) return row.gsm06;
      return preferParam === 1.5 ? row.gsm15 : row.gsm30;
    }
    if (preferParam === 0.6) return row.math06;
    return preferParam === 1.5 ? row.math15 : row.math30;
  };

  const visibleRows = useMemo<SummaryRow[]>(() => {
    const filtered = paretoOnly ? baseRows.filter((r) => r.isPareto) : [...baseRows];
    filtered.sort((a, b) => {
      const resA = pickResourceRun(a);
      const resB = pickResourceRun(b);
      const gsmA =
        scaleFilter === 0.6 ? a.gsm06 : scaleFilter === 1.5 ? a.gsm15 : a.gsm30;
      const gsmB =
        scaleFilter === 0.6 ? b.gsm06 : scaleFilter === 1.5 ? b.gsm15 : b.gsm30;
      const mathA =
        scaleFilter === 0.6 ? a.math06 : scaleFilter === 1.5 ? a.math15 : a.math30;
      const mathB =
        scaleFilter === 0.6 ? b.math06 : scaleFilter === 1.5 ? b.math15 : b.math30;

      let diff = 0;
      if (sortCol === 'order') diff = a.order - b.order;
      else if (sortCol === 'gsm8k') diff = (gsmA?.accuracy ?? 0) - (gsmB?.accuracy ?? 0);
      else if (sortCol === 'math') diff = (mathA?.accuracy ?? 0) - (mathB?.accuracy ?? 0);
      else if (sortCol === 'tokens')
        diff = (resA?.mean_tokens_emitted ?? 0) - (resB?.mean_tokens_emitted ?? 0);
      else if (sortCol === 'flops')
        diff = (resA?.flops_per_problem ?? 0) - (resB?.flops_per_problem ?? 0);
      else if (sortCol === 'kv')
        diff = (resA?.kv_cache_kb ?? 0) - (resB?.kv_cache_kb ?? 0);

      return sortAsc ? diff : -diff;
    });
    return filtered;
  }, [baseRows, paretoOnly, sortCol, sortAsc, scaleFilter, resourceBench]);

  useEffect(() => {
    const host = plotRef.current;
    if (!host || viewMode !== 'graph' || visibleRows.length === 0) return;

    const plotData = visibleRows
      .map((row) => {
        const r = pickResourceRun(row);
        if (!r) return null;
        const xVal =
          graphMetric === 'flops_per_problem'
            ? r.flops_per_problem / 1e9
            : graphMetric === 'kv_cache_kb'
              ? r.kv_cache_kb
              : r.mean_tokens_emitted;
        const yVal = yScale === 'linear' ? r.accuracy * 100 : r.accuracy_logit;
        const ciLow = yScale === 'linear' ? r.ci_95[0] * 100 : toLogit(r.ci_95[0]);
        const ciHigh = yScale === 'linear' ? r.ci_95[1] * 100 : toLogit(r.ci_95[1]);
        const label = row.kTex ? `${row.title} (${row.kTex.replace(/\s+/g, '')})` : row.title;
        return {
          label,
          xVal,
          yVal,
          ciLow,
          ciHigh,
          color: REGIME_META[row.regime].color,
          tip: `${r.model_name} · ${r.benchmark}\nAccuracy: ${(r.accuracy * 100).toFixed(1)}% [95% CI: ${(r.ci_95[0] * 100).toFixed(1)}%–${(r.ci_95[1] * 100).toFixed(1)}%]\nTokens: ${r.mean_tokens_emitted.toFixed(0)} | Compute: ${(r.flops_per_problem / 1e9).toFixed(1)} GFLOPs | KV: ${r.kv_cache_kb.toFixed(0)} KB`,
        };
      })
      .filter((item): item is NonNullable<typeof item> => item !== null);

    const xAxisLabel =
      graphMetric === 'flops_per_problem'
        ? `${resourceBench} Compute per Problem (GFLOPs)`
        : graphMetric === 'kv_cache_kb'
          ? `${resourceBench} Peak KV-Cache Footprint (KB)`
          : `${resourceBench} Mean Emitted Tokens`;

    const plot = Plot.plot({
      width: chartWidth,
      height: 250,
      marginTop: 20,
      marginRight: 30,
      marginBottom: 42,
      marginLeft: 56,
      style: {
        background: palette.plotBg,
        color: palette.plotFg,
        fontFamily: 'ui-sans-serif, system-ui, sans-serif',
        fontSize: '11.5px',
      },
      x: { type: xScale, label: xAxisLabel, grid: true, nice: true },
      y: {
        type: 'linear',
        label:
          yScale === 'linear'
            ? `${resourceBench} Accuracy (%)`
            : `${resourceBench} Logit Accuracy`,
        grid: true,
        nice: true,
      },
      marks: [
        Plot.frame({ stroke: palette.frameStroke }),
        Plot.ruleX(plotData, {
          x: 'xVal',
          y1: 'ciLow',
          y2: 'ciHigh',
          stroke: 'color',
          strokeOpacity: 0.5,
          strokeWidth: 1.5,
          marker: 'tick',
        }),
        Plot.dot(plotData, {
          x: 'xVal',
          y: 'yVal',
          r: 6.5,
          fill: 'color',
          fillOpacity: 0.9,
          stroke: palette.dotHalo,
          strokeWidth: 1.5,
        }),
        Plot.text(plotData, {
          x: 'xVal',
          y: 'yVal',
          text: 'label',
          dy: -12,
          fontSize: 10,
          fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
          fill: palette.labelFg,
        }),
        Plot.tip(
          plotData,
          Plot.pointer({
            x: 'xVal',
            y: 'yVal',
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
    visibleRows,
    viewMode,
    graphMetric,
    xScale,
    yScale,
    chartWidth,
    resourceBench,
    scaleFilter,
    palette,
  ]);

  const formatSingleScore = (run: AblationRun | undefined): string => {
    if (!run) return '—';
    if (yScale === 'logit') return run.accuracy_logit.toFixed(2);
    return `${(run.accuracy * 100).toFixed(1)}%`;
  };

  const renderScoreCell = (
    run06?: AblationRun,
    run15?: AblationRun,
    run30?: AblationRun
  ) => {
    if (scaleFilter === 0.6 || scaleFilter === 1.5 || scaleFilter === 3.0) {
      const target =
        scaleFilter === 0.6 ? run06 : scaleFilter === 1.5 ? run15 : run30;
      return (
        <div>
          <span>{formatSingleScore(target)}</span>
          {showCi && target && (
            <span className="theme-ink-faint ml-1 text-[10px]">
              [{(target.ci_95[0] * 100).toFixed(1)}, {(target.ci_95[1] * 100).toFixed(1)}]
            </span>
          )}
        </div>
      );
    }
    return (
      <div>
        {has06B && (
          <>
            <span>{formatSingleScore(run06)}</span>
            <span className="theme-ink-faint mx-1">/</span>
          </>
        )}
        <span>{formatSingleScore(run15)}</span>
        <span className="theme-ink-faint mx-1">/</span>
        <span>{formatSingleScore(run30)}</span>
        {showCi && run30 && (
          <div className="theme-ink-faint text-[10px]">
            3B CI [{(run30.ci_95[0] * 100).toFixed(1)}–{(run30.ci_95[1] * 100).toFixed(1)}%]
          </div>
        )}
      </div>
    );
  };

  const toggleSort = (col: SortColumn) => {
    if (sortCol === col) {
      setSortAsc((prev) => !prev);
    } else {
      setSortCol(col);
      setSortAsc(col === 'order' || col === 'tokens' || col === 'flops' || col === 'kv');
    }
  };

  const activeScaleLabel =
    scaleFilter === 0.6 ? '0.6B' : scaleFilter === 1.5 ? '1.5B' : '3B';
  const multiScaleHeader = has06B ? '(0.6B / 1.5B / 3B)' : '(1.5B / 3B)';

  const scaleChoices: Array<[ScaleSelection, string]> = has06B
    ? [
        ['ALL', 'All Scales'],
        [0.6, '0.6B'],
        [1.5, '1.5B'],
        [3.0, '3B'],
      ]
    : [
        ['ALL', '1.5B / 3B'],
        [1.5, '1.5B'],
        [3.0, '3B'],
      ];

  return (
    <section
      ref={wrapperRef}
      aria-label="Cross-Regime Trade-offs and Deployment Summary Widget"
      className="theme-surface theme-border w-full overflow-hidden rounded-xl border font-sans shadow-2xs"
    >
      {/* Widget Control Header */}
      <div className="theme-subtle theme-border border-b p-4 sm:px-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <span className="theme-ink-faint font-mono text-[11px] font-semibold tracking-wider uppercase">
              Table 1 · Cross-Regime Trade-off Explorer
            </span>
            <h3 className="theme-ink-strong mt-0.5 text-base font-bold tracking-tight">
              Operational Hierarchy Across Inference Regimes (GSM8K &amp; MATH)
            </h3>
          </div>

          <div className="flex flex-wrap items-center gap-2 text-xs">
            {/* Scale Selector */}
            <div className="theme-seg inline-flex rounded-lg p-0.5">
              {scaleChoices.map(([val, label]) => (
                <button
                  key={String(val)}
                  type="button"
                  onClick={() => setScaleFilter(val)}
                  className={`rounded-md px-2.5 py-1 font-medium transition ${
                    scaleFilter === val ? 'theme-btn-active' : 'theme-seg-idle'
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>

            {/* Telemetry Suite Selector */}
            <div className="theme-seg inline-flex rounded-lg p-0.5">
              {(['GSM8K', 'MATH'] as const).map((b) => (
                <button
                  key={b}
                  type="button"
                  onClick={() => setResourceBench(b)}
                  className={`rounded-md px-2.5 py-1 font-medium transition ${
                    resourceBench === b ? 'theme-btn-active' : 'theme-seg-idle'
                  }`}
                >
                  {b} Cost
                </button>
              ))}
            </div>

            {/* View Toggle */}
            <div className="theme-seg inline-flex rounded-lg p-0.5">
              <button
                type="button"
                onClick={() => setViewMode('table')}
                className={`rounded-md px-2.5 py-1 font-medium transition ${
                  viewMode === 'table' ? 'theme-btn-active' : 'theme-seg-idle'
                }`}
              >
                Table
              </button>
              <button
                type="button"
                onClick={() => setViewMode('graph')}
                className={`rounded-md px-2.5 py-1 font-medium transition ${
                  viewMode === 'graph' ? 'theme-btn-active' : 'theme-seg-idle'
                }`}
              >
                Graph
              </button>
            </div>
          </div>
        </div>

        {/* Secondary Compact Filter Strip */}
        <div className="theme-border mt-3 flex flex-wrap items-center justify-between gap-2 border-t pt-2.5 text-xs">
          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={() => setYScale((s) => (s === 'linear' ? 'logit' : 'linear'))}
              className="theme-btn-idle rounded-md border px-2.5 py-0.5 text-[11px] font-medium transition"
            >
              Accuracy: {yScale === 'linear' ? 'Linear (%)' : 'Logit ln(p/(1−p))'}
            </button>

            {viewMode === 'graph' && (
              <>
                <select
                  aria-label="Graph X-axis metric"
                  value={graphMetric}
                  onChange={(e) => setGraphMetric(e.target.value as ResourceMetric)}
                  className="theme-input rounded-md px-2 py-0.5 text-[11px] font-medium"
                >
                  <option value="flops_per_problem">X: Compute (GFLOPs)</option>
                  <option value="mean_tokens_emitted">X: Emitted Tokens</option>
                  <option value="kv_cache_kb">X: KV-Cache (KB)</option>
                </select>
                <button
                  type="button"
                  onClick={() => setXScale((s) => (s === 'linear' ? 'log' : 'linear'))}
                  className="theme-btn-idle rounded-md border px-2 py-0.5 text-[11px] font-medium transition"
                >
                  X Scale: {xScale === 'linear' ? 'Linear' : 'Log'}
                </button>
              </>
            )}
          </div>

          <div className="theme-ink-muted flex flex-wrap items-center gap-4 text-[11px]">
            <label className="inline-flex cursor-pointer items-center gap-1.5 font-medium hover:opacity-85">
              <input
                type="checkbox"
                checked={showCi}
                onChange={(e) => setShowCi(e.target.checked)}
                className="theme-checkbox h-3.5 w-3.5 rounded"
              />
              95% CI Bounds
            </label>
            <label className="inline-flex cursor-pointer items-center gap-1.5 font-medium hover:opacity-85">
              <input
                type="checkbox"
                checked={paretoOnly}
                onChange={(e) => setParetoOnly(e.target.checked)}
                className="theme-checkbox h-3.5 w-3.5 rounded"
              />
              Pareto Frontier Only
            </label>
          </div>
        </div>
      </div>

      {/* Spacious Full-Width Table or Graph Body */}
      {viewMode === 'table' ? (
        <div className="w-full overflow-x-auto">
          <table className="w-full border-collapse text-left text-xs">
            <thead>
              <tr className="theme-subtle theme-border theme-ink-muted border-b text-[11px] font-semibold tracking-wide uppercase">
                <th className="px-4 py-3 whitespace-nowrap">
                  <button
                    type="button"
                    onClick={() => toggleSort('order')}
                    className="inline-flex items-center gap-1 font-semibold hover:opacity-80"
                  >
                    Regime {sortCol === 'order' && (sortAsc ? '↑' : '↓')}
                  </button>
                </th>
                <th className="px-3 py-3 whitespace-nowrap">Role</th>
                <th className="px-3 py-3 whitespace-nowrap">
                  <button
                    type="button"
                    onClick={() => toggleSort('gsm8k')}
                    className="inline-flex items-center gap-1 font-semibold hover:opacity-80"
                  >
                    GSM8K {scaleFilter === 'ALL' ? multiScaleHeader : `(${scaleFilter}B)`}{' '}
                    {sortCol === 'gsm8k' && (sortAsc ? '↑' : '↓')}
                  </button>
                </th>
                <th className="px-3 py-3 whitespace-nowrap">
                  <button
                    type="button"
                    onClick={() => toggleSort('math')}
                    className="inline-flex items-center gap-1 font-semibold hover:opacity-80"
                  >
                    MATH {scaleFilter === 'ALL' ? multiScaleHeader : `(${scaleFilter}B)`}{' '}
                    {sortCol === 'math' && (sortAsc ? '↑' : '↓')}
                  </button>
                </th>
                <th className="px-3 py-3 text-right whitespace-nowrap">
                  <button
                    type="button"
                    onClick={() => toggleSort('tokens')}
                    className="inline-flex items-center gap-1 font-semibold hover:opacity-80"
                  >
                    Tokens ({resourceBench} {activeScaleLabel}){' '}
                    {sortCol === 'tokens' && (sortAsc ? '↑' : '↓')}
                  </button>
                </th>
                <th className="px-3 py-3 text-right whitespace-nowrap">
                  <button
                    type="button"
                    onClick={() => toggleSort('flops')}
                    className="inline-flex items-center gap-1 font-semibold hover:opacity-80"
                  >
                    Compute (GFLOPs) {sortCol === 'flops' && (sortAsc ? '↑' : '↓')}
                  </button>
                </th>
                <th className="px-3 py-3 text-right whitespace-nowrap">
                  <button
                    type="button"
                    onClick={() => toggleSort('kv')}
                    className="inline-flex items-center gap-1 font-semibold hover:opacity-80"
                  >
                    KV-Cache {sortCol === 'kv' && (sortAsc ? '↑' : '↓')}
                  </button>
                </th>
                <th className="px-4 py-3 whitespace-nowrap">Auditability</th>
              </tr>
            </thead>
            <tbody>
              {visibleRows.map((row) => {
                const meta = REGIME_META[row.regime];
                const resRun = pickResourceRun(row);
                const isProposed = row.regime === 'latent_loop';
                return (
                  <tr
                    key={row.id}
                    className={`theme-border theme-row-hover border-b transition last:border-b-0 ${
                      isProposed ? 'theme-proposed-row' : ''
                    }`}
                  >
                    <td className="px-4 py-3 whitespace-nowrap">
                      <div className="flex items-center gap-2">
                        <span
                          className="h-2.5 w-2.5 shrink-0 rounded-full"
                          style={{ backgroundColor: meta.color }}
                        />
                        <span
                          className={`text-[12.5px] ${
                            isProposed
                              ? 'theme-ink-strong font-bold'
                              : 'theme-ink-body font-medium'
                          }`}
                        >
                          {row.title}
                        </span>
                        {row.kTex && (
                          <span className="theme-subtle theme-border theme-ink-muted rounded border px-1.5 py-0.2 text-[10.5px]">
                            <KatexSpan tex={row.kTex} />
                          </span>
                        )}
                      </div>
                    </td>

                    <td className="px-3 py-3 whitespace-nowrap">
                      <span
                        className={`inline-flex rounded-md border px-2 py-0.5 text-[11px] font-medium ${
                          isProposed
                            ? 'theme-surface theme-border-strong theme-ink-strong font-semibold shadow-2xs'
                            : 'theme-subtle theme-border theme-ink-muted'
                        }`}
                      >
                        {row.roleLabel}
                      </span>
                    </td>

                    <td
                      className={`px-3 py-3 font-mono whitespace-nowrap ${
                        isProposed ? 'theme-ink-strong font-semibold' : 'theme-ink-body'
                      }`}
                    >
                      {renderScoreCell(row.gsm06, row.gsm15, row.gsm30)}
                    </td>

                    <td
                      className={`px-3 py-3 font-mono whitespace-nowrap ${
                        isProposed ? 'theme-ink-strong font-semibold' : 'theme-ink-body'
                      }`}
                    >
                      {renderScoreCell(row.math06, row.math15, row.math30)}
                    </td>

                    <td className="theme-ink-body px-3 py-3 text-right font-mono whitespace-nowrap">
                      {resRun ? resRun.mean_tokens_emitted.toFixed(0) : '—'}
                    </td>

                    <td className="theme-ink-body px-3 py-3 text-right font-mono whitespace-nowrap">
                      {resRun ? `${(resRun.flops_per_problem / 1e9).toFixed(1)}` : '—'}
                    </td>

                    <td className="theme-ink-body px-3 py-3 text-right font-mono whitespace-nowrap">
                      {resRun ? `${resRun.kv_cache_kb.toLocaleString()} KB` : '—'}
                    </td>

                    <td className="theme-ink-muted px-4 py-3">
                      <span className="inline-flex items-center gap-1.5 whitespace-nowrap">
                        <span className="theme-subtle theme-border theme-ink-strong rounded border px-1.5 py-0.2 text-[11px]">
                          <KatexSpan tex={row.auditTex} />
                        </span>
                        <span className="theme-ink-muted text-[11.5px]">{row.auditSuffix}</span>
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="p-4 sm:p-5">
          <div ref={plotRef} className="w-full overflow-x-auto" />
        </div>
      )}
    </section>
  );
}
