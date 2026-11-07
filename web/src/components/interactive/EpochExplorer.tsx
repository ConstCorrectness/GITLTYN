import { useEffect, useMemo, useRef, useState } from 'react';
import * as Plot from '@observablehq/plot';
import katex from 'katex';
import rawArtifact from '@/data/ablation_frontier.json';
import { useThemePalette } from '@/hooks/useThemePalette';
import {
  REGIME_META,
  REGIME_ORDER,
  normalizeAblationArtifact,
  type AblationRun,
  type Regime,
} from '@/types/ablation';

export type XMetricKey = 'flops_per_problem' | 'mean_tokens_emitted' | 'kv_cache_kb';
export type XScaleMode = 'log' | 'linear';
export type YScaleMode = 'linear' | 'logit';
export type ViewMode = 'graph' | 'table';
export type BenchmarkFilter = 'GSM8K' | 'MATH' | 'ALL';
export type ParamScaleFilter = 'ALL' | 0.6 | 1.5 | 3.0;
export type SortField =
  | 'model_name'
  | 'regime'
  | 'benchmark'
  | 'accuracy'
  | 'accuracy_logit'
  | 'mean_tokens_emitted'
  | 'flops_per_problem'
  | 'kv_cache_kb';

export interface EpochExplorerProps {
  defaultBenchmark?: BenchmarkFilter;
  initialXMetric?: XMetricKey;
}

interface PlottedRunPoint {
  run: AblationRun;
  xVal: number;
  yVal: number;
  ciLow: number;
  ciHigh: number;
  color: string;
  regimeLabel: string;
  radius: number;
  outerRadius: number;
  symbolType: 'circle' | 'diamond';
  pointLabel: string;
  tooltipText: string;
}

const ARTIFACT = normalizeAblationArtifact(rawArtifact);

const X_METRIC_META: Record<
  XMetricKey,
  { label: string; axisLabel: string; unit: string; toDisplay: (run: AblationRun) => number }
> = {
  flops_per_problem: {
    label: 'Compute (GFLOPs / Problem)',
    axisLabel: 'Inference Compute per Problem (GFLOPs)',
    unit: 'GFLOPs',
    toDisplay: (r) => r.flops_per_problem / 1e9,
  },
  mean_tokens_emitted: {
    label: 'Mean Tokens Emitted',
    axisLabel: 'Mean Discrete Tokens Emitted per Problem',
    unit: 'tokens',
    toDisplay: (r) => r.mean_tokens_emitted,
  },
  kv_cache_kb: {
    label: 'KV-Cache Footprint (KB)',
    axisLabel: 'Peak Sequence KV-Cache Footprint (KB, FP16)',
    unit: 'KB',
    toDisplay: (r) => r.kv_cache_kb,
  },
};

function toLogit(prob: number): number {
  const clamped = Math.min(Math.max(prob, 1e-6), 1 - 1e-6);
  return Math.log(clamped / (1 - clamped));
}

function KatexSpan({ tex }: { tex: string }) {
  const html = useMemo(
    () => katex.renderToString(tex, { displayMode: false, output: 'html', throwOnError: false }),
    [tex]
  );
  return <span dangerouslySetInnerHTML={{ __html: html }} />;
}

function triggerBlobDownload(filename: string, content: string, mimeType: string): void {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  URL.revokeObjectURL(url);
}

export default function EpochExplorer({
  defaultBenchmark = 'GSM8K',
  initialXMetric = 'flops_per_problem',
}: EpochExplorerProps) {
  const palette = useThemePalette();
  const [benchmark, setBenchmark] = useState<BenchmarkFilter>(defaultBenchmark);
  const [paramFilter, setParamFilter] = useState<ParamScaleFilter>('ALL');
  const [xMetric, setXMetric] = useState<XMetricKey>(initialXMetric);
  const [xScale, setXScale] = useState<XScaleMode>('log');
  const [yScale, setYScale] = useState<YScaleMode>('linear');
  const [paretoOnly, setParetoOnly] = useState<boolean>(false);
  const [showCiBars, setShowCiBars] = useState<boolean>(true);
  const [activeRegimes, setActiveRegimes] = useState<Record<Regime, boolean>>({
    verbose_cot: true,
    telegraphic_cot: true,
    compute_matched: true,
    latent_loop: true,
  });
  const [viewMode, setViewMode] = useState<ViewMode>('graph');
  const [sortField, setSortField] = useState<SortField>('flops_per_problem');
  const [sortAsc, setSortAsc] = useState<boolean>(true);
  const [containerWidth, setContainerWidth] = useState<number>(760);

  const plotContainerRef = useRef<HTMLDivElement>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);

  const has06B = useMemo(
    () => ARTIFACT.runs.some((r) => Math.abs(r.param_count_b - 0.6) < 0.05),
    []
  );
  const scaleOptions = useMemo<ParamScaleFilter[]>(
    () => (has06B ? ['ALL', 0.6, 1.5, 3.0] : ['ALL', 1.5, 3.0]),
    [has06B]
  );

  useEffect(() => {
    const el = wrapperRef.current;
    if (!el) return;
    const updateWidth = () => {
      const next = Math.max(320, Math.floor(el.getBoundingClientRect().width - 36));
      setContainerWidth(next);
    };
    updateWidth();
    const observer = new ResizeObserver(updateWidth);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const filteredRuns = useMemo<AblationRun[]>(() => {
    return ARTIFACT.runs.filter((run) => {
      if (benchmark !== 'ALL' && run.benchmark !== benchmark) return false;
      if (paramFilter !== 'ALL' && Math.abs(run.param_count_b - paramFilter) > 0.05) return false;
      if (!activeRegimes[run.regime]) return false;
      if (paretoOnly && !run.is_pareto_frontier) return false;
      return true;
    });
  }, [benchmark, paramFilter, activeRegimes, paretoOnly]);

  const sortedTableRuns = useMemo<AblationRun[]>(() => {
    const copy = [...filteredRuns];
    copy.sort((a, b) => {
      const aVal = a[sortField];
      const bVal = b[sortField];
      if (typeof aVal === 'string' && typeof bVal === 'string') {
        return sortAsc ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
      }
      const diff = Number(aVal) - Number(bVal);
      return sortAsc ? diff : -diff;
    });
    return copy;
  }, [filteredRuns, sortField, sortAsc]);

  const plottedPoints = useMemo<PlottedRunPoint[]>(() => {
    const metricSpec = X_METRIC_META[xMetric];
    return filteredRuns.map((run) => {
      const xVal = metricSpec.toDisplay(run);
      const yVal = yScale === 'linear' ? run.accuracy * 100 : run.accuracy_logit;
      const ciLow = yScale === 'linear' ? run.ci_95[0] * 100 : toLogit(run.ci_95[0]);
      const ciHigh = yScale === 'linear' ? run.ci_95[1] * 100 : toLogit(run.ci_95[1]);
      const meta = REGIME_META[run.regime];
      const radius = run.param_count_b >= 3.0 ? 7.25 : run.param_count_b >= 1.5 ? 5.75 : 4.6;
      const tooltipText = [
        `${run.model_name} · ${run.benchmark}`,
        `Regime: ${meta.label}`,
        `Accuracy: ${(run.accuracy * 100).toFixed(1)}% (95% CI: ${(run.ci_95[0] * 100).toFixed(1)}%–${(run.ci_95[1] * 100).toFixed(1)}%)`,
        `Logit Accuracy: ${run.accuracy_logit.toFixed(3)}`,
        `Emitted Tokens: ${run.mean_tokens_emitted.toFixed(0)} | Latent Steps K: ${run.latent_steps_k}`,
        `Compute: ${(run.flops_per_problem / 1e9).toFixed(1)} GFLOPs | KV-Cache: ${run.kv_cache_kb.toFixed(0)} KB`,
        run.is_pareto_frontier ? '★ Pareto Frontier Operating Point' : 'Dominated Operating Point',
      ].join('\n');

      return {
        run,
        xVal,
        yVal,
        ciLow,
        ciHigh,
        color: meta.color,
        regimeLabel: meta.shortLabel,
        radius,
        outerRadius: radius + 3.5,
        symbolType: run.benchmark === 'MATH' ? 'diamond' : 'circle',
        pointLabel:
          run.latent_steps_k > 0
            ? `${run.param_count_b}B · K=${run.latent_steps_k}`
            : `${run.param_count_b}B`,
        tooltipText,
      };
    });
  }, [filteredRuns, xMetric, yScale]);

  useEffect(() => {
    const host = plotContainerRef.current;
    if (!host || viewMode !== 'graph' || plottedPoints.length === 0) return;

    const metricSpec = X_METRIC_META[xMetric];
    const yAxisLabel =
      yScale === 'linear'
        ? 'Benchmark Accuracy (%)'
        : 'Logit Accuracy (log-odds)';

    const paretoGroups: PlottedRunPoint[][] = [];
    if (benchmark === 'ALL') {
      for (const bName of ['GSM8K', 'MATH']) {
        const pts = plottedPoints
          .filter((p) => p.run.is_pareto_frontier && p.run.benchmark === bName)
          .sort((a, b) => a.xVal - b.xVal);
        if (pts.length >= 2) paretoGroups.push(pts);
      }
    } else {
      const pts = plottedPoints
        .filter((p) => p.run.is_pareto_frontier)
        .sort((a, b) => a.xVal - b.xVal);
      if (pts.length >= 2) paretoGroups.push(pts);
    }

    const marks: Plot.Markish[] = [
      Plot.frame({ stroke: palette.frameStroke }),
    ];

    for (const frontierPts of paretoGroups) {
      marks.push(
        Plot.line(frontierPts, {
          x: 'xVal',
          y: 'yVal',
          stroke: REGIME_META.latent_loop.color,
          strokeWidth: 1.75,
          strokeDasharray: '5,4',
          strokeOpacity: 0.78,
          curve: 'monotone-x',
        })
      );
    }

    if (showCiBars) {
      marks.push(
        Plot.ruleX(plottedPoints, {
          x: 'xVal',
          y1: 'ciLow',
          y2: 'ciHigh',
          stroke: 'color',
          strokeOpacity: 0.45,
          strokeWidth: 1.4,
          marker: 'tick',
        })
      );
    }

    const paretoHighlight = plottedPoints.filter((p) => p.run.is_pareto_frontier);
    if (paretoHighlight.length > 0) {
      marks.push(
        Plot.dot(paretoHighlight, {
          x: 'xVal',
          y: 'yVal',
          r: 'outerRadius',
          stroke: 'color',
          strokeOpacity: 0.35,
          strokeWidth: 1.25,
          fill: 'none',
        })
      );
    }

    marks.push(
      Plot.dot(plottedPoints, {
        x: 'xVal',
        y: 'yVal',
        r: 'radius',
        fill: 'color',
        fillOpacity: 0.9,
        stroke: palette.dotHalo,
        strokeWidth: 1.5,
        symbol: 'symbolType',
      }),
      Plot.text(plottedPoints, {
        x: 'xVal',
        y: 'yVal',
        text: 'pointLabel',
        dy: -12,
        fontSize: 10,
        fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
        fill: palette.labelFg,
      }),
      Plot.tip(
        plottedPoints,
        Plot.pointer({
          x: 'xVal',
          y: 'yVal',
          title: 'tooltipText',
          fontSize: 11,
          fill: palette.tipBg,
          stroke: palette.tipStroke,
        })
      )
    );

    const plot = Plot.plot({
      width: containerWidth,
      height: 405,
      marginTop: 24,
      marginRight: 28,
      marginBottom: 46,
      marginLeft: 56,
      style: {
        background: palette.plotBg,
        color: palette.plotFg,
        fontFamily: 'ui-sans-serif, system-ui, -apple-system, sans-serif',
        fontSize: '11.5px',
        overflow: 'visible',
      },
      x: {
        type: xScale,
        label: metricSpec.axisLabel,
        grid: true,
        nice: true,
      },
      y: {
        type: 'linear',
        label: yAxisLabel,
        grid: true,
        nice: true,
      },
      r: { type: 'identity' },
      symbol: { type: 'identity' },
      marks,
    });

    host.append(plot);
    return () => plot.remove();
  }, [
    plottedPoints,
    xMetric,
    xScale,
    yScale,
    showCiBars,
    viewMode,
    containerWidth,
    benchmark,
    palette,
  ]);

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortAsc((prev) => !prev);
    } else {
      setSortField(field);
      setSortAsc(field === 'flops_per_problem' || field === 'mean_tokens_emitted');
    }
  };

  const handleExportCsv = () => {
    const headers = [
      'run_id',
      'model_name',
      'param_count_b',
      'regime',
      'benchmark',
      'accuracy',
      'accuracy_logit',
      'ci_95_low',
      'ci_95_high',
      'mean_tokens_emitted',
      'latent_steps_k',
      'flops_per_problem',
      'kv_cache_kb',
      'is_pareto_frontier',
    ];
    const rows = sortedTableRuns.map((r) =>
      [
        r.run_id,
        `"${r.model_name}"`,
        r.param_count_b,
        r.regime,
        r.benchmark,
        r.accuracy,
        r.accuracy_logit,
        r.ci_95[0],
        r.ci_95[1],
        r.mean_tokens_emitted,
        r.latent_steps_k,
        r.flops_per_problem,
        r.kv_cache_kb,
        r.is_pareto_frontier,
      ].join(',')
    );
    triggerBlobDownload(
      `gitltyn-${benchmark.toLowerCase()}-frontier.csv`,
      [headers.join(','), ...rows].join('\n'),
      'text/csv;charset=utf-8'
    );
  };

  const handleExportSvg = () => {
    const svgEl = plotContainerRef.current?.querySelector('svg');
    if (!svgEl) return;
    const cloned = svgEl.cloneNode(true) as SVGSVGElement;
    cloned.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
    const serialized = new XMLSerializer().serializeToString(cloned);
    triggerBlobDownload(
      `gitltyn-${benchmark.toLowerCase()}-${xMetric}.svg`,
      serialized,
      'image/svg+xml;charset=utf-8'
    );
  };

  const toggleRegime = (regime: Regime) => {
    setActiveRegimes((prev) => ({ ...prev, [regime]: !prev[regime] }));
  };

  return (
    <section
      ref={wrapperRef}
      aria-label="Epoch-style Reasoning Compute Frontier Explorer"
      className="theme-surface theme-border overflow-hidden rounded-xl border font-sans shadow-2xs"
    >
      {/* Top Epoch AI Control Header */}
      <div className="theme-subtle theme-border border-b p-4 sm:px-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <div className="flex items-center gap-2">
              <span className="theme-ink-faint font-mono text-[11px] font-semibold tracking-wider uppercase">
                Figure 1 · Interactive Frontier Explorer
              </span>
              <span className="theme-surface theme-border theme-ink-muted rounded-md border px-1.5 py-0.5 font-mono text-[10px]">
                <KatexSpan tex={`n = ${filteredRuns.length}`} /> runs
              </span>
            </div>
            <h3 className="theme-ink-strong mt-0.5 text-base font-bold tracking-tight">
              Accuracy vs. Inference Resource Cost Across Reasoning Regimes
            </h3>
          </div>

          {/* View Switcher + Export Actions */}
          <div className="flex flex-wrap items-center gap-2">
            <div className="theme-seg inline-flex rounded-lg p-0.5 text-xs font-medium">
              <button
                type="button"
                onClick={() => setViewMode('graph')}
                className={`rounded-md px-2.5 py-1 transition ${
                  viewMode === 'graph' ? 'theme-btn-active' : 'theme-seg-idle'
                }`}
              >
                Graph
              </button>
              <button
                type="button"
                onClick={() => setViewMode('table')}
                className={`rounded-md px-2.5 py-1 transition ${
                  viewMode === 'table' ? 'theme-btn-active' : 'theme-seg-idle'
                }`}
              >
                Table
              </button>
            </div>

            <button
              type="button"
              onClick={handleExportCsv}
              className="theme-btn-idle rounded-lg border px-2.5 py-1 text-xs font-medium transition"
            >
              Export CSV
            </button>
            {viewMode === 'graph' && (
              <button
                type="button"
                onClick={handleExportSvg}
                className="theme-btn-idle rounded-lg border px-2.5 py-1 text-xs font-medium transition"
              >
                Export SVG
              </button>
            )}
          </div>
        </div>

        {/* Control Bar Row 2: Benchmark, Axis Metric, Scales, Pareto & CI */}
        <div className="theme-border mt-3.5 grid grid-cols-1 gap-3 border-t pt-3.5 sm:grid-cols-2 lg:grid-cols-5">
          {/* Benchmark Selector */}
          <div>
            <label className="theme-ink-faint block text-[10.5px] font-semibold tracking-wide uppercase">
              Benchmark Suite
            </label>
            <div className="theme-seg mt-1 inline-flex w-full rounded-lg p-0.5 text-xs">
              {(['GSM8K', 'MATH', 'ALL'] as const).map((bOpt) => (
                <button
                  key={bOpt}
                  type="button"
                  onClick={() => setBenchmark(bOpt)}
                  className={`flex-1 rounded-md py-1 font-medium transition ${
                    benchmark === bOpt ? 'theme-btn-active' : 'theme-seg-idle'
                  }`}
                >
                  {bOpt === 'ALL' ? 'Both' : bOpt}
                </button>
              ))}
            </div>
          </div>

          {/* X-Axis Metric Selector */}
          <div>
            <label
              htmlFor="epoch-x-metric"
              className="theme-ink-faint block text-[10.5px] font-semibold tracking-wide uppercase"
            >
              X-Axis Resource Metric
            </label>
            <select
              id="epoch-x-metric"
              value={xMetric}
              onChange={(e) => setXMetric(e.target.value as XMetricKey)}
              className="theme-input mt-1 w-full rounded-lg px-2.5 py-1.5 text-xs font-medium"
            >
              <option value="flops_per_problem">Compute (GFLOPs / Problem)</option>
              <option value="mean_tokens_emitted">Emitted Tokens / Problem</option>
              <option value="kv_cache_kb">KV-Cache Footprint (KB)</option>
            </select>
          </div>

          {/* X Scale Toggle */}
          <div>
            <span className="theme-ink-faint block text-[10.5px] font-semibold tracking-wide uppercase">
              X-Axis Scale
            </span>
            <div className="theme-seg mt-1 inline-flex w-full rounded-lg p-0.5 text-xs">
              {(['log', 'linear'] as const).map((mode) => (
                <button
                  key={mode}
                  type="button"
                  onClick={() => setXScale(mode)}
                  className={`flex-1 rounded-md py-1 font-medium capitalize transition ${
                    xScale === mode ? 'theme-btn-active' : 'theme-seg-idle'
                  }`}
                >
                  {mode}
                </button>
              ))}
            </div>
          </div>

          {/* Y Scale Toggle */}
          <div>
            <span className="theme-ink-faint block text-[10.5px] font-semibold tracking-wide uppercase">
              Y-Axis Scale
            </span>
            <div className="theme-seg mt-1 inline-flex w-full rounded-lg p-0.5 text-xs">
              {(['linear', 'logit'] as const).map((mode) => (
                <button
                  key={mode}
                  type="button"
                  onClick={() => setYScale(mode)}
                  className={`flex-1 rounded-md py-1 font-medium transition ${
                    yScale === mode ? 'theme-btn-active' : 'theme-seg-idle'
                  }`}
                >
                  {mode === 'linear' ? 'Linear (%)' : 'Logit'}
                </button>
              ))}
            </div>
          </div>

          {/* SLM Scale Filter (0.6B / 1.5B / 3B) */}
          <div>
            <span className="theme-ink-faint block text-[10.5px] font-semibold tracking-wide uppercase">
              SLM Scale
            </span>
            <div className="theme-seg mt-1 inline-flex w-full rounded-lg p-0.5 text-xs">
              {scaleOptions.map((scaleOpt) => (
                <button
                  key={String(scaleOpt)}
                  type="button"
                  onClick={() => setParamFilter(scaleOpt)}
                  className={`flex-1 rounded-md px-1.5 py-1 font-medium transition ${
                    paramFilter === scaleOpt ? 'theme-btn-active' : 'theme-seg-idle'
                  }`}
                >
                  {scaleOpt === 'ALL' ? 'All' : `${scaleOpt}B`}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Interactive Regime Filter Pills + Checkboxes */}
        <div className="mt-3 flex flex-wrap items-center justify-between gap-3 pt-1 text-xs">
          <div className="flex flex-wrap items-center gap-1.5">
            {REGIME_ORDER.map((regime) => {
              const meta = REGIME_META[regime];
              const enabled = activeRegimes[regime];
              return (
                <button
                  key={regime}
                  type="button"
                  onClick={() => toggleRegime(regime)}
                  className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-medium transition ${
                    enabled
                      ? 'theme-surface theme-border-strong theme-ink-strong shadow-2xs'
                      : 'theme-muted theme-border opacity-55'
                  }`}
                >
                  <span
                    className="inline-block h-2 w-2 rounded-full"
                    style={{ backgroundColor: meta.color }}
                  />
                  {meta.shortLabel}
                </button>
              );
            })}
          </div>

          <div className="theme-ink-muted flex flex-wrap items-center gap-4 text-xs">
            <label className="inline-flex cursor-pointer items-center gap-1.5 font-medium hover:opacity-85">
              <input
                type="checkbox"
                checked={paretoOnly}
                onChange={(e) => setParetoOnly(e.target.checked)}
                className="theme-checkbox h-3.5 w-3.5 rounded"
              />
              Pareto Frontier Only
            </label>
            <label className="inline-flex cursor-pointer items-center gap-1.5 font-medium hover:opacity-85">
              <input
                type="checkbox"
                checked={showCiBars}
                onChange={(e) => setShowCiBars(e.target.checked)}
                className="theme-checkbox h-3.5 w-3.5 rounded"
              />
              95% CI Error Bars
            </label>
          </div>
        </div>
      </div>

      {/* Main Chart or Sortable Table Body */}
      <div className="p-4 sm:p-5">
        {viewMode === 'graph' ? (
          <div>
            {plottedPoints.length === 0 ? (
              <div className="theme-border theme-ink-muted flex h-64 items-center justify-center rounded-lg border border-dashed text-sm">
                No evaluation runs match the active regime/frontier filters.
              </div>
            ) : (
              <div ref={plotContainerRef} className="w-full overflow-x-auto" />
            )}

            <div className="theme-border theme-ink-muted mt-3 flex flex-wrap items-center justify-between gap-2 border-t pt-3 text-[11px]">
              <span>
                Hover any point for Wilson 95% CI, <KatexSpan tex="\operatorname{logit}(p)" />, and memory/compute telemetry. Circle = GSM8K, Diamond = MATH (dot radius scales with <KatexSpan tex="0.6\text{B} \to 3\text{B}" />).
              </span>
              <span className="inline-flex items-center gap-1.5 font-mono text-[10.5px]">
                <span
                  className="inline-block h-0.5 w-3.5 border-t border-dashed"
                  style={{ borderColor: REGIME_META.latent_loop.color }}
                />
                Pareto-optimal frontier
              </span>
            </div>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full border-collapse text-left text-xs">
              <thead>
                <tr className="theme-subtle theme-border theme-ink-muted border-b text-[11px] font-semibold tracking-wide uppercase">
                  {(
                    [
                      ['model_name', 'Model'],
                      ['regime', 'Regime'],
                      ['benchmark', 'Bench'],
                      ['accuracy', 'Accuracy (95% CI)'],
                      ['accuracy_logit', 'Logit'],
                      ['mean_tokens_emitted', 'Tokens'],
                      ['flops_per_problem', 'GFLOPs'],
                      ['kv_cache_kb', 'KV-Cache (KB)'],
                    ] as Array<[SortField, string]>
                  ).map(([field, colTitle]) => (
                    <th key={field} className="px-3 py-2.5">
                      <button
                        type="button"
                        onClick={() => handleSort(field)}
                        className="inline-flex items-center gap-1 font-semibold hover:opacity-80"
                      >
                        {field === 'accuracy_logit' ? (
                          <span className="normal-case">
                            Logit <KatexSpan tex="\ln\frac{p}{1-p}" />
                          </span>
                        ) : (
                          colTitle
                        )}
                        {sortField === field && <span>{sortAsc ? '↑' : '↓'}</span>}
                      </button>
                    </th>
                  ))}
                  <th className="px-3 py-2.5">Pareto</th>
                </tr>
              </thead>
              <tbody>
                {sortedTableRuns.map((run) => {
                  const meta = REGIME_META[run.regime];
                  return (
                    <tr
                      key={run.run_id}
                      className="theme-border theme-row-hover border-b transition last:border-b-0"
                    >
                      <td className="theme-ink-strong px-3 py-2 font-medium">
                        {run.model_name}
                      </td>
                      <td className="px-3 py-2">
                        <span className="theme-ink-body inline-flex items-center gap-1.5 font-medium">
                          <span
                            className="h-2 w-2 rounded-full"
                            style={{ backgroundColor: meta.color }}
                          />
                          {meta.shortLabel}
                        </span>
                      </td>
                      <td className="theme-ink-muted px-3 py-2 font-mono">{run.benchmark}</td>
                      <td className="theme-ink-strong px-3 py-2 font-mono">
                        {(run.accuracy * 100).toFixed(1)}%{' '}
                        <span className="theme-ink-faint text-[10.5px]">
                          [{(run.ci_95[0] * 100).toFixed(1)}, {(run.ci_95[1] * 100).toFixed(1)}]
                        </span>
                      </td>
                      <td className="theme-ink-muted px-3 py-2 font-mono">
                        {run.accuracy_logit.toFixed(3)}
                      </td>
                      <td className="theme-ink-body px-3 py-2 font-mono">
                        {run.mean_tokens_emitted.toFixed(0)}
                        {run.latent_steps_k > 0 && (
                          <span className="theme-ink-muted ml-1.5 text-[10.5px]">
                            (<KatexSpan tex={`+${run.latent_steps_k}\\,z_k`} />)
                          </span>
                        )}
                      </td>
                      <td className="theme-ink-body px-3 py-2 font-mono">
                        {(run.flops_per_problem / 1e9).toFixed(1)}
                      </td>
                      <td className="theme-ink-body px-3 py-2 font-mono">
                        {run.kv_cache_kb.toLocaleString()}
                      </td>
                      <td className="px-3 py-2">
                        {run.is_pareto_frontier ? (
                          <span className="theme-muted theme-border theme-ink-strong inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] font-medium">
                            <span
                              className="h-1.5 w-1.5 rounded-full"
                              style={{ backgroundColor: REGIME_META.latent_loop.color }}
                            />
                            Frontier
                          </span>
                        ) : (
                          <span className="theme-ink-faint text-[10px]">Dominated</span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </section>
  );
}
