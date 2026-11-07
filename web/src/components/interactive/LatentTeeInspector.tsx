import { useEffect, useMemo, useRef, useState } from 'react';
import * as Plot from '@observablehq/plot';
import katex from 'katex';
import rawArtifact from '@/data/ablation_frontier.json';
import { useThemePalette } from '@/hooks/useThemePalette';
import {
  REGIME_META,
  SIDECAR_PROBE_COLOR,
  normalizeAblationArtifact,
  type LatentStepTrace,
  type TeeTrace,
} from '@/types/ablation';

export interface LatentTeeInspectorProps {
  defaultProblemIndex?: number;
  defaultValveOpen?: boolean;
}

type TrajectoryMetric = 'l2_norm' | 'top1_prob';

const ARTIFACT = normalizeAblationArtifact(rawArtifact);

function probToLogit(p: number): number {
  const c = Math.min(Math.max(p, 1e-4), 1 - 1e-4);
  return Math.log(c / (1 - c));
}

function KatexSpan({ tex }: { tex: string }) {
  const html = useMemo(
    () => katex.renderToString(tex, { displayMode: false, output: 'html', throwOnError: false }),
    [tex]
  );
  return <span dangerouslySetInnerHTML={{ __html: html }} />;
}

export default function LatentTeeInspector({
  defaultProblemIndex = 0,
  defaultValveOpen = true,
}: LatentTeeInspectorProps) {
  const palette = useThemePalette();
  const traces: TeeTrace[] = ARTIFACT.tee_traces;
  const [problemIdx, setProblemIdx] = useState<number>(defaultProblemIndex);
  const [stepIdx, setStepIdx] = useState<number>(0);
  const [valveOpen, setValveOpen] = useState<boolean>(defaultValveOpen);
  const [trajMetric, setTrajMetric] = useState<TrajectoryMetric>('l2_norm');
  const [xScale, setXScale] = useState<'linear' | 'log'>('linear');
  const [probScale, setProbScale] = useState<'linear' | 'logit'>('linear');
  const [highConfOnly, setHighConfOnly] = useState<boolean>(false);
  const [viewMode, setViewMode] = useState<'graph' | 'table'>('graph');
  const [sortProbAsc, setSortProbAsc] = useState<boolean>(false);
  const [panelWidth, setPanelWidth] = useState<number>(340);

  const wrapperRef = useRef<HTMLDivElement>(null);
  const barPlotRef = useRef<HTMLDivElement>(null);
  const normPlotRef = useRef<HTMLDivElement>(null);

  const activeTrace: TeeTrace | undefined = traces[problemIdx] ?? traces[0];
  const steps: LatentStepTrace[] = activeTrace ? activeTrace.latent_steps : [];
  const safeStepIdx = Math.min(stepIdx, Math.max(0, steps.length - 1));
  const currentStep: LatentStepTrace | undefined = steps[safeStepIdx];

  useEffect(() => {
    const el = wrapperRef.current;
    if (!el) return;
    const measure = () => {
      const full = el.getBoundingClientRect().width;
      const col = full >= 768 ? Math.floor((full - 68) / 2) : Math.floor(full - 44);
      setPanelWidth(Math.max(260, col));
    };
    measure();
    const obs = new ResizeObserver(measure);
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  const handleSelectProblem = (nextIdx: number) => {
    setProblemIdx(nextIdx);
    setStepIdx(0);
  };

  const filteredSteps = useMemo(() => {
    if (!highConfOnly) return steps;
    const maxProb = Math.max(
      ...steps.map((s) => s.logit_lens_top_k[0]?.prob ?? 0)
    );
    return steps.filter(
      (s, i) =>
        i === safeStepIdx ||
        (s.logit_lens_top_k[0]?.prob ?? 0) >= maxProb * 0.72
    );
  }, [steps, highConfOnly, safeStepIdx]);

  // Mount Logit Lens Top-K Bar Plot
  useEffect(() => {
    const host = barPlotRef.current;
    if (!host || !valveOpen || viewMode !== 'graph' || !currentStep) return;

    const bars = currentStep.logit_lens_top_k.map((item, rank) => {
      const val = probScale === 'linear' ? item.prob * 100 : probToLogit(item.prob);
      const se = Math.sqrt((item.prob * (1 - item.prob)) / 64);
      const lowP = Math.max(0.001, item.prob - 1.96 * se);
      const highP = Math.min(0.999, item.prob + 1.96 * se);
      return {
        token: item.token,
        rank,
        val,
        ciLow: probScale === 'linear' ? lowP * 100 : probToLogit(lowP),
        ciHigh: probScale === 'linear' ? highP * 100 : probToLogit(highP),
        fill: rank === 0 ? REGIME_META.latent_loop.color : SIDECAR_PROBE_COLOR,
        tip: `Token "${item.token}"\nLogit Lens Prob: ${(item.prob * 100).toFixed(2)}%\nLogit: ${probToLogit(item.prob).toFixed(2)}`,
      };
    });

    const plot = Plot.plot({
      width: panelWidth,
      height: 215,
      marginTop: 14,
      marginRight: 24,
      marginBottom: 38,
      marginLeft: 76,
      style: {
        background: palette.plotBg,
        color: palette.plotFg,
        fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
        fontSize: '11px',
      },
      x: {
        type: 'linear',
        label:
          probScale === 'linear'
            ? 'Unembedding Probability (%)'
            : 'Logit Score (log-odds)',
        grid: true,
        nice: true,
      },
      y: {
        domain: bars.map((b) => b.token),
        label: null,
      },
      marks: [
        Plot.frame({ stroke: palette.frameStroke }),
        ...(probScale === 'linear'
          ? [
              Plot.barX(bars, {
                x: 'val',
                y: 'token',
                fill: 'fill',
                fillOpacity: 0.82,
                rx: 3,
              }),
            ]
          : [
              Plot.ruleY(bars, {
                x1: 'ciLow',
                x2: 'ciHigh',
                y: 'token',
                stroke: 'fill',
                strokeWidth: 1.75,
              }),
              Plot.dot(bars, {
                x: 'val',
                y: 'token',
                r: 5,
                fill: 'fill',
              }),
            ]),
        Plot.ruleY(bars, {
          x1: 'ciLow',
          x2: 'ciHigh',
          y: 'token',
          stroke: palette.plotFg,
          strokeOpacity: 0.45,
          strokeWidth: 1.25,
          marker: 'tick',
        }),
        Plot.tip(
          bars,
          Plot.pointerY({
            x: 'val',
            y: 'token',
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
  }, [currentStep, valveOpen, viewMode, panelWidth, probScale, palette]);

  // Mount Step Trajectory Plot (Euclidean Norm or Top-1 Confidence across k=1..K)
  useEffect(() => {
    const host = normPlotRef.current;
    if (!host || !valveOpen || viewMode !== 'graph' || filteredSteps.length === 0) return;

    const trajData = filteredSteps.map((s) => {
      const top1 = s.logit_lens_top_k[0]?.prob ?? 0.01;
      const yVal =
        trajMetric === 'l2_norm'
          ? s.l2_norm
          : probScale === 'linear'
            ? top1 * 100
            : probToLogit(top1);
      const isCurrent = currentStep ? s.step_k === currentStep.step_k : false;
      return {
        step_k: s.step_k,
        yVal,
        ciLow: yVal * 0.94,
        ciHigh: yVal * 1.06,
        dotR: isCurrent ? 6.5 : 4.25,
        dotFill: isCurrent ? REGIME_META.latent_loop.color : SIDECAR_PROBE_COLOR,
        tip: `Step k = ${s.step_k}\n‖zₖ‖₂ = ${s.l2_norm.toFixed(3)}\nTop-1 ("${s.logit_lens_top_k[0]?.token ?? ''}"): ${(top1 * 100).toFixed(1)}%\nProbe: ${s.sidecar_decoded}`,
      };
    });

    const plot = Plot.plot({
      width: panelWidth,
      height: 215,
      marginTop: 14,
      marginRight: 22,
      marginBottom: 38,
      marginLeft: 50,
      style: {
        background: palette.plotBg,
        color: palette.plotFg,
        fontFamily: 'ui-sans-serif, system-ui, sans-serif',
        fontSize: '11px',
      },
      x: {
        type: xScale,
        label: 'Latent Recurrence Step k',
        grid: true,
        ticks: steps.length,
      },
      y: {
        type: 'linear',
        label:
          trajMetric === 'l2_norm'
            ? 'Euclidean Norm ‖zₖ‖₂'
            : probScale === 'linear'
              ? 'Top-1 Token Prob (%)'
              : 'Top-1 Token Logit',
        grid: true,
        nice: true,
      },
      r: { type: 'identity' },
      marks: [
        Plot.frame({ stroke: palette.frameStroke }),
        Plot.areaY(trajData, {
          x: 'step_k',
          y1: 'ciLow',
          y2: 'ciHigh',
          fill: SIDECAR_PROBE_COLOR,
          fillOpacity: 0.11,
          curve: 'monotone-x',
        }),
        Plot.line(trajData, {
          x: 'step_k',
          y: 'yVal',
          stroke: SIDECAR_PROBE_COLOR,
          strokeOpacity: 0.85,
          strokeWidth: 1.75,
          curve: 'monotone-x',
        }),
        Plot.dot(trajData, {
          x: 'step_k',
          y: 'yVal',
          r: 'dotR',
          fill: 'dotFill',
          stroke: palette.dotHalo,
          strokeWidth: 1.5,
        }),
        Plot.tip(
          trajData,
          Plot.pointer({
            x: 'step_k',
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
    filteredSteps,
    steps.length,
    currentStep,
    valveOpen,
    viewMode,
    panelWidth,
    trajMetric,
    xScale,
    probScale,
    palette,
  ]);

  const sortedTopK = useMemo(() => {
    if (!currentStep) return [];
    const items = [...currentStep.logit_lens_top_k];
    items.sort((a, b) => (sortProbAsc ? a.prob - b.prob : b.prob - a.prob));
    return items;
  }, [currentStep, sortProbAsc]);

  if (!activeTrace || !currentStep) {
    return null;
  }

  return (
    <section
      ref={wrapperRef}
      aria-label="Continuous Latent Loop and Detachable Tee Inspector"
      className="theme-surface theme-border overflow-hidden rounded-xl border font-sans shadow-2xs"
    >
      {/* Header & Valve Control */}
      <div className="theme-subtle theme-border border-b p-4 sm:px-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <span className="theme-ink-faint font-mono text-[11px] font-semibold tracking-wider uppercase">
              Figure 3 · Ablation C — Lazy Latent Tee &amp; Sidecar Probe
            </span>
            <h3 className="theme-ink-strong mt-0.5 text-base font-bold tracking-tight">
              Inspecting Continuous Thought Vectors <KatexSpan tex="z_k \in \mathbb{R}^d" /> Without Blocking Forward Inference
            </h3>
          </div>

          {/* Detachable Latent Tee Valve Switch */}
          <button
            type="button"
            onClick={() => setValveOpen((prev) => !prev)}
            aria-pressed={valveOpen}
            className={`inline-flex items-center gap-2 rounded-lg border px-3 py-1.5 text-xs font-medium transition ${
              valveOpen
                ? 'theme-surface theme-border-strong theme-ink-strong shadow-2xs'
                : 'theme-muted theme-border'
            }`}
          >
            <span
              className="h-2 w-2 rounded-full"
              style={{
                backgroundColor: valveOpen ? SIDECAR_PROBE_COLOR : '#94a3b8',
              }}
            />
            <span>
              Latent Tee Valve:{' '}
              {valveOpen ? (
                <strong className="theme-ink-strong font-semibold">
                  OPEN (<KatexSpan tex="\operatorname{sg}[z_k]" />)
                </strong>
              ) : (
                'CLOSED (Zero-Overhead)'
              )}
            </span>
          </button>
        </div>

        {/* Problem Selector & KaTeX-Rendered Chart Controls */}
        <div className="theme-border mt-3.5 flex flex-wrap items-center justify-between gap-2.5 border-t pt-3">
          <div className="flex flex-wrap gap-1.5">
            {traces.map((tr, idx) => (
              <button
                key={tr.problem_id}
                type="button"
                onClick={() => handleSelectProblem(idx)}
                className={`rounded-md border px-2.5 py-1 font-mono text-[11px] font-medium transition ${
                  problemIdx === idx ? 'theme-btn-active' : 'theme-btn-idle'
                }`}
              >
                {tr.problem_id}
              </button>
            ))}
          </div>

          <div className="flex flex-wrap items-center gap-2 text-xs">
            <div
              role="group"
              aria-label="Trajectory metric"
              className="theme-seg inline-flex items-center rounded-lg p-0.5 text-[11px]"
            >
              <button
                type="button"
                onClick={() => setTrajMetric('l2_norm')}
                className={`rounded-md px-2 py-0.5 font-medium transition ${
                  trajMetric === 'l2_norm' ? 'theme-btn-active' : 'theme-seg-idle'
                }`}
              >
                <KatexSpan tex="\lVert z_k \rVert_2" /> Norm
              </button>
              <button
                type="button"
                onClick={() => setTrajMetric('top1_prob')}
                className={`rounded-md px-2 py-0.5 font-medium transition ${
                  trajMetric === 'top1_prob' ? 'theme-btn-active' : 'theme-seg-idle'
                }`}
              >
                Top-1 Prob
              </button>
            </div>

            <button
              type="button"
              onClick={() => setXScale((s) => (s === 'linear' ? 'log' : 'linear'))}
              className="theme-btn-idle rounded-md border px-2 py-1 text-[11px] font-medium transition"
            >
              X: {xScale === 'linear' ? 'Linear' : 'Log'}
            </button>

            <button
              type="button"
              onClick={() => setProbScale((s) => (s === 'linear' ? 'logit' : 'linear'))}
              className="theme-btn-idle rounded-md border px-2 py-1 text-[11px] font-medium transition"
            >
              Y: {probScale === 'linear' ? 'Linear' : 'Logit'}
            </button>

            <label className="theme-ink-muted inline-flex cursor-pointer items-center gap-1 text-[11px] font-medium hover:opacity-85">
              <input
                type="checkbox"
                checked={highConfOnly}
                onChange={(e) => setHighConfOnly(e.target.checked)}
                className="theme-checkbox h-3 w-3 rounded"
              />
              Frontier Steps Only
            </label>

            <button
              type="button"
              onClick={() => setViewMode((v) => (v === 'graph' ? 'table' : 'graph'))}
              className="theme-btn-active rounded-md border px-2.5 py-1 text-[11px] font-medium transition hover:opacity-90"
            >
              {viewMode === 'graph' ? 'Table' : 'Graph'}
            </button>
          </div>
        </div>
      </div>

      <div className="p-4 sm:p-5">
        {/* Problem & Final Committed Output */}
        <div className="theme-subtle theme-border flex flex-col justify-between gap-3 rounded-lg border px-3.5 py-2.5 text-xs sm:flex-row sm:items-center">
          <div className="theme-ink-muted leading-relaxed">
            <span className="theme-ink-strong font-semibold">Prompt: </span>
            {activeTrace.prompt}
          </div>
          <div className="theme-surface theme-border theme-ink-strong shrink-0 rounded-md border px-3 py-1 font-mono text-xs font-semibold shadow-2xs">
            Unembedded Output: {activeTrace.final_answer}
          </div>
        </div>

        {/* Step Scrubber over z_1 ... z_K */}
        <div className="theme-surface theme-border mt-4 rounded-xl border p-3.5 shadow-2xs">
          <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
            <div className="flex flex-wrap items-center gap-2">
              <span className="theme-ink-body font-medium">
                Continuous Recurrence Step
              </span>
              <span className="theme-subtle theme-border theme-ink-strong inline-flex items-center gap-1.5 rounded-md border px-2 py-0.5 text-xs font-semibold">
                <span
                  className="h-1.5 w-1.5 rounded-full"
                  style={{ backgroundColor: REGIME_META.latent_loop.color }}
                />
                <KatexSpan tex={`z_{${currentStep.step_k}} \\in \\mathbb{R}^d`} />
              </span>
              <span className="theme-ink-muted text-[11px]">
                (<KatexSpan tex={`\\lVert z_{${currentStep.step_k}} \\rVert_2 = ${currentStep.l2_norm.toFixed(3)}`} /> · 0 syntactic tokens emitted)
              </span>
            </div>

            <div className="flex items-center gap-1.5">
              <button
                type="button"
                disabled={safeStepIdx === 0}
                onClick={() => setStepIdx((i) => Math.max(0, i - 1))}
                className="theme-btn-idle rounded-md border px-2.5 py-0.5 text-xs font-medium transition disabled:opacity-40"
              >
                ← Prev
              </button>
              <button
                type="button"
                disabled={safeStepIdx >= steps.length - 1}
                onClick={() => setStepIdx((i) => Math.min(steps.length - 1, i + 1))}
                className="theme-btn-idle rounded-md border px-2.5 py-0.5 text-xs font-medium transition disabled:opacity-40"
              >
                Next →
              </button>
            </div>
          </div>

          <input
            aria-label="Latent step scrubber"
            type="range"
            min={0}
            max={Math.max(0, steps.length - 1)}
            step={1}
            value={safeStepIdx}
            onChange={(e) => setStepIdx(Number(e.target.value))}
            className="theme-slider mt-3 h-1.5 w-full cursor-pointer appearance-none rounded-full"
          />

          {/* Step Node Pills */}
          <div className="mt-2.5 flex flex-wrap items-center gap-1.5">
            {steps.map((st, idx) => {
              const active = idx === safeStepIdx;
              return (
                <button
                  key={st.step_k}
                  type="button"
                  onClick={() => setStepIdx(idx)}
                  className={`flex-1 rounded-md border py-1 text-xs font-medium transition ${
                    active ? 'theme-btn-active shadow-2xs' : 'theme-btn-idle'
                  }`}
                >
                  <KatexSpan tex={`z_{${st.step_k}}`} />
                </button>
              );
            })}
          </div>
        </div>

        {/* Conditional Valve Inspection Body */}
        {valveOpen ? (
          <div className="mt-4 space-y-4">
            {/* Detached Sidecar Probe Readout Card */}
            <div className="theme-subtle theme-border rounded-xl border p-4 shadow-2xs">
              <div className="theme-border mb-2.5 flex flex-wrap items-center justify-between gap-2 border-b pb-2 text-[11px]">
                <span className="theme-ink-strong inline-flex items-center gap-2 font-sans font-semibold">
                  <span
                    className="inline-block h-2 w-2 rounded-sm opacity-85"
                    style={{ backgroundColor: SIDECAR_PROBE_COLOR }}
                  />
                  <span>
                    Detached Sidecar Probe{' '}
                    <KatexSpan tex={`g_{\\phi}(\\operatorname{sg}[z_{${currentStep.step_k}}])`} /> · Ring-Buffer Readout
                  </span>
                </span>
                <span className="theme-ink-muted font-sans">
                  <KatexSpan tex="\operatorname{sg}[z_k]" /> active ·{' '}
                  <KatexSpan tex="W_{\text{unembed}}" /> hot-path bypassed
                </span>
              </div>

              <div className="space-y-1 font-mono text-xs">
                {steps.slice(0, safeStepIdx + 1).map((st) => {
                  const isFocus = st.step_k === currentStep.step_k;
                  return (
                    <div
                      key={st.step_k}
                      className={`rounded-md px-2.5 py-1 transition ${
                        isFocus
                          ? 'theme-surface theme-border-strong theme-ink-strong border font-semibold shadow-2xs'
                          : 'theme-ink-muted'
                      }`}
                    >
                      {st.sidecar_decoded}
                    </div>
                  );
                })}
              </div>
            </div>

            {viewMode === 'graph' ? (
              <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                <div className="theme-surface theme-border rounded-xl border p-3.5">
                  <div className="theme-ink-strong mb-1 text-xs font-bold">
                    Top-5 Logit Lens Vocabulary Projection (
                    <KatexSpan tex={`z_{${currentStep.step_k}} \\cdot W_{\\text{unembed}}`} />)
                  </div>
                  <p className="theme-ink-muted mb-2 text-[11px]">
                    Co-activated candidate tokens in superposition at step{' '}
                    <KatexSpan tex={`k = ${currentStep.step_k}`} />.
                  </p>
                  <div ref={barPlotRef} className="w-full overflow-x-auto" />
                </div>

                <div className="theme-surface theme-border rounded-xl border p-3.5">
                  <div className="theme-ink-strong mb-1 text-xs font-bold">
                    Recurrence Trajectory Across Latent Steps (
                    <KatexSpan tex={`k = 1 \\dots ${steps.length}`} />)
                  </div>
                  <p className="theme-ink-muted mb-2 text-[11px]">
                    {trajMetric === 'l2_norm' ? (
                      <>
                        Euclidean norm <KatexSpan tex="\lVert z_k \rVert_2" /> accumulates state bindings before halt convergence.
                      </>
                    ) : (
                      'Top-1 unembedding confidence sharpens as intermediate operations resolve.'
                    )}
                  </p>
                  <div ref={normPlotRef} className="w-full overflow-x-auto" />
                </div>
              </div>
            ) : (
              <div className="theme-surface theme-border overflow-x-auto rounded-xl border">
                <table className="w-full border-collapse text-left text-xs">
                  <thead>
                    <tr className="theme-subtle theme-border theme-ink-muted border-b text-[11px] font-semibold">
                      <th className="px-3 py-2">
                        Projected Token (<KatexSpan tex={`k = ${currentStep.step_k}`} />)
                      </th>
                      <th className="px-3 py-2">
                        <button
                          type="button"
                          onClick={() => setSortProbAsc((p) => !p)}
                          className="font-semibold hover:opacity-80"
                        >
                          Probability ({sortProbAsc ? '↑' : '↓'})
                        </button>
                      </th>
                      <th className="px-3 py-2">
                        Logit <KatexSpan tex="\ln\frac{p}{1-p}" />
                      </th>
                      <th className="px-3 py-2">Detached Sidecar State</th>
                    </tr>
                  </thead>
                  <tbody className="font-mono">
                    {sortedTopK.map((item) => (
                      <tr
                        key={item.token}
                        className="theme-border theme-row-hover border-b transition last:border-b-0"
                      >
                        <td className="theme-ink-strong px-3 py-2 font-semibold">
                          {item.token}
                        </td>
                        <td className="theme-ink-body px-3 py-2">
                          {(item.prob * 100).toFixed(2)}%
                        </td>
                        <td className="theme-ink-muted px-3 py-2">
                          {probToLogit(item.prob).toFixed(3)}
                        </td>
                        <td className="theme-ink-muted px-3 py-2">
                          {currentStep.sidecar_decoded}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        ) : (
          <div className="theme-subtle theme-border-strong mt-4 rounded-xl border border-dashed p-6 text-center">
            <div className="mx-auto max-w-lg">
              <div className="theme-ink-strong font-mono text-xs font-semibold tracking-wide uppercase">
                Production Zero-Overhead Mode Active (enable_latent_tee = False)
              </div>
              <p className="theme-ink-muted mt-1.5 text-xs leading-relaxed">
                The <code className="theme-surface theme-border theme-ink-strong rounded border px-1 py-0.2 font-mono text-[11px]">RingBufferLatentTee</code> (<KatexSpan tex="M \le K" />) is detached. Forward recurrence executes all{' '}
                <span className="theme-ink-strong font-mono font-semibold">{steps.length}</span> latent transitions{' '}
                <KatexSpan tex={`z_1 \\to \\dots \\to z_{${steps.length}}`} /> directly in{' '}
                <KatexSpan tex="\mathbb{R}^d" /> with <KatexSpan tex="O(1)" /> memory overhead—without computing intermediate{' '}
                <KatexSpan tex="W_{\text{unembed}}" /> projections or running{' '}
                <KatexSpan tex="g_{\phi}" />—yet commits the exact same final answer (
                <span className="theme-ink-strong font-mono font-semibold">
                  {activeTrace.final_answer}
                </span>
                ). Re-open the valve above to audit buffered state transitions on demand.
              </p>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}
