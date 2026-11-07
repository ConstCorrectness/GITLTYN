export type Regime = 'verbose_cot' | 'telegraphic_cot' | 'compute_matched' | 'latent_loop';

export type TokenCategory =
  | 'discourse_marker'
  | 'determiner_aux'
  | 'preposition_conj'
  | 'state_transition';

export const SIDECAR_PROBE_COLOR = '#6366f1';

export const REGIME_META: Record<Regime, { label: string; shortLabel: string; color: string }> = {
  verbose_cot: {
    label: 'Verbose CoT (Baseline)',
    shortLabel: 'Verbose CoT',
    color: '#64748b',
  },
  telegraphic_cot: {
    label: 'Telegraphic CoT (Pruned Syntax)',
    shortLabel: 'Telegraphic CoT',
    color: '#f59e0b',
  },
  compute_matched: {
    label: 'Compute-Matched Budget',
    shortLabel: 'Compute-Matched',
    color: '#ef4444',
  },
  latent_loop: {
    label: 'Latent Loop + Tee (Ours)',
    shortLabel: 'Latent Loop + Tee',
    color: '#10b981',
  },
};

export const REGIME_ORDER: readonly Regime[] = [
  'verbose_cot',
  'telegraphic_cot',
  'compute_matched',
  'latent_loop',
] as const;

export interface AblationRun {
  run_id: string;
  model_name: string;
  param_count_b: number;
  regime: Regime;
  benchmark: string;
  accuracy: number;
  accuracy_logit: number;
  ci_95: [number, number];
  mean_tokens_emitted: number;
  latent_steps_k: number;
  flops_per_problem: number;
  kv_cache_kb: number;
  is_pareto_frontier: boolean;
}

export interface LogitLensItem {
  token: string;
  prob: number;
}

export interface LatentStepTrace {
  step_k: number;
  l2_norm: number;
  logit_lens_top_k: LogitLensItem[];
  sidecar_decoded: string;
}

export interface TeeTrace {
  problem_id: string;
  prompt: string;
  final_answer: string;
  latent_steps: LatentStepTrace[];
}

export interface ClassifiedTokenRecord {
  index: number;
  text: string;
  category: TokenCategory;
  is_syntactic: boolean;
  prune_threshold: number;
}

export interface PruningTrace {
  trace_id: string;
  benchmark: string;
  model_name: string;
  param_count_b: number;
  prompt: string;
  prompt_tokens: number;
  kv_kb_per_token: number;
  flops_per_token: number;
  final_answer: string;
  baseline_accuracy: number;
  telegraphic_accuracy: number;
  tokens: ClassifiedTokenRecord[];
}

export interface AblationArtifact {
  generated_at: string;
  runs: AblationRun[];
  tee_traces: TeeTrace[];
  pruning_traces?: PruningTrace[];
}

interface RawRunInput {
  run_id: string;
  model_name: string;
  param_count_b: number;
  regime: string;
  benchmark: string;
  accuracy: number;
  accuracy_logit: number;
  ci_95: number[];
  mean_tokens_emitted: number;
  latent_steps_k: number;
  flops_per_problem: number;
  kv_cache_kb: number;
  is_pareto_frontier: boolean;
}

interface RawPruningTokenInput {
  index: number;
  text: string;
  category: string;
  is_syntactic: boolean;
  prune_threshold: number;
}

interface RawPruningTraceInput {
  trace_id: string;
  benchmark: string;
  model_name: string;
  param_count_b: number;
  prompt: string;
  prompt_tokens: number;
  kv_kb_per_token: number;
  flops_per_token: number;
  final_answer: string;
  baseline_accuracy: number;
  telegraphic_accuracy: number;
  tokens: RawPruningTokenInput[];
}

interface RawArtifactInput {
  generated_at: string;
  runs: RawRunInput[];
  tee_traces: TeeTrace[];
  pruning_traces?: RawPruningTraceInput[];
}

function isRegime(value: string): value is Regime {
  return (
    value === 'verbose_cot' ||
    value === 'telegraphic_cot' ||
    value === 'compute_matched' ||
    value === 'latent_loop'
  );
}

function isTokenCategory(value: string): value is TokenCategory {
  return (
    value === 'discourse_marker' ||
    value === 'determiner_aux' ||
    value === 'preposition_conj' ||
    value === 'state_transition'
  );
}

export function normalizeAblationArtifact(raw: RawArtifactInput): AblationArtifact {
  const runs: AblationRun[] = raw.runs.map((r) => {
    if (!isRegime(r.regime)) {
      throw new Error(`Invalid regime in artifact: ${r.regime}`);
    }
    const low = r.ci_95[0] ?? r.accuracy;
    const high = r.ci_95[1] ?? r.accuracy;
    return {
      ...r,
      regime: r.regime,
      ci_95: [low, high],
    };
  });

  const pruningTraces: PruningTrace[] | undefined = raw.pruning_traces?.map((trace) => ({
    ...trace,
    tokens: trace.tokens.map((tok) => {
      if (!isTokenCategory(tok.category)) {
        throw new Error(`Invalid token category in artifact: ${tok.category}`);
      }
      return {
        ...tok,
        category: tok.category,
      };
    }),
  }));

  return {
    generated_at: raw.generated_at,
    runs,
    tee_traces: raw.tee_traces,
    pruning_traces: pruningTraces,
  };
}
