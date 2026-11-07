export type Regime = 'verbose_cot' | 'telegraphic_cot' | 'compute_matched' | 'latent_loop';

export const REGIME_META: Record<Regime, { label: string; color: string }> = {
  verbose_cot: { label: 'Verbose CoT (Baseline)', color: '#64748b' },
  telegraphic_cot: { label: 'Telegraphic CoT (Pruned Syntax)', color: '#f59e0b' },
  compute_matched: { label: 'Compute-Matched Budget', color: '#ef4444' },
  latent_loop: { label: 'Latent Loop + Tee (Ours)', color: '#10b981' },
};

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

export interface LatentStepTrace {
  step_k: number;
  l2_norm: number;
  logit_lens_top_k: Array<{ token: string; prob: number }>;
  sidecar_decoded: string;
}

export interface TeeTrace {
  problem_id: string;
  prompt: string;
  final_answer: string;
  latent_steps: LatentStepTrace[];
}

export interface AblationArtifact {
  generated_at: string;
  runs: AblationRun[];
  tee_traces: TeeTrace[];
}

