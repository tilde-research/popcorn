export interface KernelIndexEntry {
  name: string;
  summary: string | null;
  tags: string[];
  impls: string[];
  rows: number;
  cases: number;
  curves: number;
}

export interface KernelImpl {
  name: string;
  source: string | null;
  forward_only: boolean;
}

export interface KernelResult {
  impl: string;
  device: string;
  grad: boolean;
  status: string;
  benchmarked: boolean;
  bench_error?: true;
  fwd_ms?: number;
  bwd_ms?: number;
  ref_fwd_ms?: number;
  ref_bwd_ms?: number;
  fwd_mem_mb?: number;
  bwd_mem_mb?: number;
  ref_fwd_mem_mb?: number;
  ref_bwd_mem_mb?: number;
  fwd_err?: number;
  bwd_err?: number;
  fwd_cut?: number;
  bwd_cut?: number;
  fwd_rel?: number;
  bwd_rel?: number;
  fwd_rel_cut?: number;
  bwd_rel_cut?: number;
}

export interface KernelCase {
  case_id: string;
  dtype: string;
  dims: Record<string, number>;
  batch: number[];
  args: Record<string, unknown>;
  present: string[];
  results: KernelResult[];
}

/** A case/result join used by the pure plot model, not part of the JSON wire format. */
export interface KernelRow extends KernelResult {
  case_id: string;
  dtype: string;
  dims: Record<string, number>;
  batch: number[];
  args: Record<string, unknown>;
  present: string[];
}

export interface KernelCurve {
  id: string;
  profile: string;
  axis: string;
  dtype: string;
  variant: string;
  fixed: {
    dims: Record<string, number>;
    batch: number[] | null;
    args: Record<string, unknown>;
    present: string[];
  };
  case_ids: string[];
}

export interface KernelFrontier {
  curve_id: string;
  impl: string;
  device: string;
  grad: boolean;
  observed: number;
  timed: number;
  observed_max: number;
  pass_max: number | null;
  terminal?: { x: number; status: string };
}

export interface KernelCoverage {
  planned_samples: number;
  planned_curve_cases: number;
  observed_cases: number;
  results: number;
  timed: number;
  statuses: Record<string, number>;
  by_impl: Record<
    string,
    {
      results: number;
      cases: number;
      timed: number;
      statuses: Record<string, number>;
    }
  >;
}

export interface KernelEvidence {
  cases: Record<string, KernelCase>;
  curves: KernelCurve[];
  samples: string[];
  frontiers: KernelFrontier[];
  coverage: KernelCoverage;
  freshness: {
    latest: string | null;
    current_results: number;
    stale_results_omitted: number;
    ref_hash: string | null;
    impl_hashes: Record<string, string | null>;
  };
}

export interface Kernel {
  name: string;
  summary: string | null;
  math: string | null;
  citations: { label: string; url: string }[];
  tags: string[];
  params: { name: string; default?: string }[];
  impls: KernelImpl[];
  evidence: KernelEvidence;
}

