export interface KernelIndexEntry {
  name: string;
  summary: string | null;
  tags: string[];
  impls: string[];
  rows: number;
}

export interface KernelImpl {
  name: string;
  source: string | null;
  forward_only: boolean;
}

export interface KernelRow {
  impl: string;
  device: string;
  dtype: string;
  grad: boolean;
  dims: Record<string, number>;
  batch: number[];
  args: Record<string, unknown>;
  present: string[];
  fwd_ms: number;
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

export interface Kernel {
  name: string;
  summary: string | null;
  math: string | null;
  citations: { label: string; url: string }[];
  tags: string[];
  params: { name: string; default?: string }[];
  impls: KernelImpl[];
  rows: KernelRow[];
}

