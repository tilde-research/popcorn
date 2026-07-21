import fs from 'node:fs/promises';
import path from 'node:path';

export interface KernelIndexEntry {
  name: string;
  summary: string | null;
  tags: string[];
  backends: string[];
  rows: number;
}

export interface KernelBackend {
  name: string;
  source: string | null;
  forward_only: boolean;
  supports: Record<string, string>;
}

export interface KernelRow {
  backend: string;
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
}

export interface Kernel {
  name: string;
  summary: string | null;
  math: string | null;
  citations: { label: string; url: string }[];
  tags: string[];
  params: { name: string; default?: string }[];
  backends: KernelBackend[];
  rows: KernelRow[];
}

const DATA = path.join(process.cwd(), 'public', 'data');

export async function kernelIndex(): Promise<KernelIndexEntry[]> {
  return JSON.parse(await fs.readFile(path.join(DATA, 'index.json'), 'utf-8'));
}

export async function kernel(name: string): Promise<Kernel> {
  return JSON.parse(await fs.readFile(path.join(DATA, `${name}.json`), 'utf-8'));
}
