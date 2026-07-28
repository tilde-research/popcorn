import type { Kernel, KernelRow } from '@/lib/kernels';

export const MAX_PINNED = 4;

/** Line dash by position in the plotted kernel list (pinned first, transient selection last). */
export const DASHES = ['solid', 'dot', 'dash', 'dashdot', 'longdash'];

/** Stable backend colors, consistent across every graph. */
const ASSIGNED: Record<string, string> = {
  torch: '#9ca3af',
  fa3: '#6366f1',
  fla: '#f59e0b',
  liger: '#10b981',
  quack: '#ef4444',
  popcorn: '#06b6d4',
};
const FALLBACK = ['#d946ef', '#84cc16', '#f97316', '#0ea5e9', '#a855f7'];

export function backendColor(name: string): string {
  if (name in ASSIGNED) return ASSIGNED[name];
  let hash = 0;
  for (const char of name) hash = (hash * 31 + char.charCodeAt(0)) >>> 0;
  return FALLBACK[hash % FALLBACK.length];
}

export function fmt(value: unknown): string {
  if (value === null || value === undefined) return 'None';
  if (value === true) return 'True';
  if (value === false) return 'False';
  return String(value);
}

/** Short chip label for a recorded `device` string (full name stays the filter key). */
const HARDWARE_TAGS = [
  'B200',
  'H200',
  'H100',
  'A100',
  'A10',
  'L40S',
  'L40',
  'V100',
  'MI355',
  'MI300',
  'MI250',
] as const;

export function hardwareLabel(device: string): string {
  const upper = device.toUpperCase();
  for (const tag of HARDWARE_TAGS) {
    if (upper.includes(tag)) return tag;
  }
  const rtx = device.match(/((?:RTX|GTX)\s*\d+\s*\w*)/i);
  if (rtx) return rtx[1].replace(/\s+/g, ' ');
  return device.replace(/^NVIDIA\s+/i, '').replace(/\s+\d+GB\b.*$/i, '').trim() || device;
}

/** Tokens processed by one call: batch shape x token-carrying dims. */
const TOKEN_DIMS = new Set(['batch', 'seq', 'total', 'tokens', 'resp', 'response']);

export function tokens(row: KernelRow): number {
  let count = row.batch.reduce((product, size) => product * size, 1);
  for (const [name, size] of Object.entries(row.dims)) if (TOKEN_DIMS.has(name)) count *= size;
  return count;
}

export type Pass = 'forward' | 'backward' | 'both';
export const PASSES: Pass[] = ['forward', 'backward', 'both'];

export function combine(pass: Pass, fwd: number | undefined, bwd: number | undefined, how: 'sum' | 'max'): number | null {
  if (pass === 'forward') return fwd ?? null;
  if (pass === 'backward') return bwd ?? null;
  if (fwd === undefined || bwd === undefined) return null;
  return how === 'sum' ? fwd + bwd : Math.max(fwd, bwd);
}

export interface Metric {
  id: string;
  unit: string;
  value: (row: KernelRow, pass: Pass) => number | null;
  cutoff?: (row: KernelRow, pass: Pass) => number | null;
  /** Hide the metric when no row can produce it. */
  available: (rows: KernelRow[]) => boolean;
}

const latency = (row: KernelRow, pass: Pass) => combine(pass, row.fwd_ms, row.bwd_ms, 'sum');

export const METRICS: Metric[] = [
  { id: 'latency', unit: 'ms', value: latency, available: () => true },
  {
    id: 'throughput',
    unit: 'tokens/s',
    value: (row, pass) => {
      const ms = latency(row, pass);
      return ms === null ? null : (tokens(row) * 1000) / ms;
    },
    available: (rows) => rows.some((row) => tokens(row) > 1),
  },
  {
    id: 'memory',
    unit: 'MB',
    value: (row, pass) => combine(pass, row.fwd_mem_mb, row.bwd_mem_mb, 'max'),
    available: (rows) => rows.some((row) => row.fwd_mem_mb !== undefined),
  },
  {
    id: 'abs error',
    unit: 'abs err',
    value: (row, pass) => combine(pass, row.fwd_err, row.bwd_err, 'max'),
    cutoff: (row, pass) => combine(pass, row.fwd_cut, row.bwd_cut, 'max'),
    available: (rows) => rows.some((row) => row.fwd_err !== undefined),
  },
  {
    id: 'rel error',
    unit: 'rel err',
    value: (row, pass) => combine(pass, row.fwd_rel, row.bwd_rel, 'max'),
    cutoff: (row, pass) => combine(pass, row.fwd_rel_cut, row.bwd_rel_cut, 'max'),
    available: (rows) => rows.some((row) => row.fwd_rel !== undefined),
  },
];

/** Everything selectable, derived from the union of the plotted kernels' rows;
 * x-axis candidates are the *intersection* of per-kernel multi-valued dims. */
export interface Controls {
  xOptions: string[];
  dims: [string, number[]][];
  args: [string, string[]][];
  /** Single-valued dims, shown as fixed context rather than controls. */
  fixed: [string, number][];
  /** Most benchmarked value per dim/arg; used when nothing is selected. */
  dimDefaults: Record<string, number>;
  argDefaults: Record<string, string>;
  dtypes: string[];
  devices: string[];
  presents: string[];
  batches: string[];
}

const distinct = <T,>(values: T[]): T[] => [...new Set(values)];
const presentKey = (row: KernelRow) => row.present.join('+') || 'none';
const batchKey = (row: KernelRow) => row.batch.join('x') || 'none';

export function controlsFor(kernels: Kernel[]): Controls {
  const all = kernels.flatMap((kernel) => kernel.rows);
  const dims = new Map<string, number[]>();
  for (const row of all)
    for (const [name, value] of Object.entries(row.dims)) {
      const pool = dims.get(name) ?? [];
      if (!pool.includes(value)) pool.push(value);
      dims.set(name, pool);
    }
  for (const pool of dims.values()) pool.sort((a, b) => a - b);

  const multiValued = (kernel: Kernel) => {
    const seen = new Map<string, Set<number>>();
    for (const row of kernel.rows)
      for (const [name, value] of Object.entries(row.dims)) (seen.get(name) ?? seen.set(name, new Set()).get(name)!).add(value);
    return new Set([...seen].filter(([, values]) => values.size > 1).map(([name]) => name));
  };
  const xOptions = [...dims.keys()].filter((name) => kernels.every((kernel) => multiValued(kernel).has(name)));

  const argKeys = distinct(all.flatMap((row) => Object.keys(row.args)));
  const args: [string, string[]][] = argKeys
    .map((key): [string, string[]] => [
      key,
      distinct(all.filter((row) => key in row.args).map((row) => fmt(row.args[key]))).sort(),
    ])
    .filter(([, values]) => values.length > 1);

  const dimDefaults: Record<string, number> = {};
  for (const [name, pool] of dims)
    if (pool.length > 1) dimDefaults[name] = mode(all.filter((row) => name in row.dims).map((row) => row.dims[name]));
  const argDefaults: Record<string, string> = {};
  for (const [key] of args) argDefaults[key] = mode(all.filter((row) => key in row.args).map((row) => fmt(row.args[key])));

  return {
    xOptions,
    dims: [...dims.entries()],
    args,
    fixed: [...dims.entries()].filter(([, pool]) => pool.length === 1).map(([name, pool]): [string, number] => [name, pool[0]]),
    dimDefaults,
    argDefaults,
    dtypes: distinct(all.map((row) => row.dtype)).sort(),
    devices: distinct(all.map((row) => row.device)).sort(),
    presents: distinct(all.map(presentKey)).sort(),
    batches: distinct(all.map(batchKey)).sort(),
  };
}

export interface Selection {
  x: string;
  metric: string;
  pass: Pass;
  dtype: string;
  device: string;
  present: string;
  batch: string;
  dims: Record<string, number>;
  args: Record<string, string>;
}

export function mode<T>(values: T[]): T {
  const counts = new Map<T, number>();
  for (const value of values) counts.set(value, (counts.get(value) ?? 0) + 1);
  let best = values[0];
  for (const [value, count] of counts) if (count > (counts.get(best) ?? 0)) best = value;
  return best;
}

export function defaultSelection(kernels: Kernel[], controls: Controls): Selection {
  const all = kernels.flatMap((kernel) => kernel.rows);
  // Prefer a sequence-like sweep over batch when both are available.
  const priority = ['seq', 'total', 'tokens', 'resp', 'response', 'batch'];
  const seqish = priority.find((name) => controls.xOptions.includes(name));
  return {
    x: seqish ?? controls.xOptions[0] ?? '',
    metric: 'latency',
    pass: 'forward',
    dtype: mode(all.map((row) => row.dtype)),
    device: mode(all.map((row) => row.device)),
    present: mode(all.map(presentKey)),
    batch: mode(all.map(batchKey)),
    dims: {},
    args: {},
  };
}

/** Nearest recorded value in log space, for kernels whose grid lacks the exact selection. */
function snap(target: number, pool: number[]): number {
  let best = pool[0];
  for (const value of pool)
    if (Math.abs(Math.log(value) - Math.log(target)) < Math.abs(Math.log(best) - Math.log(target))) best = value;
  return best;
}

/** Rows of one kernel matching the selection; dims snap to the kernel's own
 * grid, keys the kernel does not have are ignored (union semantics). */
export function filterRows(kernel: Kernel, selection: Selection, controls: Controls): KernelRow[] {
  const pools = new Map<string, number[]>();
  for (const row of kernel.rows)
    for (const [name, value] of Object.entries(row.dims)) {
      const pool = pools.get(name) ?? [];
      if (!pool.includes(value)) pool.push(value);
      pools.set(name, pool);
    }
  const wanted = new Map<string, number>();
  for (const [name, pool] of pools) {
    if (name === selection.x || pool.length === 1) continue;
    wanted.set(name, snap(selection.dims[name] ?? controls.dimDefaults[name] ?? pool[0], pool));
  }
  const grad = selection.pass !== 'forward';
  const candidates = kernel.rows.filter(
    (row) =>
      row.dtype === selection.dtype &&
      row.device === selection.device &&
      presentKey(row) === selection.present &&
      batchKey(row) === selection.batch &&
      (!grad || row.grad) &&
      [...wanted].every(([name, value]) => row.dims[name] === value) &&
      Object.entries(row.args).every(
        ([key, value]) => (selection.args[key] ?? controls.argDefaults[key] ?? fmt(value)) === fmt(value),
      ),
  );
  // Forward pass: prefer the no-grad measurement when a case was recorded both ways.
  const byCase = new Map<string, KernelRow>();
  for (const row of candidates) {
    const key = `${row.impl}|${JSON.stringify(row.dims)}`;
    const current = byCase.get(key);
    if (!current || (selection.pass === 'forward' ? !row.grad && current.grad : row.grad && !current.grad))
      byCase.set(key, row);
  }
  return [...byCase.values()];
}
