import type {
  Kernel,
  KernelCase,
  KernelCurve,
  KernelFrontier,
  KernelResult,
  KernelRow,
} from '../../lib/kernels.js';

export const MAX_PINNED = 4;
export const DASHES = ['solid', 'dot', 'dash', 'dashdot', 'longdash'];

const ASSIGNED: Record<string, string> = {
  torch: '#9ca3af',
  fa3: '#6366f1',
  fla: '#f59e0b',
  liger: '#10b981',
  quack: '#ef4444',
  popcorn: '#06b6d4',
};
const FALLBACK = ['#d946ef', '#84cc16', '#f97316', '#0ea5e9', '#a855f7'];
const AXIS_PRIORITY = ['seq', 'total', 'tokens', 'response', 'batch', 'hidden', 'head_dim'];

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

const HARDWARE_TAGS = ['B200', 'H200', 'H100', 'A100', 'A10', 'L40S', 'L40', 'V100', 'MI355', 'MI300', 'MI250'];

export function hardwareLabel(device: string): string {
  const upper = device.toUpperCase();
  for (const tag of HARDWARE_TAGS) if (upper.includes(tag)) return tag;
  const rtx = device.match(/((?:RTX|GTX)\s*\d+\s*\w*)/i);
  if (rtx) return rtx[1].replace(/\s+/g, ' ');
  return device.replace(/^NVIDIA\s+/i, '').replace(/\s+\d+GB\b.*$/i, '').trim() || device;
}

const TOKEN_DIMS = new Set(['batch', 'seq', 'total', 'tokens', 'resp', 'response']);

export function tokens(row: KernelRow): number {
  let count = row.batch.reduce((product, size) => product * size, 1);
  for (const [name, size] of Object.entries(row.dims)) if (TOKEN_DIMS.has(name)) count *= size;
  return count;
}

export type Pass = 'forward' | 'backward' | 'both';
export const PASSES: Pass[] = ['forward', 'backward', 'both'];
export type ExplorerMode = 'curves' | 'coverage';

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
  available: (rows: KernelRow[]) => boolean;
}

const latency = (row: KernelRow, pass: Pass) => combine(pass, row.fwd_ms, row.bwd_ms, 'sum');

export const METRICS: Metric[] = [
  { id: 'latency', unit: 'ms', value: latency, available: (rows) => rows.some((row) => row.fwd_ms !== undefined) },
  {
    id: 'throughput',
    unit: 'tokens/s',
    value: (row, pass) => {
      const ms = latency(row, pass);
      return ms === null ? null : (tokens(row) * 1000) / ms;
    },
    available: (rows) => rows.some((row) => row.fwd_ms !== undefined && tokens(row) > 1),
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

export interface Selection {
  mode: ExplorerMode;
  curve: string;
  metric: string;
  pass: Pass;
  device: string;
}

export interface CurveOption {
  key: string;
  profile: string;
  axis: string;
  dtype: string;
  variant: string;
  sharedContext: boolean;
}

export function curveKey(curve: KernelCurve): string {
  return [curve.profile, curve.axis, curve.dtype, curve.variant].join('|');
}

export function contextKey(curve: KernelCurve): string {
  return JSON.stringify(curve.fixed);
}

export function contextLabel(curve: KernelCurve): string {
  const dims = Object.entries(curve.fixed.dims).map(([name, value]) => `${name}=${value}`);
  const args = Object.entries(curve.fixed.args).map(([name, value]) => `${name}=${fmt(value)}`);
  const batch = curve.fixed.batch === null ? [] : [`batch=${curve.fixed.batch.join('x') || 'none'}`];
  const present = curve.fixed.present.length > 0 ? [`+${curve.fixed.present.join(',+')}`] : [];
  return [...batch, ...dims, ...args, ...present].join(' · ') || 'no fixed dimensions';
}

export function curveFor(kernel: Kernel, key: string): KernelCurve | undefined {
  return kernel.evidence.curves.find((curve) => curveKey(curve) === key);
}

export function curveOptions(kernels: Kernel[]): CurveOption[] {
  if (kernels.length === 0) return [];
  const maps = kernels.map((kernel) => new Map(kernel.evidence.curves.map((curve) => [curveKey(curve), curve])));
  const keys = [...maps[0].keys()].filter((key) => maps.every((map) => map.has(key)));
  return keys.flatMap((key) => {
    const curves = maps.map((map) => map.get(key)!);
    const contexts = new Set(curves.map(contextKey));
    const devicePools = curves.map((curve, index) => new Set(rowsForCurve(kernels[index], curve).map((item) => item.device)));
    const sharedDevices = [...devicePools[0]].filter((device) => devicePools.every((pool) => pool.has(device)));
    if (contexts.size !== 1 || sharedDevices.length === 0) return [];
    return {
      key,
      profile: curves[0].profile,
      axis: curves[0].axis,
      dtype: curves[0].dtype,
      variant: curves[0].variant,
      sharedContext: true,
    };
  });
}

function row(case_: KernelCase, result: KernelResult): KernelRow {
  return {
    case_id: case_.case_id,
    dtype: case_.dtype,
    dims: case_.dims,
    batch: case_.batch,
    args: case_.args,
    present: case_.present,
    ...result,
  };
}

export function rowsForCurve(kernel: Kernel, curve: KernelCurve, device = '', pass: Pass = 'forward'): KernelRow[] {
  const requireGrad = pass !== 'forward';
  const candidates = curve.case_ids.flatMap((caseId) => {
    const case_ = kernel.evidence.cases[caseId];
    if (!case_) return [];
    return case_.results
      .filter(
        (result) =>
          result.status === 'pass' &&
          (!device || result.device === device) &&
          (!requireGrad || result.grad),
      )
      .map((result) => row(case_, result));
  });
  const chosen = new Map<string, KernelRow>();
  for (const candidate of candidates) {
    const key = `${candidate.case_id}|${candidate.impl}|${candidate.device}`;
    const current = chosen.get(key);
    if (!current || (pass === 'forward' ? !candidate.grad && current.grad : candidate.grad && !current.grad)) {
      chosen.set(key, candidate);
    }
  }
  return [...chosen.values()];
}

export function rowsForSamples(kernel: Kernel, device = ''): KernelRow[] {
  return kernel.evidence.samples.flatMap((caseId) => {
    const case_ = kernel.evidence.cases[caseId];
    if (!case_) return [];
    return case_.results.filter((result) => !device || result.device === device).map((result) => row(case_, result));
  });
}

export function devicesFor(kernels: Kernel[], key: string): string[] {
  const pools = kernels.map((kernel) => {
    const curve = curveFor(kernel, key);
    if (!curve) return new Set<string>();
    return new Set(rowsForCurve(kernel, curve).map((item) => item.device));
  });
  if (pools.length === 0) return [];
  const common = [...pools[0]].filter((device) => pools.every((pool) => pool.has(device)));
  return (kernels.length === 1 ? [...pools[0]] : common).sort();
}

function timedPoints(kernel: Kernel, curve: KernelCurve): number {
  return new Set(rowsForCurve(kernel, curve).filter((item) => item.fwd_ms !== undefined).map((item) => item.case_id)).size;
}

function axisRank(axis: string): number {
  const index = AXIS_PRIORITY.indexOf(axis);
  return index < 0 ? 0 : AXIS_PRIORITY.length - index;
}

export function defaultCurveKey(kernels: Kernel[]): string {
  const options = curveOptions(kernels);
  let best = '';
  let score = [-1, -1, -1, -1];
  for (const option of options) {
    const points = Math.min(...kernels.map((kernel) => timedPoints(kernel, curveFor(kernel, option.key)!)));
    const next = [points >= 2 ? 1 : 0, option.profile === 'production' ? 1 : 0, axisRank(option.axis), points];
    if (next.some((value, index) => value !== score[index] && value > score[index])) {
      const firstDifference = next.findIndex((value, index) => value !== score[index]);
      if (firstDifference >= 0 && next[firstDifference] > score[firstDifference]) {
        best = option.key;
        score = next;
      }
    }
  }
  return best;
}

export function bestCurve(kernel: Kernel): KernelCurve | undefined {
  return [...kernel.evidence.curves].sort((left, right) => {
    const leftScore = [timedPoints(kernel, left) >= 2 ? 1 : 0, left.profile === 'production' ? 1 : 0, axisRank(left.axis), timedPoints(kernel, left)];
    const rightScore = [timedPoints(kernel, right) >= 2 ? 1 : 0, right.profile === 'production' ? 1 : 0, axisRank(right.axis), timedPoints(kernel, right)];
    for (let index = 0; index < leftScore.length; index++) if (leftScore[index] !== rightScore[index]) return rightScore[index] - leftScore[index];
    return left.id.localeCompare(right.id);
  })[0];
}

export function defaultSelection(kernels: Kernel[]): Selection {
  const curve = defaultCurveKey(kernels);
  return {
    mode: 'curves',
    curve,
    metric: 'latency',
    pass: 'forward',
    device: devicesFor(kernels, curve)[0] ?? '',
  };
}

export function normalizedSelection(kernels: Kernel[], selection: Selection): Selection {
  const options = curveOptions(kernels);
  const curve = options.some((option) => option.key === selection.curve) ? selection.curve : defaultCurveKey(kernels);
  const devices = devicesFor(kernels, curve);
  const device = devices.includes(selection.device) ? selection.device : (devices[0] ?? '');
  const rows = kernels.flatMap((kernel) => {
    const selected = curveFor(kernel, curve);
    return selected ? rowsForCurve(kernel, selected, device, selection.pass) : [];
  });
  const metrics = METRICS.filter((metric) => metric.available(rows));
  const metric = metrics.some((item) => item.id === selection.metric) ? selection.metric : (metrics[0]?.id ?? 'latency');
  const pass = selection.pass !== 'forward' && !rows.some((item) => item.bwd_ms !== undefined) ? 'forward' : selection.pass;
  return { ...selection, curve, device, metric, pass };
}

export function comparisonMode(kernels: Kernel[], key: string): 'shared' | 'separate' {
  if (kernels.length < 2) return 'shared';
  const curves = kernels.map((kernel) => curveFor(kernel, key)).filter((curve): curve is KernelCurve => curve !== undefined);
  return curves.length === kernels.length && new Set(curves.map(contextKey)).size === 1 ? 'shared' : 'separate';
}

export function xValue(case_: Pick<KernelCase, 'dims' | 'batch'>, curve: KernelCurve): number {
  return curve.axis === '...' ? (case_.batch[0] ?? 0) : case_.dims[curve.axis];
}

export function frontiersFor(kernel: Kernel, curve: KernelCurve, device: string, pass: Pass): KernelFrontier[] {
  const matching = kernel.evidence.frontiers.filter((frontier) => frontier.curve_id === curve.id && (!device || frontier.device === device));
  if (pass !== 'forward') return matching.filter((frontier) => frontier.grad);
  const noGrad = matching.filter((frontier) => !frontier.grad);
  return noGrad.length > 0 ? noGrad : matching.filter((frontier) => frontier.grad);
}

export function curveStats(kernel: Kernel, curve: KernelCurve, device: string, pass: Pass) {
  const rows = rowsForCurve(kernel, curve, device, pass);
  const xs = rows.map((item) => xValue(item, curve));
  return {
    planned: curve.case_ids.length,
    timed: new Set(rows.filter((item) => item.fwd_ms !== undefined).map((item) => item.case_id)).size,
    max: xs.length > 0 ? Math.max(...xs) : null,
  };
}

export function caseLabel(row: KernelRow): string {
  const dims = Object.entries(row.dims).map(([name, value]) => `${name}=${value}`);
  const batch = row.batch.length > 0 ? [`batch=${row.batch.join('x')}`] : [];
  const args = Object.entries(row.args).map(([name, value]) => `${name}=${fmt(value)}`);
  const present = row.present.map((name) => `+${name}`);
  return [...batch, ...dims, ...args, ...present, row.dtype].join(' ');
}
