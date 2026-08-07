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
export type ExplorerMode = 'curves' | 'slices' | 'coverage';

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

export function rowsForKernel(kernel: Kernel): KernelRow[] {
  return Object.values(kernel.evidence.cases).flatMap((case_) =>
    case_.results.map((result) => row(case_, result)),
  );
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

export interface SliceControls {
  xOptions: string[];
  dims: [string, number[]][];
  args: [string, string[]][];
  fixed: [string, number][];
  dimDefaults: Record<string, number>;
  argDefaults: Record<string, string>;
  dtypes: string[];
  devices: string[];
  presents: string[];
  batches: string[];
}

export interface SliceSelection {
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

export interface MeasuredSlice {
  key: string;
  dims: Record<string, number>;
  batch: string;
  batchValues: number[];
  args: Record<string, string>;
  argValues: Record<string, unknown>;
  present: string;
  presentValues: string[];
  points: number;
  results: number;
}

export type SliceAnchor =
  | { kind: 'dim'; name: string; value: number }
  | { kind: 'arg'; name: string; value: string }
  | { kind: 'batch' | 'present'; value: string };

const distinct = <T,>(values: T[]): T[] => [...new Set(values)];
const presentKey = (row: KernelRow): string => row.present.join('+') || 'none';
const batchKey = (row: KernelRow): string => row.batch.join('x') || 'none';

function mode<T>(values: T[]): T | undefined {
  const counts = new Map<T, number>();
  for (const value of values) counts.set(value, (counts.get(value) ?? 0) + 1);
  return [...counts].sort((left, right) => right[1] - left[1])[0]?.[0];
}

export function sliceControlsFor(kernels: Kernel[]): SliceControls {
  const rowSets = kernels.map(rowsForKernel);
  const all = rowSets.flat();
  const dims = new Map<string, number[]>();
  for (const row of all) {
    for (const [name, value] of Object.entries(row.dims)) {
      const pool = dims.get(name) ?? [];
      if (!pool.includes(value)) pool.push(value);
      dims.set(name, pool);
    }
  }
  for (const pool of dims.values()) pool.sort((left, right) => left - right);

  const multiValued = (rows: KernelRow[]) => {
    const seen = new Map<string, Set<number>>();
    for (const row of rows) {
      for (const [name, value] of Object.entries(row.dims)) {
        const values = seen.get(name) ?? new Set<number>();
        values.add(value);
        seen.set(name, values);
      }
    }
    return new Set([...seen].filter(([, values]) => values.size > 1).map(([name]) => name));
  };
  const xOptions = [...dims.keys()].filter((name) => rowSets.every((rows) => multiValued(rows).has(name)));
  const argKeys = distinct(all.flatMap((row) => Object.keys(row.args)));
  const args: [string, string[]][] = argKeys
    .map((key): [string, string[]] => [
      key,
      distinct(all.filter((row) => key in row.args).map((row) => fmt(row.args[key]))).sort(),
    ])
    .filter(([, values]) => values.length > 1);
  const dimDefaults: Record<string, number> = {};
  for (const [name, pool] of dims) {
    if (pool.length > 1) {
      dimDefaults[name] = mode(all.filter((row) => name in row.dims).map((row) => row.dims[name])) ?? pool[0];
    }
  }
  const argDefaults: Record<string, string> = {};
  for (const [key, values] of args) {
    argDefaults[key] = mode(all.filter((row) => key in row.args).map((row) => fmt(row.args[key]))) ?? values[0];
  }
  return {
    xOptions,
    dims: [...dims.entries()],
    args,
    fixed: [...dims.entries()]
      .filter(([, pool]) => pool.length === 1)
      .map(([name, pool]): [string, number] => [name, pool[0]]),
    dimDefaults,
    argDefaults,
    dtypes: distinct(all.map((row) => row.dtype)).sort(),
    devices: distinct(all.map((row) => row.device)).sort(),
    presents: distinct(all.map(presentKey)).sort(),
    batches: distinct(all.map(batchKey)).sort((left, right) =>
      left.localeCompare(right, undefined, { numeric: true }),
    ),
  };
}

export function defaultSliceSelection(kernels: Kernel[], controls: SliceControls): SliceSelection {
  const all = kernels.flatMap(rowsForKernel);
  const seqish = ['seq', 'total', 'tokens', 'response', 'hidden'].find((name) => controls.xOptions.includes(name));
  const selection: SliceSelection = {
    x: seqish ?? controls.xOptions[0] ?? '',
    metric: 'latency',
    pass: 'forward',
    dtype: mode(all.map((row) => row.dtype)) ?? controls.dtypes[0] ?? '',
    device: mode(all.map((row) => row.device)) ?? controls.devices[0] ?? '',
    present: mode(all.map(presentKey)) ?? controls.presents[0] ?? 'none',
    batch: mode(all.map(batchKey)) ?? controls.batches[0] ?? 'none',
    dims: {},
    args: {},
  };
  const preferred = selectionForSliceAxis(kernels, selection, selection.x, selection.dtype);
  const choices = measuredSlices(all, preferred);
  const closest = closestMeasuredSlice(choices, preferred);
  return closest ? selectionFromMeasuredSlice(preferred, closest) : preferred;
}

export function selectionForSliceAxis(
  kernels: Kernel[],
  selection: SliceSelection,
  axis: string,
  dtype = selection.dtype,
): SliceSelection {
  const shared = curveOptions(kernels)
    .filter((option) => option.axis === axis && option.dtype === dtype)
    .map((option) => curveFor(kernels[0], option.key))
    .filter((curve): curve is KernelCurve => curve !== undefined);
  const candidates =
    shared.length > 0
      ? shared
      : (kernels[0]?.evidence.curves.filter((curve) => curve.axis === axis && curve.dtype === dtype) ?? []);
  const curve = [...candidates].sort((left, right) => {
    const score = (candidate: KernelCurve) => [
      candidate.profile === 'production' ? 1 : 0,
      candidate.variant === 'base' ? 1 : 0,
      candidate.case_ids.length,
    ];
    const leftScore = score(left);
    const rightScore = score(right);
    for (let index = 0; index < leftScore.length; index++) {
      if (leftScore[index] !== rightScore[index]) return rightScore[index] - leftScore[index];
    }
    return left.id.localeCompare(right.id);
  })[0];
  if (!curve) return { ...selection, x: axis, dtype };
  return {
    ...selection,
    x: axis,
    dtype,
    batch: curve.fixed.batch === null ? selection.batch : curve.fixed.batch.join('x') || 'none',
    present: curve.fixed.present.join('+') || 'none',
    dims: { ...curve.fixed.dims },
    args: Object.fromEntries(Object.entries(curve.fixed.args).map(([name, value]) => [name, fmt(value)])),
  };
}

function measuredSliceKey(
  dims: Record<string, number>,
  batch: string,
  args: Record<string, string>,
  present: string,
): string {
  return JSON.stringify([
    Object.entries(dims).sort(([left], [right]) => left.localeCompare(right)),
    batch,
    Object.entries(args).sort(([left], [right]) => left.localeCompare(right)),
    present,
  ]);
}

export function measuredSlices(rows: KernelRow[], selection: SliceSelection): MeasuredSlice[] {
  const metric = METRICS.find((candidate) => candidate.id === selection.metric);
  if (!metric || !selection.x) return [];
  const groups = new Map<
    string,
    Omit<MeasuredSlice, 'points' | 'results'> & { xs: Set<number>; resultKeys: Set<string> }
  >();
  for (const row of rows) {
    const x = row.dims[selection.x];
    if (
      x === undefined ||
      row.status !== 'pass' ||
      row.dtype !== selection.dtype ||
      row.device !== selection.device ||
      (selection.pass !== 'forward' && !row.grad) ||
      metric.value(row, selection.pass) === null
    ) {
      continue;
    }
    const dims = Object.fromEntries(
      Object.entries(row.dims)
        .filter(([name]) => name !== selection.x)
        .sort(([left], [right]) => left.localeCompare(right)),
    );
    const args = Object.fromEntries(
      Object.entries(row.args)
        .map(([name, value]): [string, string] => [name, fmt(value)])
        .sort(([left], [right]) => left.localeCompare(right)),
    );
    const argValues = Object.fromEntries(
      Object.entries(row.args).sort(([left], [right]) => left.localeCompare(right)),
    );
    const batch = batchKey(row);
    const present = presentKey(row);
    const key = measuredSliceKey(dims, batch, args, present);
    const group = groups.get(key) ?? {
      key,
      dims,
      batch,
      batchValues: [...row.batch],
      args,
      argValues,
      present,
      presentValues: [...row.present],
      xs: new Set<number>(),
      resultKeys: new Set<string>(),
    };
    group.xs.add(x);
    group.resultKeys.add(`${row.case_id}|${row.impl}`);
    groups.set(key, group);
  }
  return [...groups.values()]
    .map(({ xs, resultKeys, ...slice }) => ({
      ...slice,
      points: xs.size,
      results: resultKeys.size,
    }))
    .sort(
      (left, right) =>
        right.points - left.points || right.results - left.results || left.key.localeCompare(right.key),
    );
}

function numericDistance(left: number, right: number): number {
  if (left === right) return 0;
  if (left > 0 && right > 0) return Math.abs(Math.log2(left / right));
  return Math.abs(left - right);
}

function anchorMatches(slice: MeasuredSlice, anchor: SliceAnchor): boolean {
  if (anchor.kind === 'dim') return slice.dims[anchor.name] === anchor.value;
  if (anchor.kind === 'arg') return slice.args[anchor.name] === anchor.value;
  return slice[anchor.kind] === anchor.value;
}

function sliceDistance(slice: MeasuredSlice, selection: SliceSelection, anchor?: SliceAnchor): number {
  let distance = 0;
  for (const [name, value] of Object.entries(slice.dims)) {
    if (anchor?.kind === 'dim' && anchor.name === name) continue;
    const current = selection.dims[name];
    if (current !== undefined) distance += numericDistance(value, current);
  }
  for (const [name, value] of Object.entries(slice.args)) {
    if (anchor?.kind === 'arg' && anchor.name === name) continue;
    const current = selection.args[name];
    if (current !== undefined && current !== value) distance += 1;
  }
  if (anchor?.kind !== 'batch' && slice.batch !== selection.batch) distance += 1;
  if (anchor?.kind !== 'present' && slice.present !== selection.present) distance += 1;
  return distance;
}

export function closestMeasuredSlice(
  slices: MeasuredSlice[],
  selection: SliceSelection,
  anchor?: SliceAnchor,
): MeasuredSlice | undefined {
  const candidates = anchor ? slices.filter((slice) => anchorMatches(slice, anchor)) : slices;
  return [...candidates].sort((left, right) => {
    const distance = sliceDistance(left, selection, anchor) - sliceDistance(right, selection, anchor);
    return distance || right.points - left.points || right.results - left.results || left.key.localeCompare(right.key);
  })[0];
}

export function selectionFromMeasuredSlice(
  selection: SliceSelection,
  slice: MeasuredSlice,
): SliceSelection {
  return {
    ...selection,
    dims: { ...slice.dims },
    batch: slice.batch,
    args: { ...slice.args },
    present: slice.present,
  };
}

export function measuredSliceLabel(slice: MeasuredSlice): string {
  const context = [
    ...(slice.batch === 'none' ? [] : [`batch=${slice.batch}`]),
    ...Object.entries(slice.dims).map(([name, value]) => `${name}=${value}`),
    ...Object.entries(slice.args).map(([name, value]) => `${name}=${value}`),
    ...(slice.present === 'none' ? [] : [`+${slice.present.replaceAll('+', ',+')}`]),
  ];
  return `${slice.points} ${slice.points === 1 ? 'point' : 'points'} · ${context.join(' · ') || 'default context'}`;
}

function shellQuote(value: string): string {
  if (/^[A-Za-z0-9_@%+=:,./-]+$/.test(value)) return value;
  return `'${value.replaceAll("'", `'"'"'`)}'`;
}

export function sliceFillCommand(kernel: string, axis: string, dtype: string, slice: MeasuredSlice): string {
  const context = [
    `dtype=${dtype}`,
    ...(slice.batchValues.length > 0 ? [`...=${JSON.stringify(slice.batchValues)}`] : []),
    ...Object.entries(slice.dims)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([name, value]) => `${name}=${JSON.stringify(value)}`),
    ...Object.entries(slice.argValues)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([name, value]) => `${name}=${JSON.stringify(value)}`),
    ...[...slice.presentValues].sort().map((name) => `+${name}`),
  ];
  return ['popcorn', 'bench', 'fill', kernel, '--live', '--timeout', '300', '--slice', axis, ...context]
    .map(shellQuote)
    .join(' ');
}

function snap(target: number, pool: number[]): number {
  let best = pool[0];
  for (const value of pool) {
    if (Math.abs(Math.log(value) - Math.log(target)) < Math.abs(Math.log(best) - Math.log(target))) {
      best = value;
    }
  }
  return best;
}

export function filterSliceRows(
  rows: KernelRow[],
  selection: SliceSelection,
  controls: SliceControls,
): KernelRow[] {
  const pools = new Map<string, number[]>();
  for (const row of rows) {
    for (const [name, value] of Object.entries(row.dims)) {
      const pool = pools.get(name) ?? [];
      if (!pool.includes(value)) pool.push(value);
      pools.set(name, pool);
    }
  }
  const wanted = new Map<string, number>();
  for (const [name, pool] of pools) {
    if (name === selection.x || pool.length === 1) continue;
    wanted.set(name, snap(selection.dims[name] ?? controls.dimDefaults[name] ?? pool[0], pool));
  }
  const requireGrad = selection.pass !== 'forward';
  const candidates = rows.filter(
    (row) =>
      row.status === 'pass' &&
      row.dtype === selection.dtype &&
      (!selection.device || row.device === selection.device) &&
      presentKey(row) === selection.present &&
      batchKey(row) === selection.batch &&
      (!requireGrad || row.grad) &&
      [...wanted].every(([name, value]) => row.dims[name] === value) &&
      Object.entries(row.args).every(
        ([key, value]) => (selection.args[key] ?? controls.argDefaults[key] ?? fmt(value)) === fmt(value),
      ),
  );
  const chosen = new Map<string, KernelRow>();
  for (const candidate of candidates) {
    const key = `${candidate.case_id}|${candidate.impl}|${candidate.device}`;
    const current = chosen.get(key);
    if (!current || (selection.pass === 'forward' ? !candidate.grad && current.grad : candidate.grad && !current.grad)) {
      chosen.set(key, candidate);
    }
  }
  return [...chosen.values()];
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
