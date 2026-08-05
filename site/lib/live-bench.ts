import type { KernelCurve, KernelRow } from './kernels';

export const DEFAULT_LIVE_PORT = 8765;
export const LIVE_PROTOCOL = 1;
export const LIVE_SERVICE = 'popcorn.bench.live';
export const LIVE_PORT_STORAGE = 'popcorn.live.port';

export interface LiveStatus {
  service: typeof LIVE_SERVICE;
  protocol: number;
  version: string;
  session: string;
  state: 'running' | 'complete' | 'error';
  command: string;
  host: string;
  port: number;
  device: string;
  ops: string[];
  total: number;
  completed: number;
  measured: number;
  pruned: number;
  cached: number;
  skipped: number;
  statuses: Record<string, number>;
  op_statuses?: Record<string, Record<string, number>>;
  started_at: string;
  error: string;
}

export interface LiveGauge {
  err: number;
  scale: number;
  budget: number;
}

export interface LiveRecord {
  schema: number;
  op: string;
  impl: string;
  case: string;
  case_id: string;
  config: {
    dtype: string;
    dims: Record<string, number>;
    batch: number[];
    args: Record<string, unknown>;
    present: string[];
  };
  device: string;
  status: string;
  reason: string;
  grad: boolean;
  fwd: Record<string, LiveGauge>;
  bwd: Record<string, LiveGauge>;
  bench: Record<string, number>;
  benchmarked: boolean;
  bench_error: string;
  ts: string;
}

export interface LiveRow {
  sequence: number;
  record: LiveRecord;
}

export type LiveProbe =
  | { kind: 'popcorn'; status: LiveStatus }
  | { kind: 'incompatible'; status: LiveStatus }
  | { kind: 'occupied'; detail: string }
  | { kind: 'blocked'; detail: string }
  | { kind: 'offline'; detail: string };

type LoopbackRequestInit = RequestInit & { targetAddressSpace: 'loopback' };
type Fetcher = (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>;

export function validLivePort(value: string | number): number | null {
  const port = typeof value === 'number' ? value : Number(value);
  return Number.isInteger(port) && port >= 1 && port <= 65535 ? port : null;
}

export function loadLivePort(): number {
  if (typeof localStorage === 'undefined') return DEFAULT_LIVE_PORT;
  try {
    return validLivePort(localStorage.getItem(LIVE_PORT_STORAGE) ?? '') ?? DEFAULT_LIVE_PORT;
  } catch {
    return DEFAULT_LIVE_PORT;
  }
}

export function saveLivePort(port: number): void {
  if (typeof localStorage === 'undefined') return;
  try {
    localStorage.setItem(LIVE_PORT_STORAGE, String(port));
  } catch {
    // Storage may be disabled; the connection itself can still work.
  }
}

export function liveBaseUrl(port: number): string {
  return `http://127.0.0.1:${port}`;
}

export function liveStatusUrl(port: number): string {
  return `${liveBaseUrl(port)}/v1/status`;
}

export function liveEventsUrl(port: number, op?: string): string {
  return `${liveBaseUrl(port)}/v1/events${op === undefined ? '' : `?op=${encodeURIComponent(op)}`}`;
}

function object(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function numeric(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

export function liveStatus(value: unknown): LiveStatus | null {
  if (
    !object(value) ||
    value.service !== LIVE_SERVICE ||
    typeof value.protocol !== 'number' ||
    typeof value.version !== 'string' ||
    typeof value.session !== 'string' ||
    !['running', 'complete', 'error'].includes(String(value.state)) ||
    typeof value.command !== 'string' ||
    typeof value.host !== 'string' ||
    !numeric(value.port) ||
    typeof value.device !== 'string' ||
    !Array.isArray(value.ops) ||
    !value.ops.every((op) => typeof op === 'string') ||
    !numeric(value.total) ||
    !numeric(value.completed) ||
    !numeric(value.measured) ||
    !numeric(value.pruned) ||
    !numeric(value.cached) ||
    !numeric(value.skipped) ||
    !object(value.statuses) ||
    (value.op_statuses !== undefined && !object(value.op_statuses)) ||
    typeof value.started_at !== 'string' ||
    typeof value.error !== 'string'
  ) {
    return null;
  }
  return value as unknown as LiveStatus;
}

export function liveRecord(value: unknown): LiveRecord | null {
  const envelope = object(value) && object(value.record) ? value.record : null;
  if (
    !envelope ||
    typeof envelope.schema !== 'number' ||
    typeof envelope.op !== 'string' ||
    typeof envelope.impl !== 'string' ||
    typeof envelope.case !== 'string' ||
    typeof envelope.case_id !== 'string' ||
    !object(envelope.config) ||
    typeof envelope.config.dtype !== 'string' ||
    typeof envelope.device !== 'string' ||
    typeof envelope.status !== 'string' ||
    typeof envelope.reason !== 'string' ||
    typeof envelope.grad !== 'boolean' ||
    !object(envelope.fwd) ||
    !object(envelope.bwd) ||
    !object(envelope.bench) ||
    typeof envelope.benchmarked !== 'boolean' ||
    typeof envelope.bench_error !== 'string' ||
    typeof envelope.ts !== 'string'
  ) {
    return null;
  }
  return envelope as unknown as LiveRecord;
}

export function appendLiveRow(rows: LiveRow[], row: LiveRow, limit = 100): LiveRow[] {
  if (rows.some((candidate) => candidate.sequence === row.sequence)) return rows;
  return [...rows, row].slice(-limit);
}

export type LocalKernelState = 'none' | 'running' | 'success' | 'partial' | 'failed';

export function localKernelState(status: LiveStatus | null, rows: LiveRow[], kernel: string): LocalKernelState {
  if (status === null || !status.ops.includes(kernel)) return 'none';
  if (status.state === 'running') return 'running';
  const counts =
    status.op_statuses?.[kernel] ??
    rows.reduce<Record<string, number>>((all, row) => {
      const outcome = row.record.bench_error ? 'bench_error' : row.record.status;
      all[outcome] = (all[outcome] ?? 0) + 1;
      return all;
    }, {});
  const passed = counts.pass ?? 0;
  let failed = Object.entries(counts)
    .filter(([name]) => name !== 'pass')
    .reduce((total, [, count]) => total + count, 0);
  if (status.state === 'error') failed += 1;
  if (failed === 0) return 'success';
  if (passed === 0) return 'failed';
  return 'partial';
}

const FLOORS: Record<string, number> = {
  float64: 1e-10,
  float32: 2e-5,
  float16: 1e-3,
  bfloat16: 2e-2,
};

function gaugeMetrics(gauges: Record<string, LiveGauge>, dtype: string) {
  const values = Object.values(gauges).filter(
    (gauge) =>
      Number.isFinite(gauge.err) &&
      Number.isFinite(gauge.scale) &&
      Number.isFinite(gauge.budget) &&
      gauge.scale > 0,
  );
  if (values.length === 0) return null;
  const floor = FLOORS[dtype] ?? 2e-5;
  return {
    err: Math.max(...values.map((gauge) => gauge.err)),
    cut: Math.max(...values.map((gauge) => Math.max(2 * gauge.budget, floor * gauge.scale))),
    rel: Math.max(...values.map((gauge) => gauge.err / gauge.scale)),
    relCut: Math.max(
      ...values.map((gauge) => Math.max(2 * gauge.budget, floor * gauge.scale) / gauge.scale),
    ),
  };
}

export function liveKernelRow(record: LiveRecord): KernelRow {
  const bench = Object.fromEntries(
    Object.entries(record.bench).filter(([, value]) => typeof value === 'number' && Number.isFinite(value)),
  );
  const fwd = gaugeMetrics(record.fwd, record.config.dtype);
  const bwd = gaugeMetrics(record.bwd, record.config.dtype);
  return {
    case_id: record.case_id,
    dtype: record.config.dtype,
    dims: record.config.dims,
    batch: record.config.batch,
    args: record.config.args,
    present: record.config.present,
    impl: record.impl,
    device: record.device,
    grad: record.grad,
    status: record.status,
    benchmarked: record.benchmarked,
    ...(record.bench_error ? { bench_error: true as const } : {}),
    ...bench,
    ...(fwd
      ? { fwd_err: fwd.err, fwd_cut: fwd.cut, fwd_rel: fwd.rel, fwd_rel_cut: fwd.relCut }
      : {}),
    ...(bwd
      ? { bwd_err: bwd.err, bwd_cut: bwd.cut, bwd_rel: bwd.rel, bwd_rel_cut: bwd.relCut }
      : {}),
  };
}

function sorted(value: Record<string, unknown>): string {
  return JSON.stringify(Object.entries(value).sort(([left], [right]) => left.localeCompare(right)));
}

export function liveRecordMatchesCurve(record: LiveRecord, curve: KernelCurve): boolean {
  const config = record.config;
  if (config.dtype !== curve.dtype) return false;
  if (curve.case_ids.includes(record.case_id)) return true;
  if (curve.axis === '...') {
    if (config.batch.length === 0) return false;
  } else if (config.dims[curve.axis] === undefined) {
    return false;
  }
  const expectedDims = new Set(Object.keys(curve.fixed.dims));
  if (curve.axis !== '...') expectedDims.add(curve.axis);
  if (
    Object.keys(config.dims).some((name) => !expectedDims.has(name)) ||
    Object.entries(curve.fixed.dims).some(([name, value]) => config.dims[name] !== value)
  ) {
    return false;
  }
  if (
    curve.fixed.batch !== null &&
    (curve.fixed.batch.length !== config.batch.length ||
      curve.fixed.batch.some((value, index) => config.batch[index] !== value))
  ) {
    return false;
  }
  if (sorted(config.args) !== sorted(curve.fixed.args)) return false;
  return [...config.present].sort().join('|') === [...curve.fixed.present].sort().join('|');
}

export function liveRowsForCurve(
  rows: LiveRow[],
  curve: KernelCurve,
  pass: 'forward' | 'backward' | 'both' = 'forward',
): KernelRow[] {
  const requireGrad = pass !== 'forward';
  const candidates = rows
    .map((item) => item.record)
    .filter(
      (record) =>
        record.status === 'pass' &&
        (!requireGrad || record.grad) &&
        liveRecordMatchesCurve(record, curve),
    )
    .map(liveKernelRow);
  const chosen = new Map<string, KernelRow>();
  for (const candidate of candidates) {
    const key = `${candidate.case_id}|${candidate.impl}|${candidate.device}`;
    const current = chosen.get(key);
    if (
      !current ||
      current.grad === candidate.grad ||
      (pass === 'forward' ? !candidate.grad && current.grad : candidate.grad && !current.grad)
    ) {
      chosen.set(key, candidate);
    }
  }
  return [...chosen.values()];
}

async function timedFetch(fetcher: Fetcher, url: string, init: LoopbackRequestInit, timeoutMs: number): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetcher(url, { ...init, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

async function timedJson(fetcher: Fetcher, url: string, init: LoopbackRequestInit, timeoutMs: number): Promise<unknown> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetcher(url, { ...init, signal: controller.signal });
    return await response.json();
  } finally {
    clearTimeout(timer);
  }
}

async function loopbackPermission(): Promise<PermissionState | null> {
  if (typeof navigator === 'undefined' || !navigator.permissions) return null;
  try {
    const status = await navigator.permissions.query({ name: 'loopback-network' as PermissionName });
    return status.state;
  } catch {
    try {
      const status = await navigator.permissions.query({ name: 'local-network-access' as PermissionName });
      return status.state;
    } catch {
      return null;
    }
  }
}

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export async function probeLive(port: number, fetcher: Fetcher = fetch, timeoutMs = 1500): Promise<LiveProbe> {
  const url = liveStatusUrl(port);
  let verifiedError = '';
  try {
    const payload = await timedJson(
      fetcher,
      url,
      { method: 'GET', mode: 'cors', cache: 'no-store', targetAddressSpace: 'loopback' },
      timeoutMs,
    );
    const status = liveStatus(payload);
    if (status) {
      return status.protocol === LIVE_PROTOCOL ? { kind: 'popcorn', status } : { kind: 'incompatible', status };
    }
    return { kind: 'occupied', detail: `Port ${port} responded without a Popcorn live marker.` };
  } catch (error) {
    verifiedError = errorText(error);
  }

  try {
    await timedFetch(
      fetcher,
      url,
      { method: 'GET', mode: 'no-cors', cache: 'no-store', targetAddressSpace: 'loopback' },
      timeoutMs,
    );
    return { kind: 'occupied', detail: `Port ${port} responds, but its service could not be verified as Popcorn.` };
  } catch {
    const permission = await loopbackPermission();
    if (permission === 'denied') {
      return { kind: 'blocked', detail: 'Loopback access is blocked in this site’s browser permissions.' };
    }
    return {
      kind: 'offline',
      detail: `Nothing responded on 127.0.0.1:${port}${verifiedError ? ` (${verifiedError})` : ''}.`,
    };
  }
}
