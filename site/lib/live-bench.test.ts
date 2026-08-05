import assert from 'node:assert/strict';
import test from 'node:test';
import {
  DEFAULT_LIVE_PORT,
  LIVE_PROTOCOL,
  LIVE_SERVICE,
  appendLiveRow,
  liveEventsUrl,
  liveKernelRow,
  liveRecord,
  liveRowsForCurve,
  liveStatusUrl,
  localKernelState,
  probeLive,
  validLivePort,
  type LiveRecord,
  type LiveStatus,
} from './live-bench.ts';
import type { KernelCurve } from './kernels.ts';

const status = (protocol = LIVE_PROTOCOL): LiveStatus => ({
  service: LIVE_SERVICE,
  protocol,
  version: '0.1.0',
  session: 'session',
  state: 'running',
  command: 'fill',
  host: '127.0.0.1',
  port: DEFAULT_LIVE_PORT,
  device: 'Test GPU',
  ops: ['rms_norm'],
  total: 4,
  completed: 1,
  measured: 1,
  pruned: 0,
  cached: 2,
  skipped: 0,
  statuses: { pass: 1 },
  op_statuses: { rms_norm: { pass: 1 } },
  started_at: '2026-08-04T00:00:00+00:00',
  error: '',
});

const record: LiveRecord = {
  schema: 3,
  op: 'rms_norm',
  impl: 'fast',
  case: 'hidden=128,float32',
  case_id: 'case',
  config: { dtype: 'float32', dims: { hidden: 128 }, batch: [], args: {}, present: [] },
  device: 'Test GPU',
  status: 'pass',
  reason: '',
  grad: false,
  fwd: { out: { err: 0.1, scale: 2, budget: 0.01 } },
  bwd: {},
  bench: { fwd_ms: 1, ref_fwd_ms: 2 },
  benchmarked: true,
  bench_error: '',
  ts: '2026-08-04T00:00:00+00:00',
};

test('live ports and URLs stay on IPv4 loopback', () => {
  assert.equal(validLivePort('8765'), 8765);
  assert.equal(validLivePort('0'), null);
  assert.equal(validLivePort('65536'), null);
  assert.equal(validLivePort('not-a-port'), null);
  assert.equal(liveStatusUrl(9000), 'http://127.0.0.1:9000/v1/status');
  assert.equal(liveEventsUrl(9000), 'http://127.0.0.1:9000/v1/events');
  assert.equal(liveEventsUrl(9000, 'a/b'), 'http://127.0.0.1:9000/v1/events?op=a%2Fb');
});

test('probe distinguishes Popcorn, protocol mismatch, occupied port, and no listener', async () => {
  const popcorn = await probeLive(
    DEFAULT_LIVE_PORT,
    async () => new Response(JSON.stringify(status()), { headers: { 'Content-Type': 'application/json' } }),
  );
  assert.equal(popcorn.kind, 'popcorn');

  const incompatible = await probeLive(
    DEFAULT_LIVE_PORT,
    async () => new Response(JSON.stringify(status(99)), { headers: { 'Content-Type': 'application/json' } }),
  );
  assert.equal(incompatible.kind, 'incompatible');

  let calls = 0;
  const occupied = await probeLive(DEFAULT_LIVE_PORT, async () => {
    calls += 1;
    if (calls === 1) throw new TypeError('CORS blocked');
    return new Response(null, { status: 204 });
  });
  assert.equal(occupied.kind, 'occupied');

  const offline = await probeLive(DEFAULT_LIVE_PORT, async () => {
    throw new TypeError('connection refused');
  });
  assert.equal(offline.kind, 'offline');
});

test('live record parser and bounded reducer reject malformed or duplicate rows', () => {
  assert.deepEqual(liveRecord({ record }), record);
  assert.equal(liveRecord({ record: { op: 'missing fields' } }), null);

  const first = appendLiveRow([], { sequence: 1, record }, 2);
  assert.equal(appendLiveRow(first, { sequence: 1, record }, 2), first);
  const second = appendLiveRow(first, { sequence: 2, record: { ...record, case_id: 'second' } }, 2);
  const third = appendLiveRow(second, { sequence: 3, record: { ...record, case_id: 'third' } }, 2);
  assert.deepEqual(
    third.map((row) => row.sequence),
    [2, 3],
  );
});

test('kernel status distinguishes running, successful, partial, failed, and unrelated sessions', () => {
  const row = { sequence: 1, record };
  assert.equal(localKernelState(status(), [row], 'rms_norm'), 'running');
  assert.equal(
    localKernelState({ ...status(), state: 'complete', op_statuses: { rms_norm: { pass: 2 } } }, [row], 'rms_norm'),
    'success',
  );
  assert.equal(
    localKernelState(
      { ...status(), state: 'complete', op_statuses: { rms_norm: { pass: 1, fail: 1 } } },
      [row],
      'rms_norm',
    ),
    'partial',
  );
  assert.equal(
    localKernelState({ ...status(), state: 'complete', op_statuses: { rms_norm: { fail: 2 } } }, [row], 'rms_norm'),
    'failed',
  );
  assert.equal(
    localKernelState(
      { ...status(), state: 'complete', op_statuses: undefined },
      [{ sequence: 1, record: { ...record, bench_error: 'timer failed' } }],
      'rms_norm',
    ),
    'failed',
  );
  assert.equal(localKernelState(status(), [row], 'softmax'), 'none');
});

test('live rows join the matching published curve with plot metrics', () => {
  const curve: KernelCurve = {
    id: 'curve',
    profile: 'production',
    axis: 'hidden',
    dtype: 'float32',
    variant: 'base',
    fixed: { dims: {}, batch: [], args: {}, present: [] },
    case_ids: ['case'],
  };
  const converted = liveKernelRow(record);
  assert.equal(converted.fwd_ms, 1);
  assert.equal(converted.ref_fwd_ms, 2);
  assert.equal(converted.fwd_err, 0.1);
  assert.equal(converted.fwd_rel, 0.05);
  assert.deepEqual(liveRowsForCurve([{ sequence: 1, record }], curve), [converted]);
  assert.deepEqual(
    liveRowsForCurve([{ sequence: 1, record: { ...record, config: { ...record.config, dtype: 'float16' } } }], curve),
    [],
  );
});
