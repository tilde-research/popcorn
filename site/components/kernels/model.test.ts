import assert from 'node:assert/strict';
import test from 'node:test';
import type { Kernel, KernelCase, KernelCurve, KernelResult } from '../../lib/kernels.js';
import {
  comparisonMode,
  curveKey,
  curveOptions,
  defaultCurveKey,
  normalizedSelection,
  rowsForCurve,
  xValue,
} from './model.ts';

const result = (impl: string, grad = false): KernelResult => ({
  impl,
  device: 'NVIDIA H100',
  grad,
  status: 'pass',
  benchmarked: true,
  fwd_ms: 1,
  ...(grad ? { bwd_ms: 2 } : {}),
});

const case_ = (id: string, seq: number, qHeads = 8, kvHeads = 2, results = [result('fast')]): KernelCase => ({
  case_id: id,
  dtype: 'float32',
  dims: { seq, q_heads: qHeads, kv_heads: kvHeads },
  batch: [1],
  args: { causal: false },
  present: [],
  results,
});

const curve = (fixedHidden = 256): KernelCurve => ({
  id: 'curve:production:seq:float32:base',
  profile: 'production',
  axis: 'seq',
  dtype: 'float32',
  variant: 'base',
  fixed: {
    dims: { hidden: fixedHidden, q_heads: 8, kv_heads: 2 },
    batch: [1],
    args: { causal: false },
    present: [],
  },
  case_ids: ['a', 'b', 'c'],
});

function kernel(name: string, fixedHidden = 256): Kernel {
  const cases = {
    a: case_('a', 128),
    b: case_('b', 512),
    c: case_('c', 2048, 8, 2, [result('fast', true)]),
  };
  const selected = curve(fixedHidden);
  return {
    name,
    summary: null,
    math: null,
    citations: [],
    tags: [],
    params: [],
    impls: [],
    evidence: {
      cases,
      curves: [selected],
      samples: [],
      frontiers: [],
      coverage: {
        planned_samples: 0,
        planned_curve_cases: 3,
        observed_cases: 3,
        results: 3,
        timed: 3,
        statuses: { pass: 3 },
        by_impl: {},
      },
      freshness: {
        latest: '2026-08-03T00:00:00+00:00',
        current_results: 3,
        stale_results_omitted: 0,
        ref_hash: null,
        impl_hashes: {},
      },
    },
  };
}

test('curve controls expose only precomputed series', () => {
  const first = kernel('first');
  const second = kernel('second');
  assert.deepEqual(curveOptions([first, second]).map((option) => option.key), [curveKey(first.evidence.curves[0])]);
  assert.equal(defaultCurveKey([first, second]), 'production|seq|float32|base');

  second.evidence.curves[0] = { ...second.evidence.curves[0], variant: 'causal', id: 'curve:production:seq:float32:causal' };
  assert.deepEqual(curveOptions([first, second]), []);
});

test('comparison offers only curves with a shared context and hardware', () => {
  const first = kernel('first', 256);
  const second = kernel('second', 4096);
  const key = curveKey(first.evidence.curves[0]);
  assert.deepEqual(curveOptions([first, second]), []);
  assert.equal(comparisonMode([first, second], key), 'separate');
  assert.equal(comparisonMode([first, kernel('same', 256)], key), 'shared');

  const otherHardware = kernel('other-hardware', 256);
  for (const case_ of Object.values(otherHardware.evidence.cases)) case_.results[0].device = 'NVIDIA A100';
  assert.deepEqual(curveOptions([first, otherHardware]), []);
});

test('curve rows preserve measured head combinations and pass choice', () => {
  const item = kernel('attention');
  const selected = item.evidence.curves[0];
  const forward = rowsForCurve(item, selected, 'NVIDIA H100', 'forward');
  assert.deepEqual(
    forward.map((row) => [row.dims.q_heads, row.dims.kv_heads]),
    [
      [8, 2],
      [8, 2],
      [8, 2],
    ],
  );
  assert.deepEqual(forward.map((row) => xValue(row, selected)), [128, 512, 2048]);
  assert.equal(rowsForCurve(item, selected, 'NVIDIA H100', 'backward').length, 1);
});

test('selection resets invalid metric, pass, device, and curve', () => {
  const item = kernel('one');
  const normalized = normalizedSelection([item], {
    mode: 'curves',
    curve: 'missing',
    metric: 'memory',
    pass: 'backward',
    device: 'missing',
  });
  assert.equal(normalized.curve, 'production|seq|float32|base');
  assert.equal(normalized.metric, 'latency');
  assert.equal(normalized.device, 'NVIDIA H100');
  assert.equal(normalized.pass, 'backward');
});
