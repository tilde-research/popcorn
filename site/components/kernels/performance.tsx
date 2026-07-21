'use client';
import type { KernelRow } from '@/lib/kernels';
import dynamic from 'next/dynamic';
import { useMemo, useState } from 'react';

// plotly.js touches `self` at import time, so it must never load during prerender.
const PlotFrame = dynamic(() => import('./plot-frame').then((m) => m.PlotFrame), {
  ssr: false,
  loading: () => <div className="h-[420px] w-full animate-pulse rounded-lg bg-fd-muted/50" />,
});

const PALETTE = ['#6366f1', '#f59e0b', '#10b981', '#ef4444', '#06b6d4', '#d946ef', '#84cc16', '#f97316'];
const SEQ_DIMS = ['seq', 'total'];

function fmt(value: unknown): string {
  if (value === null) return 'None';
  if (value === true) return 'True';
  if (value === false) return 'False';
  return String(value);
}

function latency(row: KernelRow, grad: boolean): number {
  return grad ? row.fwd_ms + (row.bwd_ms ?? 0) : row.fwd_ms;
}

function mode<T>(values: T[]): T {
  const counts = new Map<T, number>();
  for (const value of values) counts.set(value, (counts.get(value) ?? 0) + 1);
  let best = values[0];
  for (const [value, count] of counts) if (count > (counts.get(best) ?? 0)) best = value;
  return best;
}

function Segmented({
  options,
  value,
  onChange,
}: {
  options: string[];
  value: string;
  onChange: (next: string) => void;
}) {
  return (
    <div className="flex overflow-hidden rounded-lg border text-xs">
      {options.map((option) => (
        <button
          key={option}
          onClick={() => onChange(option)}
          className={`px-2.5 py-1 transition-colors ${
            option === value ? 'bg-fd-primary text-fd-primary-foreground' : 'bg-fd-card hover:bg-fd-accent'
          }`}
        >
          {option}
        </button>
      ))}
    </div>
  );
}

export function PerformancePanel({ rows }: { rows: KernelRow[] }) {
  // Distinct values per control, derived once from the recorded rows.
  const facts = useMemo(() => {
    const dims = new Map<string, number[]>();
    for (const row of rows)
      for (const [name, value] of Object.entries(row.dims)) {
        if (!dims.has(name)) dims.set(name, []);
        const values = dims.get(name)!;
        if (!values.includes(value)) values.push(value);
      }
    for (const values of dims.values()) values.sort((a, b) => a - b);
    const distinct = (pick: (row: KernelRow) => string) => [...new Set(rows.map(pick))].sort();
    return {
      dims,
      xCandidates: [...dims.keys()].filter((name) => dims.get(name)!.length > 1),
      dtypes: distinct((row) => row.dtype),
      devices: distinct((row) => row.device),
      grads: [...new Set(rows.map((row) => row.grad))],
      argKeys: [...new Set(rows.flatMap((row) => Object.keys(row.args)))].filter(
        (key) => new Set(rows.map((row) => fmt(row.args[key]))).size > 1,
      ),
      argValues: (key: string) => [...new Set(rows.map((row) => fmt(row.args[key])))].sort(),
      presents: distinct((row) => row.present.join('+') || 'none'),
      batches: distinct((row) => row.batch.join('x') || 'none'),
      seqDim: [...dims.keys()].find((name) => SEQ_DIMS.includes(name)),
    };
  }, [rows]);

  const [x, setX] = useState(() => facts.seqDim ?? facts.xCandidates[0]);
  const [metric, setMetric] = useState<'latency' | 'throughput'>('latency');
  const [grad, setGrad] = useState(() => mode(rows.map((row) => row.grad)));
  const [dtype, setDtype] = useState(() => mode(rows.map((row) => row.dtype)));
  const [device, setDevice] = useState(() => mode(rows.map((row) => row.device)));
  const [argSel, setArgSel] = useState<Record<string, string>>({});
  const [presentSel, setPresentSel] = useState(() => mode(rows.map((row) => row.present.join('+') || 'none')));
  const [batchSel, setBatchSel] = useState(() => mode(rows.map((row) => row.batch.join('x') || 'none')));
  const [dimSel, setDimSel] = useState<Record<string, number>>({});

  if (rows.length === 0)
    return <p className="py-8 text-center text-sm text-fd-muted-foreground">No recorded benchmarks yet.</p>;

  const argValue = (key: string) => argSel[key] ?? mode(rows.map((row) => fmt(row.args[key])));
  const dimValue = (name: string) => dimSel[name] ?? mode(rows.filter((row) => row.dtype === dtype).map((row) => row.dims[name]));
  const sliderDims = facts.xCandidates.filter((name) => name !== x);

  const filtered = rows.filter(
    (row) =>
      row.dtype === dtype &&
      row.device === device &&
      row.grad === grad &&
      (row.present.join('+') || 'none') === presentSel &&
      (row.batch.join('x') || 'none') === batchSel &&
      facts.argKeys.every((key) => fmt(row.args[key]) === argValue(key)) &&
      sliderDims.every((name) => row.dims[name] === dimValue(name)),
  );

  const backends = [...new Set(filtered.map((row) => row.backend))].sort();
  const unit = metric === 'throughput' ? 'ms/pos' : 'ms';
  const perPosition = (row: KernelRow, ms: number) =>
    metric === 'throughput' && facts.seqDim ? ms / row.dims[facts.seqDim] : ms;
  const traces: unknown[] = backends.map((backend, index) => {
    const points = filtered.filter((row) => row.backend === backend).sort((a, b) => a.dims[x] - b.dims[x]);
    return {
      name: backend,
      x: points.map((row) => row.dims[x]),
      y: points.map((row) => perPosition(row, latency(row, grad))),
      mode: 'lines+markers',
      line: { color: PALETTE[index % PALETTE.length] },
      marker: { size: 6 },
      hovertemplate: `${backend}<br>${x}=%{x}<br>%{y:.4g} ${unit}<extra></extra>`,
    };
  });

  // The reference has no rows of its own; every row carries the reference
  // timing measured alongside it. Median across backends per x value.
  const refByX = new Map<number, number[]>();
  for (const row of filtered) {
    const ms = grad ? (row.ref_fwd_ms ?? NaN) + (row.ref_bwd_ms ?? NaN) : (row.ref_fwd_ms ?? NaN);
    if (Number.isNaN(ms)) continue;
    if (!refByX.has(row.dims[x])) refByX.set(row.dims[x], []);
    refByX.get(row.dims[x])!.push(perPosition(row, ms));
  }
  if (refByX.size > 0) {
    const xs = [...refByX.keys()].sort((a, b) => a - b);
    traces.push({
      name: 'torch (reference)',
      x: xs,
      y: xs.map((value) => {
        const sorted = refByX.get(value)!.sort((a, b) => a - b);
        return sorted[Math.floor(sorted.length / 2)];
      }),
      mode: 'lines+markers',
      line: { color: '#9ca3af', dash: 'dash' },
      marker: { size: 6 },
      hovertemplate: `torch (reference)<br>${x}=%{x}<br>%{y:.4g} ${unit}<extra></extra>`,
    });
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center gap-x-5 gap-y-3 text-xs">
        <label className="flex items-center gap-2">
          <span className="text-fd-muted-foreground">x axis</span>
          <Segmented options={facts.xCandidates} value={x} onChange={setX} />
        </label>
        <label className="flex items-center gap-2">
          <span className="text-fd-muted-foreground">y</span>
          <Segmented
            options={facts.seqDim ? ['latency', 'throughput'] : ['latency']}
            value={metric}
            onChange={(next) => setMetric(next as 'latency' | 'throughput')}
          />
        </label>
        {facts.grads.length > 1 && (
          <label className="flex items-center gap-2">
            <span className="text-fd-muted-foreground">pass</span>
            <Segmented
              options={['forward', 'forward+backward']}
              value={grad ? 'forward+backward' : 'forward'}
              onChange={(next) => setGrad(next === 'forward+backward')}
            />
          </label>
        )}
        {facts.dtypes.length > 1 && (
          <label className="flex items-center gap-2">
            <span className="text-fd-muted-foreground">dtype</span>
            <Segmented options={facts.dtypes} value={dtype} onChange={setDtype} />
          </label>
        )}
        {facts.devices.length > 1 && (
          <label className="flex items-center gap-2">
            <span className="text-fd-muted-foreground">device</span>
            <Segmented options={facts.devices} value={device} onChange={setDevice} />
          </label>
        )}
        {facts.presents.length > 1 && (
          <label className="flex items-center gap-2">
            <span className="text-fd-muted-foreground">optional inputs</span>
            <Segmented options={facts.presents} value={presentSel} onChange={setPresentSel} />
          </label>
        )}
        {facts.batches.length > 1 && (
          <label className="flex items-center gap-2">
            <span className="text-fd-muted-foreground">batch</span>
            <Segmented options={facts.batches} value={batchSel} onChange={setBatchSel} />
          </label>
        )}
        {facts.argKeys.map((key) => (
          <label key={key} className="flex items-center gap-2">
            <span className="font-mono text-fd-muted-foreground">{key}</span>
            <Segmented options={facts.argValues(key)} value={argValue(key)} onChange={(next) => setArgSel((s) => ({ ...s, [key]: next }))} />
          </label>
        ))}
      </div>
      {sliderDims.length > 0 && (
        <div className="grid grid-cols-1 gap-x-8 gap-y-2 sm:grid-cols-2">
          {sliderDims.map((name) => {
            const values = facts.dims.get(name)!;
            const current = dimValue(name);
            const index = Math.max(0, values.indexOf(current));
            return (
              <label key={name} className="flex items-center gap-3 text-xs">
                <span className="w-32 shrink-0 truncate font-mono text-fd-muted-foreground">{name}</span>
                <input
                  type="range"
                  min={0}
                  max={values.length - 1}
                  value={index}
                  onChange={(e) => setDimSel((s) => ({ ...s, [name]: values[Number(e.target.value)] }))}
                  className="flex-1 accent-fd-primary"
                />
                <span className="w-14 shrink-0 text-right font-mono">{current}</span>
              </label>
            );
          })}
        </div>
      )}
      {filtered.length === 0 ? (
        <p className="py-8 text-center text-sm text-fd-muted-foreground">
          No recorded rows at this exact selection — move a slider or switch dtype.
        </p>
      ) : (
        <PlotFrame
          data={traces}
          layout={{
            xaxis: { title: { text: x }, type: 'log', gridcolor: 'rgba(128,128,128,0.2)' },
            yaxis: {
              title: { text: metric === 'throughput' ? 'ms / position' : 'ms' },
              type: 'log',
              gridcolor: 'rgba(128,128,128,0.2)',
            },
            legend: { orientation: 'h', y: 1.08 },
          }}
        />
      )}
      <p className="text-right text-xs text-fd-muted-foreground">
        {filtered.length} of {rows.length} recorded rows at this selection
      </p>
    </div>
  );
}
