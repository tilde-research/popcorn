'use client';
import type { Kernel, KernelRow } from '@/lib/kernels';
import dynamic from 'next/dynamic';
import { useMemo, useState } from 'react';
import {
  DASHES,
  METRICS,
  PASSES,
  backendColor,
  combine,
  controlsFor,
  defaultSelection,
  filterRows,
  fmt,
  mode,
  tokens,
  type Pass,
  type Selection,
} from './model';

// plotly.js touches `self` at import time, so it must never load during prerender.
const PlotFrame = dynamic(() => import('./plot-frame').then((m) => m.PlotFrame), {
  ssr: false,
  loading: () => <div className="h-full min-h-64 w-full animate-pulse rounded-lg bg-fd-muted/50" />,
});

function Segmented({
  options,
  value,
  onChange,
  disabled = [],
  mono = false,
}: {
  options: string[];
  value: string;
  onChange: (next: string) => void;
  disabled?: string[];
  mono?: boolean;
}) {
  return (
    <div className={`flex overflow-hidden rounded-lg border text-xs ${mono ? 'font-mono' : ''}`}>
      {options.map((option) => (
        <button
          key={option}
          disabled={disabled.includes(option)}
          onClick={() => onChange(option)}
          className={`px-2.5 py-1 transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${
            option === value ? 'bg-fd-primary text-fd-primary-foreground' : 'bg-fd-card hover:bg-fd-accent'
          }`}
        >
          {option}
        </button>
      ))}
    </div>
  );
}

function Control({ label, children, mono = true }: { label: string; children: React.ReactNode; mono?: boolean }) {
  return (
    <label className="flex items-center gap-2">
      <span className={`text-xs text-fd-muted-foreground ${mono ? 'font-mono' : ''}`}>{label}</span>
      {children}
    </label>
  );
}

const median = (values: number[]): number => {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
};

export function PlotPane({ kernels }: { kernels: Kernel[] }) {
  const controls = useMemo(() => controlsFor(kernels), [kernels]);
  const [overrides, setOverrides] = useState<Partial<Selection>>({});
  const patch = (next: Partial<Selection>) => setOverrides((current) => ({ ...current, ...next }));

  const rows = kernels.flatMap((kernel) => kernel.rows);
  if (kernels.length === 0) return <Empty message="Select a kernel." />;
  if (rows.length === 0) return <Empty message="No recorded benchmarks yet." />;
  if (controls.xOptions.length === 0)
    return <Empty message="The plotted kernels share no sweepable dimension." />;

  // Overrides survive kernel switches; anything invalid for the current
  // kernel set silently falls back to the defaults.
  const defaults = defaultSelection(kernels, controls);
  const valid = <T,>(value: T | undefined, pool: T[]): T | undefined =>
    value !== undefined && pool.includes(value) ? value : undefined;
  const metrics = METRICS.filter((metric) => metric.available(rows));
  const hasBwd = rows.some((row) => row.bwd_ms !== undefined);
  const selection: Selection = {
    ...defaults,
    ...overrides,
    x: valid(overrides.x, controls.xOptions) ?? defaults.x,
    metric: valid(overrides.metric, metrics.map((m) => m.id)) ?? defaults.metric,
    pass: (valid(overrides.pass, hasBwd ? PASSES : ['forward']) ?? defaults.pass) as Pass,
    dtype: valid(overrides.dtype, controls.dtypes) ?? defaults.dtype,
    device: valid(overrides.device, controls.devices) ?? defaults.device,
    present: valid(overrides.present, controls.presents) ?? defaults.present,
    batch: valid(overrides.batch, controls.batches) ?? defaults.batch,
    dims: overrides.dims ?? {},
    args: overrides.args ?? {},
  };
  const metric = metrics.find((m) => m.id === selection.metric)!;

  // torch reference values ride along on every row as ref_*; error metrics
  // have no reference series (the reference is the yardstick itself).
  const referenceValue = (row: KernelRow): number | null => {
    if (selection.metric === 'memory') return combine(selection.pass, row.ref_fwd_mem_mb, row.ref_bwd_mem_mb, 'max');
    if (selection.metric !== 'latency' && selection.metric !== 'throughput') return null;
    const ms = combine(selection.pass, row.ref_fwd_ms, row.ref_bwd_ms, 'sum');
    if (ms === null) return null;
    return selection.metric === 'throughput' ? (tokens(row) * 1000) / ms : ms;
  };

  const multi = kernels.length > 1;
  const traces: unknown[] = [];
  const cutoffPoints: { x: number; y: number }[] = [];
  for (const [position, kernel] of kernels.entries()) {
    const filtered = filterRows(kernel, selection, controls);
    const dash = DASHES[position % DASHES.length];
    const label = (backend: string) => (multi ? `${kernel.name} · ${backend}` : backend);
    const series = [...new Set(filtered.map((row) => row.backend))].sort().map((backend) => ({
      backend,
      points: filtered
        .filter((row) => row.backend === backend)
        .map((row) => ({ row, x: row.dims[selection.x], y: metric.value(row, selection.pass) })),
    }));
    // One synthetic torch series per kernel (median across backends' ref values).
    const byX = new Map<number, number[]>();
    for (const row of filtered) {
      const value = referenceValue(row);
      if (value !== null) byX.set(row.dims[selection.x], [...(byX.get(row.dims[selection.x]) ?? []), value]);
    }
    if (byX.size > 0)
      series.push({
        backend: 'torch',
        points: [...byX.entries()].map(([x, values]) => ({ row: filtered[0], x, y: median(values) })),
      });

    for (const { backend, points } of series) {
      const kept = points
        .filter((point): point is { row: KernelRow; x: number; y: number } => point.y !== null)
        .sort((a, b) => a.x - b.x);
      if (kept.length === 0) continue;
      traces.push({
        name: label(backend),
        x: kept.map((point) => point.x),
        y: kept.map((point) => point.y),
        mode: 'lines+markers',
        line: { color: backendColor(backend), dash },
        marker: { size: 6 },
        hovertemplate: `${label(backend)}<br>${selection.x}=%{x}<br>%{y:.4g} ${metric.unit}<extra></extra>`,
      });
    }
    if (metric.cutoff)
      for (const row of filtered) {
        const cut = metric.cutoff(row, selection.pass);
        if (cut !== null) cutoffPoints.push({ x: row.dims[selection.x], y: cut });
      }
  }

  if (cutoffPoints.length > 0) {
    const byX = new Map<number, number[]>();
    for (const point of cutoffPoints) byX.set(point.x, [...(byX.get(point.x) ?? []), point.y]);
    const xs = [...byX.keys()].sort((a, b) => a - b);
    traces.push({
      name: 'pass cutoff',
      x: xs,
      y: xs.map((x) => median(byX.get(x)!)),
      mode: 'lines',
      line: { color: '#f43f5e', dash: 'dot', width: 1.5 },
      hovertemplate: `pass cutoff<br>${selection.x}=%{x}<br>%{y:.4g}<extra></extra>`,
    });
  }

  return (
    <div className="flex h-full flex-col gap-3">
      <div className="flex shrink-0 flex-wrap items-center gap-x-5 gap-y-2">
        <Control label="y" mono={false}>
          <Segmented options={metrics.map((m) => m.id)} value={selection.metric} onChange={(metric) => patch({ metric })} />
        </Control>
        <Control label="pass" mono={false}>
          <Segmented
            options={PASSES}
            disabled={hasBwd ? [] : ['backward', 'both']}
            value={selection.pass}
            onChange={(pass) => patch({ pass: pass as Pass })}
          />
        </Control>
        {controls.dtypes.length > 1 && (
          <Control label="dtype" mono={false}>
            <Segmented options={controls.dtypes} value={selection.dtype} onChange={(dtype) => patch({ dtype })} mono />
          </Control>
        )}
        {controls.devices.length > 1 && (
          <Control label="device" mono={false}>
            <Segmented options={controls.devices} value={selection.device} onChange={(device) => patch({ device })} />
          </Control>
        )}
      </div>

      <div className="min-h-56 min-w-0 flex-1 overflow-hidden">
        {traces.length > 0 ? (
          <PlotFrame
            data={traces}
            layout={{
              xaxis: { title: { text: selection.x }, type: 'log', gridcolor: 'rgba(128,128,128,0.2)' },
              yaxis: { title: { text: metric.unit }, type: 'log', gridcolor: 'rgba(128,128,128,0.2)' },
              legend: { orientation: 'h', y: -0.28 },
            }}
          />
        ) : (
          <Empty message="No recorded rows at this selection — adjust the controls below." />
        )}
      </div>

      <div className="flex max-h-[45%] shrink-0 flex-col gap-2 overflow-y-auto border-t pt-3">
      <div className="flex flex-wrap items-center gap-x-5 gap-y-2">
        <Control label="x axis" mono={false}>
          <Segmented options={controls.xOptions} value={selection.x} onChange={(x) => patch({ x })} mono />
        </Control>
        {controls.args.map(([key, values]) => (
          <Control key={key} label={key}>
            <Segmented
              options={values}
              value={selection.args[key] ?? mode(rows.filter((row) => key in row.args).map((row) => fmt(row.args[key])))}
              onChange={(value) => patch({ args: { ...selection.args, [key]: value } })}
              mono
            />
          </Control>
        ))}
        {controls.presents.length > 1 && (
          <Control label="optionals" mono={false}>
            <Segmented options={controls.presents} value={selection.present} onChange={(present) => patch({ present })} mono />
          </Control>
        )}
        {controls.batches.length > 1 && (
          <Control label="batch" mono={false}>
            <Segmented options={controls.batches} value={selection.batch} onChange={(batch) => patch({ batch })} mono />
          </Control>
        )}
      </div>
      <div className="grid grid-cols-1 gap-x-8 gap-y-1.5 sm:grid-cols-2">
        {controls.dims
          .filter(([, values]) => values.length > 1)
          .map(([name, values]) => {
            const onAxis = name === selection.x;
            const current = selection.dims[name] ?? mode(values);
            const index = Math.max(0, values.indexOf(current));
            return (
              <label key={name} className={`flex items-center gap-3 text-xs ${onAxis ? 'opacity-40' : ''}`}>
                <span className="w-28 shrink-0 truncate font-mono text-fd-muted-foreground">{name}</span>
                <input
                  type="range"
                  disabled={onAxis}
                  min={0}
                  max={values.length - 1}
                  value={index}
                  onChange={(e) => patch({ dims: { ...selection.dims, [name]: values[Number(e.target.value)] } })}
                  className="flex-1 accent-fd-primary disabled:cursor-not-allowed"
                />
                <span className="w-14 shrink-0 text-right font-mono">{onAxis ? '—' : current}</span>
              </label>
            );
          })}
      </div>
      </div>
    </div>
  );
}

function Empty({ message }: { message: string }) {
  return <p className="flex h-full items-center justify-center py-8 text-sm text-fd-muted-foreground">{message}</p>;
}
