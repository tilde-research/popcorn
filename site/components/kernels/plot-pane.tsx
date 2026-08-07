'use client';

import type { Kernel, KernelCurve, KernelRow } from '@/lib/kernels';
import { liveKernelRow, liveRowsForCurve, type LiveRow } from '@/lib/live-bench';
import dynamic from 'next/dynamic';
import { useEffect, useMemo, useRef, useState } from 'react';
import {
  DASHES,
  METRICS,
  PASSES,
  backendColor,
  caseLabel,
  closestMeasuredSlice,
  comparisonMode,
  contextLabel,
  curveFor,
  curveOptions,
  curveStats,
  defaultSliceSelection,
  defaultSelection,
  devicesFor,
  filterSliceRows,
  frontiersFor,
  hardwareLabel,
  measuredSliceLabel,
  measuredSlices,
  normalizedSelection,
  rowsForKernel,
  rowsForCurve,
  rowsForSamples,
  selectionFromMeasuredSlice,
  sliceControlsFor,
  sliceFillCommand,
  tokens,
  xValue,
  type Metric,
  type Pass,
  type Selection,
  type SliceAnchor,
  type SliceSelection,
} from './model';

const PlotFrame = dynamic(() => import('./plot-frame').then((module) => module.PlotFrame), {
  ssr: false,
  loading: () => <div className="h-full min-h-64 w-full animate-pulse rounded-lg bg-fd-muted/50" />,
});

function Segmented({
  options,
  value,
  onChange,
  disabled = [],
  labels = {},
}: {
  options: string[];
  value: string;
  onChange: (next: string) => void;
  disabled?: string[];
  labels?: Record<string, string>;
}) {
  return (
    <div className="flex max-w-full overflow-x-auto rounded-lg border text-xs">
      {options.map((option) => (
        <button
          type="button"
          key={option}
          disabled={disabled.includes(option)}
          aria-pressed={option === value}
          onClick={() => onChange(option)}
          className={`shrink-0 px-2.5 py-1 transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${
            option === value ? 'bg-fd-primary text-fd-primary-foreground' : 'bg-fd-card hover:bg-fd-accent'
          }`}
        >
          {labels[option] ?? option}
        </button>
      ))}
    </div>
  );
}

function Select({
  label,
  value,
  options,
  onChange,
  labels = {},
  wide = false,
}: {
  label: string;
  value: string;
  options: string[];
  onChange: (next: string) => void;
  labels?: Record<string, string>;
  wide?: boolean;
}) {
  if (options.length < 2) return null;
  return (
    <label className="flex min-w-0 items-center gap-2 text-xs">
      <span className="shrink-0 text-fd-muted-foreground">{label}</span>
      <select
        aria-label={label}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className={`min-w-0 rounded-lg border bg-fd-card px-2 py-1 font-mono outline-none focus:ring-2 focus:ring-fd-ring ${
          wide ? 'max-w-[32rem]' : 'max-w-52'
        }`}
      >
        {options.map((option) => (
          <option key={option} value={option}>
            {labels[option] ?? option}
          </option>
        ))}
      </select>
    </label>
  );
}

const median = (values: number[]): number => {
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.floor(sorted.length / 2)];
};

const passLabel = (pass: Pass): string =>
  ({ forward: 'forward', backward: 'backward', both: 'forward + backward' })[pass];

function referenceValue(row: KernelRow, metric: Metric, pass: Pass): number | null {
  const combine = (fwd: number | undefined, bwd: number | undefined, how: 'sum' | 'max') => {
    if (pass === 'forward') return fwd ?? null;
    if (pass === 'backward') return bwd ?? null;
    if (fwd === undefined || bwd === undefined) return null;
    return how === 'sum' ? fwd + bwd : Math.max(fwd, bwd);
  };
  if (metric.id === 'memory') return combine(row.ref_fwd_mem_mb, row.ref_bwd_mem_mb, 'max');
  if (metric.id !== 'latency' && metric.id !== 'throughput') return null;
  const ms = combine(row.ref_fwd_ms, row.ref_bwd_ms, 'sum');
  if (ms === null) return null;
  return metric.id === 'throughput' ? (tokens(row) * 1000) / ms : ms;
}

function CurveChart({
  items,
  selection,
  metric,
}: {
  items: {
    kernel: Kernel;
    curve: KernelCurve;
    rows: KernelRow[];
    liveRows: KernelRow[];
    liveRevision: number;
  }[];
  selection: { pass: Pass; device: string };
  metric: Metric;
}) {
  const traces: unknown[] = [];
  const yValues: number[] = [];
  const xValues: number[] = [];
  const multi = items.length > 1;

  for (const [position, item] of items.entries()) {
    const label = (impl: string) => (multi ? `${item.kernel.name} · ${impl}` : impl);
    const localLabel = (impl: string) =>
      multi ? `${item.kernel.name} · local · ${impl}` : `local · ${impl}`;
    const dash = DASHES[position % DASHES.length];
    const addImplementations = (rows: KernelRow[], local: boolean) => {
      const hasSyntheticReference = ['latency', 'throughput', 'memory'].includes(metric.id);
      const impls = [...new Set(rows.map((row) => row.impl))]
        .filter((impl) => impl !== 'torch' || !hasSyntheticReference)
        .sort();
      for (const impl of impls) {
        const points = rows
          .filter((row) => row.impl === impl)
          .map((row) => ({ x: xValue(row, item.curve), y: metric.value(row, selection.pass) }))
          .filter((point): point is { x: number; y: number } => point.y !== null)
          .sort((left, right) => left.x - right.x);
        if (points.length === 0) continue;
        const name = local ? localLabel(impl) : label(impl);
        xValues.push(...points.map((point) => point.x));
        yValues.push(...points.map((point) => point.y));
        traces.push({
          name,
          x: points.map((point) => point.x),
          y: points.map((point) => point.y),
          mode: 'lines+markers',
          line: { color: backendColor(impl), dash: local ? 'dot' : dash, width: local ? 3 : 2 },
          marker: local
            ? { size: 8, symbol: 'diamond', line: { color: 'rgba(255,255,255,0.85)', width: 1 } }
            : { size: 6 },
          ...(local
            ? {
                legendgroup: `local:${item.kernel.name}`,
                legendgrouptitle: { text: multi ? `Local · ${item.kernel.name}` : 'Local' },
              }
            : {}),
          hovertemplate: `${name}<br>${item.curve.axis}=%{x}<br>${passLabel(selection.pass)}: %{y:.4g} ${metric.unit}<extra></extra>`,
        });
      }
    };
    const addReference = (rows: KernelRow[], local: boolean) => {
      const references = new Map<number, number[]>();
      for (const row of rows) {
        const value = referenceValue(row, metric, selection.pass);
        const x = xValue(row, item.curve);
        if (value !== null) references.set(x, [...(references.get(x) ?? []), value]);
      }
      if (references.size === 0) return;
      const points = [...references].sort(([left], [right]) => left - right);
      const values = points.map(([, group]) => median(group));
      const name = local ? localLabel('torch') : label('torch');
      xValues.push(...points.map(([x]) => x));
      yValues.push(...values);
      traces.push({
        name,
        x: points.map(([x]) => x),
        y: values,
        mode: local ? 'lines+markers' : 'lines',
        line: { color: backendColor('torch'), dash: local ? 'dot' : dash, width: local ? 2.5 : 1.5 },
        ...(local
          ? {
              marker: { size: 7, symbol: 'diamond-open' },
              legendgroup: `local:${item.kernel.name}`,
              legendgrouptitle: { text: multi ? `Local · ${item.kernel.name}` : 'Local' },
            }
          : {}),
        hovertemplate: `${name}<br>${item.curve.axis}=%{x}<br>${passLabel(selection.pass)}: %{y:.4g} ${metric.unit}<extra></extra>`,
      });
    };

    addImplementations(item.rows, false);
    addReference(item.rows, false);
    addImplementations(item.liveRows, true);
    addReference(item.liveRows, true);
  }

  if (traces.length === 0) return <Empty message="This series has no timed passing points yet." />;
  const axis = items[0].curve.axis;
  const logX = xValues.length > 0 && xValues.every((value) => value > 0);
  const logY = yValues.length > 0 && yValues.every((value) => value > 0);
  const ticks = { dtick: 1, tickformat: '~s', showexponent: 'none' as const };
  return (
    <PlotFrame
      data={traces}
      layout={{
        datarevision: [
          selection.pass,
          selection.device,
          metric.id,
          ...items.map((item) => `${item.kernel.name}:${item.curve.id}:${item.liveRevision}`),
        ].join('|'),
        xaxis: {
          title: { text: axis === '...' ? 'batch' : axis },
          type: logX ? 'log' : 'linear',
          ...(logX ? ticks : {}),
          gridcolor: 'rgba(128,128,128,0.2)',
        },
        yaxis: {
          title: { text: `${passLabel(selection.pass)} ${metric.id} (${metric.unit})` },
          type: logY ? 'log' : 'linear',
          ...(logY ? ticks : { rangemode: 'tozero' }),
          gridcolor: 'rgba(128,128,128,0.2)',
        },
        legend: { orientation: 'h', y: -0.28 },
      }}
    />
  );
}

function CurveContext({
  kernel,
  curve,
  selection,
}: {
  kernel: Kernel;
  curve: KernelCurve;
  selection: Selection;
}) {
  const stats = curveStats(kernel, curve, selection.device, selection.pass);
  const frontiers = frontiersFor(kernel, curve, selection.device, selection.pass);
  return (
    <div className="rounded-lg border bg-fd-card/60 p-2.5 text-[11px]">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <strong className="font-mono text-fd-foreground">{kernel.name}</strong>
        <span className="text-fd-muted-foreground">{contextLabel(curve)}</span>
        <span className="ms-auto font-mono text-fd-muted-foreground">
          {stats.timed}/{stats.planned} timed{stats.max === null ? '' : ` · max ${stats.max}`}
        </span>
      </div>
      {frontiers.length > 0 && (
        <div className="mt-1 flex flex-wrap gap-x-3 gap-y-0.5 text-fd-muted-foreground">
          {frontiers.map((frontier) => (
            <span key={`${frontier.impl}|${frontier.grad}`}>
              <span className="font-mono text-fd-foreground">{frontier.impl}</span>:{' '}
              {frontier.pass_max === null ? 'no pass' : `passes through ${frontier.pass_max}`}
              {frontier.terminal ? `; ${frontier.terminal.status} at ${frontier.terminal.x}` : ''}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function SliceExplorer({
  kernels,
  liveRows,
}: {
  kernels: Kernel[];
  liveRows: Record<string, LiveRow[]>;
}) {
  const controls = useMemo(() => sliceControlsFor(kernels), [kernels]);
  const published = useMemo(
    () => new Map(kernels.map((kernel) => [kernel.name, rowsForKernel(kernel)])),
    [kernels],
  );
  const allRows = useMemo(() => [...published.values()].flat(), [published]);
  const defaults = useMemo(() => defaultSliceSelection(kernels, controls), [kernels, controls]);
  const [selection, setSelection] = useState<SliceSelection>(() => defaults);

  const valid = <T,>(value: T, pool: T[], fallback: T): T => (pool.includes(value) ? value : fallback);
  const tentative: SliceSelection = {
    ...defaults,
    ...selection,
    x: valid(selection.x, controls.xOptions, defaults.x),
    metric: valid(selection.metric, METRICS.map((metric) => metric.id), defaults.metric),
    pass: valid(selection.pass, PASSES, defaults.pass) as Pass,
    dtype: valid(selection.dtype, controls.dtypes, defaults.dtype),
    device: valid(selection.device, controls.devices, defaults.device),
  };
  const requestedSlices = measuredSlices(allRows, tentative);
  const slices = requestedSlices.length > 0 ? requestedSlices : measuredSlices(allRows, defaults);
  const basis = requestedSlices.length > 0 ? tentative : defaults;
  const currentSlice = closestMeasuredSlice(slices, basis);
  const selected = currentSlice ? selectionFromMeasuredSlice(basis, currentSlice) : basis;
  const metric = METRICS.find((candidate) => candidate.id === selected.metric) ?? METRICS[0];
  const hasSlices = (next: Partial<SliceSelection>) =>
    measuredSlices(allRows, { ...selected, ...next }).length > 0;
  const axisOptions = controls.xOptions.filter((x) => hasSlices({ x }));
  const dtypeOptions = controls.dtypes.filter((dtype) => hasSlices({ dtype }));
  const deviceOptions = controls.devices.filter((device) => hasSlices({ device }));
  const metricOptions = METRICS.filter((candidate) => hasSlices({ metric: candidate.id }));
  const passOptions = PASSES.filter((pass) => hasSlices({ pass }));
  const navigate = (next: Partial<SliceSelection>, anchor?: SliceAnchor) => {
    const desired = { ...selected, ...next };
    const destination = closestMeasuredSlice(measuredSlices(allRows, desired), desired, anchor);
    if (destination) setSelection(selectionFromMeasuredSlice(desired, destination));
  };

  const dimensionPools = new Map<string, number[]>();
  const argumentPools = new Map<string, string[]>();
  for (const slice of slices) {
    for (const [name, value] of Object.entries(slice.dims)) {
      const values = dimensionPools.get(name) ?? [];
      if (!values.includes(value)) values.push(value);
      dimensionPools.set(name, values);
    }
    for (const [name, value] of Object.entries(slice.args)) {
      const values = argumentPools.get(name) ?? [];
      if (!values.includes(value)) values.push(value);
      argumentPools.set(name, values);
    }
  }
  for (const values of dimensionPools.values()) values.sort((left, right) => left - right);
  for (const values of argumentPools.values()) {
    values.sort((left, right) => left.localeCompare(right, undefined, { numeric: true }));
  }
  const dimensionOrder = new Map(controls.dims.map(([name], index) => [name, index]));
  const dimensions = [...dimensionPools].sort(
    ([left], [right]) => (dimensionOrder.get(left) ?? 999) - (dimensionOrder.get(right) ?? 999),
  );
  const arguments_ = [...argumentPools];
  const batches = [...new Set(slices.map((slice) => slice.batch))].sort((left, right) =>
    left.localeCompare(right, undefined, { numeric: true }),
  );
  const presents = [...new Set(slices.map((slice) => slice.present))].sort();
  const fixed = [
    ...dimensions.filter(([, values]) => values.length === 1).map(([name, values]) => `${name}=${values[0]}`),
    ...arguments_.filter(([, values]) => values.length === 1).map(([name, values]) => `${name}=${values[0]}`),
    ...(batches.length === 1 && batches[0] !== 'none' ? [`batch=${batches[0]}`] : []),
    ...(presents.length === 1 && presents[0] !== 'none' ? [`+${presents[0].replaceAll('+', ',+')}`] : []),
  ];
  const curve: KernelCurve = {
    id: `slice:${selected.x}:${selected.dtype}:${JSON.stringify([
      selected.batch,
      selected.present,
      selected.dims,
      selected.args,
    ])}`,
    profile: 'slice',
    axis: selected.x,
    dtype: selected.dtype,
    variant: 'slice',
    fixed: { dims: {}, batch: null, args: {}, present: [] },
    case_ids: [],
  };
  const items = kernels.map((kernel) => {
    const streamed = liveRows[kernel.name] ?? [];
    return {
      kernel,
      curve,
      rows: filterSliceRows(published.get(kernel.name) ?? [], selected, controls),
      liveRows: filterSliceRows(
        streamed.map(({ record }) => liveKernelRow(record)),
        { ...selected, device: '' },
        controls,
      ),
      liveRevision: streamed[streamed.length - 1]?.sequence ?? 0,
    };
  });
  const fillCommands = currentSlice
    ? kernels.flatMap((kernel) => {
        const exact = measuredSlices(published.get(kernel.name) ?? [], selected).find(
          (slice) => slice.key === currentSlice.key,
        );
        const backends = [...new Set(kernel.impls.map((impl) => impl.name.split(':')[0]))].filter(
          (name) => name !== 'torch',
        );
        return exact
          ? [
              {
                kernel: kernel.name,
                command: sliceFillCommand(kernel.name, selected.x, selected.dtype, exact),
                requiredBackends: backends.includes('popcorn') ? [] : backends,
              },
            ]
          : [];
      })
    : [];

  if (allRows.length === 0) return <Empty message="No recorded benchmarks yet." />;
  if (controls.xOptions.length === 0) {
    return <Empty message="This selection has no measured dimension with enough values to form a slice." />;
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3">
      <div className="flex shrink-0 flex-wrap items-center gap-3">
        <Select
          label="x axis"
          value={selected.x}
          options={axisOptions}
          onChange={(x) => navigate({ x })}
        />
        <Select
          label="dtype"
          value={selected.dtype}
          options={dtypeOptions}
          onChange={(dtype) => navigate({ dtype })}
        />
        <Select
          label="hardware"
          value={selected.device}
          options={deviceOptions}
          labels={Object.fromEntries(deviceOptions.map((device) => [device, hardwareLabel(device)]))}
          onChange={(device) => navigate({ device })}
        />
        <Select
          label="metric"
          value={metric.id}
          options={metricOptions.map((candidate) => candidate.id)}
          onChange={(nextMetric) => navigate({ metric: nextMetric })}
        />
        <Segmented
          options={PASSES}
          disabled={PASSES.filter((pass) => !passOptions.includes(pass))}
          value={selected.pass}
          onChange={(pass) => navigate({ pass: pass as Pass })}
        />
      </div>

      <div className="min-h-64 min-w-0 flex-1">
        <CurveChart items={items} selection={selected} metric={metric} />
      </div>

      <div className="flex max-h-[45%] shrink-0 flex-col gap-3 overflow-y-auto border-t pt-3">
        <div className="flex flex-wrap items-center gap-3">
          <Select
            label={`all measured slices (${slices.length})`}
            value={currentSlice?.key ?? ''}
            options={slices.map((slice) => slice.key)}
            labels={Object.fromEntries(slices.map((slice) => [slice.key, measuredSliceLabel(slice)]))}
            onChange={(key) => {
              const slice = slices.find((candidate) => candidate.key === key);
              if (slice) setSelection(selectionFromMeasuredSlice(selected, slice));
            }}
            wide
          />
          {arguments_
            .filter(([, values]) => values.length > 1)
            .map(([key, values]) => (
              <Select
                key={key}
                label={key}
                value={selected.args[key]}
                options={values}
                onChange={(value) => navigate({}, { kind: 'arg', name: key, value })}
              />
            ))}
          <Select
            label="optionals"
            value={selected.present}
            options={presents}
            onChange={(value) => navigate({}, { kind: 'present', value })}
          />
          <Select
            label="batch"
            value={selected.batch}
            options={batches}
            onChange={(value) => navigate({}, { kind: 'batch', value })}
          />
          {fillCommands.length > 0 && <AddSliceData commands={fillCommands} />}
        </div>
        <p className="text-[11px] text-fd-muted-foreground">
          Every choice moves to the nearest compatible measured slice; the selector above lists all{' '}
          {slices.length}.
        </p>
        {fixed.length > 0 && (
          <p className="font-mono text-[11px] text-fd-muted-foreground">
            fixed: {fixed.join(' · ')}
          </p>
        )}
        <div className="grid grid-cols-1 gap-x-8 gap-y-1.5 sm:grid-cols-2">
          {dimensions
            .filter(([, values]) => values.length > 1)
            .map(([name, values]) => {
              const current = selected.dims[name];
              const index = Math.max(0, values.indexOf(current));
              return (
                <label key={name} className="flex items-center gap-3 text-xs">
                  <span className="w-28 shrink-0 truncate font-mono text-fd-muted-foreground">{name}</span>
                  <input
                    type="range"
                    min={0}
                    max={values.length - 1}
                    value={index}
                    onChange={(event) =>
                      navigate(
                        {},
                        {
                          kind: 'dim',
                          name,
                          value: values[Number(event.target.value)],
                        },
                      )
                    }
                    className="flex-1 accent-fd-primary"
                  />
                  <span className="w-14 shrink-0 text-right font-mono">{current}</span>
                </label>
              );
            })}
        </div>
      </div>
    </div>
  );
}

function Coverage({ kernels, device, onDevice }: { kernels: Kernel[]; device: string; onDevice: (value: string) => void }) {
  const devices = [...new Set(kernels.flatMap((kernel) => rowsForSamples(kernel).map((row) => row.device)))].sort();
  const selected = devices.includes(device) ? device : (devices[0] ?? '');
  const rows = kernels.flatMap((kernel) => rowsForSamples(kernel, selected).map((row) => ({ kernel: kernel.name, row })));
  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3">
      <div className="flex flex-wrap items-center gap-3">
        <p className="text-xs text-fd-muted-foreground">
          Pairwise coverage points are independent single shots. They are never connected into benchmark lines.
        </p>
        <div className="ms-auto">
          <Select
            label="hardware"
            value={selected}
            options={devices}
            labels={Object.fromEntries(devices.map((item) => [item, hardwareLabel(item)]))}
            onChange={onDevice}
          />
        </div>
      </div>
      <div className="min-h-0 flex-1 overflow-auto rounded-lg border">
        <table className="w-full min-w-[48rem] text-left text-xs">
          <thead className="sticky top-0 bg-fd-card text-fd-muted-foreground">
            <tr>
              <th className="px-3 py-2 font-medium">kernel</th>
              <th className="px-3 py-2 font-medium">case</th>
              <th className="px-3 py-2 font-medium">implementation</th>
              <th className="px-3 py-2 font-medium">status</th>
              <th className="px-3 py-2 text-right font-medium">forward</th>
            </tr>
          </thead>
          <tbody>
            {rows.slice(0, 500).map(({ kernel, row }) => (
              <tr key={`${kernel}|${row.case_id}|${row.impl}|${row.grad}`} className="border-t">
                <td className="px-3 py-2 font-mono">{kernel}</td>
                <td className="max-w-xl px-3 py-2 font-mono text-[11px] text-fd-muted-foreground">{caseLabel(row)}</td>
                <td className="px-3 py-2 font-mono">{row.impl}</td>
                <td className="px-3 py-2">{row.bench_error ? 'bench_error' : row.status}</td>
                <td className="px-3 py-2 text-right font-mono">{row.fwd_ms === undefined ? '·' : `${row.fwd_ms.toPrecision(4)} ms`}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {rows.length === 0 && <Empty message="No observed coverage samples on this hardware." />}
      </div>
      {rows.length > 500 && <p className="text-[11px] text-fd-muted-foreground">Showing 500 of {rows.length} results.</p>}
    </div>
  );
}

export function PlotPane({
  kernels,
  liveRows,
}: {
  kernels: Kernel[];
  liveRows: Record<string, LiveRow[]>;
}) {
  const kernelKey = kernels.map((kernel) => `${kernel.name}:${kernel.evidence.freshness.latest}`).join('|');
  const [selection, setSelection] = useState<Selection>(() => defaultSelection(kernels));

  useEffect(() => {
    setSelection(defaultSelection(kernels));
  }, [kernelKey]);

  const normalized = useMemo(() => normalizedSelection(kernels, selection), [kernels, selection]);
  useEffect(() => {
    if (JSON.stringify(normalized) !== JSON.stringify(selection)) setSelection(normalized);
  }, [normalized, selection]);

  if (kernels.length === 0) return <Empty message="Select a kernel." />;
  const options = curveOptions(kernels);
  const option = options.find((item) => item.key === normalized.curve);
  const selected = kernels
    .map((kernel) => ({ kernel, curve: curveFor(kernel, normalized.curve) }))
    .filter((item): item is { kernel: Kernel; curve: KernelCurve } => item.curve !== undefined);
  const chartItems = selected.map((item) => {
    const streamed = liveRows[item.kernel.name] ?? [];
    return {
      ...item,
      rows: rowsForCurve(item.kernel, item.curve, normalized.device, normalized.pass),
      liveRows: liveRowsForCurve(streamed, item.curve, normalized.pass),
      liveRevision: streamed[streamed.length - 1]?.sequence ?? 0,
    };
  });
  const allRows = chartItems.flatMap((item) => [...item.rows, ...item.liveRows]);
  const metrics = METRICS.filter((metric) => metric.available(allRows));
  const metric = metrics.find((item) => item.id === normalized.metric) ?? METRICS[0];
  const hasBackward = allRows.some((row) => row.bwd_ms !== undefined);
  const compare = comparisonMode(kernels, normalized.curve);

  const optionFields = ['profile', 'axis', 'dtype', 'variant'] as const;
  const choose = (field: (typeof optionFields)[number], value: string) => {
    const current = option;
    if (!current) return;
    const next = options.find(
      (candidate) =>
        candidate[field] === value &&
        optionFields.every((other) => other === field || candidate[other] === current[other]),
    );
    if (next) setSelection((state) => ({ ...state, curve: next.key, device: devicesFor(kernels, next.key)[0] ?? '' }));
  };

  const values = (field: (typeof optionFields)[number]) => {
    if (!option) return [];
    return [
      ...new Set(
        options
          .filter((candidate) =>
            optionFields.every((other) => other === field || candidate[other] === option[other]),
          )
          .map((candidate) => candidate[field]),
      ),
    ];
  };

  return (
    <div className="flex h-full min-h-0 flex-col gap-3">
      <div className="flex shrink-0 flex-wrap items-center gap-3">
        <Segmented
          options={['curves', 'slices', 'coverage']}
          value={normalized.mode}
          labels={{ curves: 'Curves', slices: 'Slices', coverage: 'Coverage' }}
          onChange={(mode) => setSelection((state) => ({ ...state, mode: mode as Selection['mode'] }))}
        />
        {normalized.mode === 'curves' && (
          <>
            <Select label="profile" value={option?.profile ?? ''} options={values('profile')} onChange={(value) => choose('profile', value)} />
            <Select label="axis" value={option?.axis ?? ''} options={values('axis')} onChange={(value) => choose('axis', value)} />
            <Select label="dtype" value={option?.dtype ?? ''} options={values('dtype')} onChange={(value) => choose('dtype', value)} />
            <Select
              label="context"
              value={option?.variant ?? ''}
              options={values('variant')}
              labels={Object.fromEntries(values('variant').map((variant) => [variant, variant === 'base' ? 'base' : variant.replaceAll('-', ' ')]))}
              onChange={(value) => choose('variant', value)}
            />
            <Select
              label="hardware"
              value={normalized.device}
              options={devicesFor(kernels, normalized.curve)}
              labels={Object.fromEntries(devicesFor(kernels, normalized.curve).map((device) => [device, hardwareLabel(device)]))}
              onChange={(device) => setSelection((state) => ({ ...state, device }))}
            />
            <Select
              label="metric"
              value={metric.id}
              options={metrics.map((item) => item.id)}
              onChange={(id) => setSelection((state) => ({ ...state, metric: id }))}
            />
            <Segmented
              options={hasBackward ? PASSES : ['forward']}
              value={normalized.pass}
              onChange={(pass) => setSelection((state) => ({ ...state, pass: pass as Pass }))}
            />
          </>
        )}
      </div>

      {normalized.mode === 'coverage' ? (
        <Coverage
          kernels={kernels}
          device={normalized.device}
          onDevice={(device) => setSelection((state) => ({ ...state, device }))}
        />
      ) : normalized.mode === 'slices' ? (
        // Live rows rerender this pane; remount slice state only when the published kernel selection changes.
        <SliceExplorer key={kernelKey} kernels={kernels} liveRows={liveRows} />
      ) : options.length === 0 ? (
        <Empty message="No curve has the same measured context and hardware across these kernels. Compare their coverage samples or select one kernel." />
      ) : (
        <>
          {kernels.length > 1 && compare === 'separate' && (
            <p className="rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-fd-muted-foreground">
              These kernels do not share the same fixed context. Their plots are separated so unlike shapes are not overlaid.
            </p>
          )}
          <div className={`min-h-64 min-w-0 flex-1 ${compare === 'separate' && selected.length > 1 ? 'grid gap-3 xl:grid-cols-2' : ''}`}>
            {compare === 'shared' ? (
              <CurveChart
                items={chartItems}
                selection={normalized}
                metric={metric}
              />
            ) : (
              chartItems.map((item) => (
                <div key={item.kernel.name} className="min-h-72 rounded-lg border p-2">
                  <p className="px-1 font-mono text-xs font-semibold">{item.kernel.name}</p>
                  <div className="h-[calc(100%-1.25rem)]">
                    <CurveChart items={[item]} selection={normalized} metric={metric} />
                  </div>
                </div>
              ))
            )}
          </div>
          <div className="grid shrink-0 gap-2 xl:grid-cols-2">
            {selected.map((item) => (
              <CurveContext key={item.kernel.name} {...item} selection={normalized} />
            ))}
          </div>
        </>
      )}
    </div>
  );
}

function AddSliceData({
  commands,
}: {
  commands: { kernel: string; command: string; requiredBackends: string[] }[];
}) {
  const [open, setOpen] = useState(false);
  const [copyState, setCopyState] = useState<'idle' | 'copied' | 'selected' | 'failed'>('idle');
  const panel = useRef<HTMLDivElement>(null);
  const commandBlock = useRef<HTMLDivElement>(null);
  const text = commands.map(({ command }) => command).join('\n');
  const requiredBackends = [...new Set(commands.flatMap(({ requiredBackends }) => requiredBackends))];

  useEffect(() => {
    setCopyState('idle');
  }, [text]);

  useEffect(() => {
    if (!open) return;
    const frame = requestAnimationFrame(() => panel.current?.scrollIntoView({ block: 'nearest' }));
    return () => cancelAnimationFrame(frame);
  }, [open, text]);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopyState('copied');
    } catch {
      const input = document.createElement('textarea');
      input.value = text;
      input.setAttribute('readonly', '');
      input.style.position = 'fixed';
      input.style.opacity = '0';
      document.body.append(input);
      input.select();
      let copied = false;
      try {
        copied = document.execCommand('copy');
      } catch {
        copied = false;
      } finally {
        input.remove();
      }
      if (copied) {
        setCopyState('copied');
        return;
      }
      const selection = window.getSelection();
      if (!selection || !commandBlock.current) {
        setCopyState('failed');
        return;
      }
      const range = document.createRange();
      range.selectNodeContents(commandBlock.current);
      selection.removeAllRanges();
      selection.addRange(range);
      setCopyState('selected');
    }
  };

  return (
    <>
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen((value) => !value)}
        className="rounded-lg border border-fd-primary/30 bg-fd-primary/10 px-3 py-1.5 text-xs font-medium text-fd-primary transition-colors hover:bg-fd-primary/15"
      >
        {open ? 'Hide command' : 'Add data'}
      </button>
      {open && (
        <div ref={panel} className="flex basis-full flex-col gap-2 rounded-lg border bg-fd-card p-3 text-xs">
          <div className="flex min-w-0 items-start gap-2">
            <div ref={commandBlock} className="min-w-0 flex-1">
              {commands.map(({ kernel, command }) => (
                <div key={kernel} className="min-w-0">
                  {commands.length > 1 && <p className="mb-1 font-mono text-fd-muted-foreground">{kernel}</p>}
                  <code className="block overflow-x-auto rounded-md bg-fd-muted/60 px-3 py-2 font-mono text-[11px] text-fd-foreground">
                    {command}
                  </code>
                </div>
              ))}
            </div>
            <button
              type="button"
              onClick={() => void copy()}
              className="shrink-0 rounded-md border px-2.5 py-1 font-medium transition-colors hover:bg-fd-accent"
            >
              {copyState === 'copied'
                ? 'Copied'
                : copyState === 'selected'
                  ? 'Selected — copy'
                  : copyState === 'failed'
                    ? 'Copy failed'
                    : 'Copy'}
            </button>
          </div>
          <p className="text-fd-muted-foreground">
            Replays fingerprint-matching cached rows immediately, then fills missing production points for this
            exact slice. <code className="font-mono text-fd-foreground">--live</code> adds each result here, and the
            longer timeout leaves room for first-time backend compilation.
          </p>
          {requiredBackends.length > 0 && (
            <p className="text-fd-muted-foreground">
              Requires one matching backend extra locally:{' '}
              <code className="font-mono text-fd-foreground">
                {requiredBackends.map((backend) => `popcorn[${backend}]`).join(' or ')}
              </code>
              .
            </p>
          )}
          <p aria-live="polite" className="sr-only">
            {copyState === 'copied'
              ? 'Command copied'
              : copyState === 'selected'
                ? 'Command selected; copy it manually'
                : copyState === 'failed'
                  ? 'Could not copy command'
                  : ''}
          </p>
        </div>
      )}
    </>
  );
}

function Empty({ message }: { message: string }) {
  return <p className="flex h-full min-h-32 items-center justify-center px-4 py-8 text-center text-sm text-fd-muted-foreground">{message}</p>;
}
