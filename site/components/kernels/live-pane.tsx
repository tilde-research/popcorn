'use client';

import {
  DEFAULT_LIVE_PORT,
  LIVE_PROTOCOL,
  localKernelState,
  validLivePort,
  type LocalKernelState,
  type LiveProbe,
  type LiveRecord,
} from '@/lib/live-bench';
import type { FormEvent } from 'react';
import type { LiveBenchController, StreamState } from './use-live-bench';

function timing(record: LiveRecord, direction: 'fwd' | 'bwd'): string {
  const value = record.bench[`${direction}_ms`];
  if (value === undefined) return '—';
  const reference = record.bench[`ref_${direction}_ms`];
  const speedup = reference !== undefined && value > 0 ? ` · ${(reference / value).toFixed(2)}×` : '';
  return `${value.toFixed(3)} ms${speedup}`;
}

function statusClass(status: string): string {
  if (status === 'pass') return 'text-emerald-600 dark:text-emerald-400';
  if (status === 'skip') return 'text-fd-muted-foreground';
  if (status === 'oom' || status === 'timeout') return 'text-amber-600 dark:text-amber-400';
  return 'text-red-600 dark:text-red-400';
}

function connectionText(probe: LiveProbe | null, state: StreamState, port: string): string {
  if (state === 'checking') return `Checking 127.0.0.1:${port}…`;
  if (probe?.kind === 'blocked' || probe?.kind === 'offline' || probe?.kind === 'occupied') return probe.detail;
  if (probe?.kind === 'incompatible') {
    return `Popcorn is running, but its live protocol is ${probe.status.protocol}; this site expects ${LIVE_PROTOCOL}.`;
  }
  if (probe?.kind === 'popcorn') {
    if (state === 'opening') return `Popcorn ${probe.status.version} found; opening the live stream…`;
    if (state === 'reconnecting') return `Popcorn ${probe.status.version} found; the stream is reconnecting…`;
    if (state === 'complete' && probe.status.state === 'error') {
      return `Popcorn ${probe.status.version} ended this live session with an error.`;
    }
    if (state === 'complete') return `Popcorn ${probe.status.version} completed this live session.`;
    return `Connected to Popcorn ${probe.status.version}.`;
  }
  if (state === 'idle') return 'Not checked. Connecting may ask for browser permission to access loopback.';
  return 'Could not identify the service on this port.';
}

export function LocalStatusDot({ state }: { state: LocalKernelState }) {
  if (state === 'running') {
    return (
      <span
        className="size-3 shrink-0 animate-spin rounded-full border-2 border-emerald-500 border-t-transparent"
        aria-hidden="true"
      />
    );
  }
  const color =
    state === 'success'
      ? 'bg-emerald-500'
      : state === 'partial'
        ? 'bg-orange-500'
        : state === 'failed'
          ? 'bg-red-500'
          : 'bg-fd-muted-foreground/40';
  return <span className={`size-2.5 shrink-0 rounded-full ${color}`} aria-hidden="true" />;
}

export function LocalBenchmarkMark({
  kernel,
  live,
  expanded,
  onClick,
}: {
  kernel: string;
  live: LiveBenchController;
  expanded: boolean;
  onClick: () => void;
}) {
  const rows = live.rowsByOp[kernel] ?? [];
  const status = live.probe?.kind === 'popcorn' ? live.status : null;
  const state = localKernelState(status, rows, kernel);
  const text =
    state === 'running'
      ? 'local benchmark running'
      : state === 'none'
        ? 'no local tests detected'
        : 'local benchmark completed';

  return (
    <button
      type="button"
      aria-expanded={expanded}
      onClick={onClick}
      className="inline-flex items-center gap-2 rounded-full border bg-fd-card px-2.5 py-1 text-xs transition-colors hover:bg-fd-accent"
    >
      <LocalStatusDot state={state} />
      <span>{text}</span>
    </button>
  );
}

export function LiveMenu({ kernel, live }: { kernel: string; live: LiveBenchController }) {
  const { portText, probe, streamState, status } = live;
  const rows = live.rowsByOp[kernel] ?? [];
  const port = validLivePort(portText);
  const commandPort = port ?? DEFAULT_LIVE_PORT;
  const command = `popcorn bench fill ${kernel} --live${
    commandPort === DEFAULT_LIVE_PORT ? '' : ` ${commandPort}`
  }`;
  const selectedIsRunning = status?.ops.includes(kernel) ?? false;
  const progress = status && status.total > 0 ? Math.min(100, (status.completed / status.total) * 100) : 0;
  const recent = [...rows].slice(-100).reverse();
  const check = (event: FormEvent) => {
    event.preventDefault();
    void live.check();
  };

  return (
    <div className="flex flex-col gap-3 text-sm">
      <form onSubmit={check} className="flex flex-wrap items-end gap-2">
        <label className="flex flex-col gap-1 text-xs text-fd-muted-foreground">
          Local port
          <input
            type="number"
            min={1}
            max={65535}
            inputMode="numeric"
            value={portText}
            onChange={(event) => live.changePort(event.target.value)}
            className="w-28 rounded-lg border bg-fd-card px-2.5 py-1.5 font-mono text-sm text-fd-foreground outline-none focus:ring-2 focus:ring-fd-ring"
          />
        </label>
        <button
          type="submit"
          disabled={streamState === 'checking'}
          className="rounded-lg border bg-fd-card px-3 py-1.5 text-xs font-medium transition-colors hover:bg-fd-accent disabled:cursor-wait disabled:opacity-60"
        >
          {streamState === 'checking' ? 'Checking…' : probe?.kind === 'popcorn' ? 'Reconnect' : 'Check'}
        </button>
        {live.portError && (
          <span role="alert" className="text-xs text-red-600 dark:text-red-400">
            {live.portError}
          </span>
        )}
      </form>

      <div className="rounded-lg border bg-fd-card p-3" aria-live="polite">
        <p className="text-xs">{connectionText(probe, streamState, portText)}</p>
        <code className="mt-2 block overflow-x-auto rounded bg-fd-muted px-2.5 py-2 text-[11px]">{command}</code>
        <p className="mt-1.5 text-[11px] text-fd-muted-foreground">
          The server exists only while this command is running. Add <code>--force</code> to remeasure cached cases.
        </p>
      </div>

      {status && probe?.kind === 'popcorn' && (
        <div className="flex flex-col gap-2 rounded-lg border p-3">
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs">
            <span className="font-mono">{status.device}</span>
            <span className="font-mono text-fd-muted-foreground" title={status.session}>
              session {status.session.slice(0, 8)}
            </span>
            <span className="text-fd-muted-foreground">
              {status.completed.toLocaleString()} / {status.total.toLocaleString()} considered
            </span>
            <span className="text-fd-muted-foreground">
              {status.measured.toLocaleString()} measured · {status.pruned.toLocaleString()} pruned
            </span>
            {(status.cached > 0 || status.skipped > 0) && (
              <span className="text-fd-muted-foreground">
                {status.cached.toLocaleString()} cached · {status.skipped.toLocaleString()} skipped
              </span>
            )}
          </div>
          <div className="h-1.5 overflow-hidden rounded-full bg-fd-muted">
            <div className="h-full bg-fd-primary transition-[width]" style={{ width: `${progress}%` }} />
          </div>
          {Object.keys(status.statuses).length > 0 && (
            <div className="flex flex-wrap gap-2 text-[11px]">
              {Object.entries(status.statuses).map(([name, count]) => (
                <span key={name} className={statusClass(name)}>
                  {name} {count.toLocaleString()}
                </span>
              ))}
            </div>
          )}
          {!selectedIsRunning && status.state === 'running' && (
            <p className="text-xs text-amber-600 dark:text-amber-400">
              This Popcorn session is running {status.ops.join(', ') || 'another kernel'}, not {kernel}.
            </p>
          )}
          {status.error && <p className="text-xs text-red-600 dark:text-red-400">{status.error}</p>}
        </div>
      )}

      {status && probe?.kind === 'popcorn' && selectedIsRunning && (
        <div className="overflow-x-auto rounded-lg border">
          {recent.length === 0 ? (
            <p className="p-4 text-center text-xs text-fd-muted-foreground">Waiting for {kernel} results…</p>
          ) : (
            <table className="w-full min-w-[42rem] text-xs">
              <thead>
                <tr className="border-b bg-fd-muted/40 text-left text-fd-muted-foreground">
                  <th className="px-2.5 py-2 font-medium">backend</th>
                  <th className="px-2.5 py-2 font-medium">status</th>
                  <th className="px-2.5 py-2 font-medium">dtype / case</th>
                  <th className="px-2.5 py-2 text-right font-medium">forward</th>
                  <th className="px-2.5 py-2 text-right font-medium">backward</th>
                </tr>
              </thead>
              <tbody>
                {recent.map(({ sequence, record }) => {
                  const detail = record.reason || (record.bench_error ? `benchmark: ${record.bench_error}` : '');
                  return (
                    <tr key={sequence} className="border-b last:border-0">
                      <td className="px-2.5 py-2 font-mono font-medium">{record.impl}</td>
                      <td className={`px-2.5 py-2 font-medium ${statusClass(record.status)}`}>{record.status}</td>
                      <td className="max-w-80 px-2.5 py-2">
                        <span className="font-mono">{record.config.dtype}</span>
                        <span className="ml-2 text-fd-muted-foreground" title={record.case}>
                          {record.case}
                        </span>
                        {detail && <div className="mt-0.5 text-[11px] text-red-600 dark:text-red-400">{detail}</div>}
                      </td>
                      <td className="whitespace-nowrap px-2.5 py-2 text-right font-mono">{timing(record, 'fwd')}</td>
                      <td className="whitespace-nowrap px-2.5 py-2 text-right font-mono">{timing(record, 'bwd')}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>
      )}
    </div>
  );
}
