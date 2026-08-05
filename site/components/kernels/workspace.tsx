'use client';
import type { Kernel, KernelIndexEntry } from '@/lib/kernels';
import { localKernelState } from '@/lib/live-bench';
import { basePath, gitConfig } from '@/lib/shared';
import { useKernels } from '@/lib/use-kernels';
import { DynamicCodeBlock } from 'fumadocs-ui/components/dynamic-codeblock';
import { Tab, Tabs } from 'fumadocs-ui/components/tabs';
import katex from 'katex';
import { Download, FileCode, Pin } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import { LiveMenu, LocalBenchmarkMark, LocalStatusDot } from './live-pane';
import { MAX_PINNED } from './model';
import { PlotPane } from './plot-pane';
import { useLiveBench, type LiveBenchController } from './use-live-bench';

function usageSnippet(k: Kernel): string {
  const call = k.params.filter((p) => p.default === undefined).map((p) => p.name).join(', ');
  const forced = k.impls.find((b) => b.name !== 'torch')?.name;
  const lines = [`from popcorn.kernels import ${k.name}`, '', `out = ${k.name}(${call})  # auto-dispatch`];
  if (forced) lines.push(`out = ${k.name}(${call}, backend="${forced}")  # force a backend`);
  return lines.join('\n');
}

function benchmarkSnippet(k: Kernel): string {
  const call = k.params.filter((p) => p.default === undefined).map((p) => p.name).join(', ');
  return [
    `from popcorn.bench import report`,
    `from popcorn.kernels import ${k.name}`,
    '',
    `# time every eligible backend against the reference, then print one block each`,
    `for record in ${k.name}.benchmark(${call}):`,
    `    report(record)`,
    '',
    `# measure and route on first use instead: POPCORN_BENCH=1 python train.py`,
    `out = ${k.name}(${call}, bench=True)`,
  ].join('\n');
}

function localSnippet(k: Kernel): string {
  return [
    `python -m popcorn.bench fill ${k.name} --live`,
    '',
    `# use another port`,
    `python -m popcorn.bench fill ${k.name} --live 9000`,
  ].join('\n');
}

function InfoPane({ k, live }: { k: Kernel; live: LiveBenchController }) {
  const [localOpen, setLocalOpen] = useState(false);
  const math = k.math
    ? katex.renderToString(k.math, { displayMode: true, throwOnError: false, strict: false })
    : null;
  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center gap-3">
        <h1 className="font-mono text-xl font-bold">{k.name}</h1>
        <div className="flex flex-wrap gap-1">
          {k.tags.map((tag) => (
            <span key={tag} className="rounded-full bg-fd-muted px-2 py-0.5 text-[11px] text-fd-muted-foreground">
              {tag}
            </span>
          ))}
        </div>
        <LocalBenchmarkMark
          kernel={k.name}
          live={live}
          expanded={localOpen}
          onClick={() => setLocalOpen((open) => !open)}
        />
        <div className="ms-auto flex items-center gap-2 text-xs">
          <a
            href={`https://github.com/${gitConfig.user}/${gitConfig.repo}/blob/${gitConfig.branch}/src/popcorn/kernels/${k.name}.py`}
            aria-label="View source"
            className="flex items-center gap-1.5 rounded-lg border bg-fd-card p-2 transition-colors hover:bg-fd-accent sm:px-2.5 sm:py-1.5"
          >
            <FileCode className="size-3.5" /> <span className="max-sm:hidden">source</span>
          </a>
          <a
            href={`${basePath}/data/${k.name}.json`}
            download={`${k.name}.json`}
            aria-label="Download benchmark data"
            className="flex items-center gap-1.5 rounded-lg border bg-fd-card p-2 transition-colors hover:bg-fd-accent sm:px-2.5 sm:py-1.5"
          >
            <Download className="size-3.5" /> <span className="max-sm:hidden">data</span>
          </a>
        </div>
      </div>
      {localOpen && (
        <div className="rounded-xl border bg-fd-muted/20 p-3">
          <LiveMenu kernel={k.name} live={live} />
        </div>
      )}
      <p className="text-sm text-fd-muted-foreground">{k.summary}</p>
      {math && <div className="overflow-x-auto text-sm" dangerouslySetInnerHTML={{ __html: math }} />}
      {k.citations.length > 0 && (
        <p className="text-xs text-fd-muted-foreground">
          {k.citations.map((citation, index) => (
            <span key={citation.url}>
              {index > 0 && ', '}
              <a href={citation.url} className="underline hover:text-fd-foreground">
                {citation.label}
              </a>
            </span>
          ))}
        </p>
      )}
      <div className="grid grid-cols-1 gap-5 xl:grid-cols-2">
        <div className="overflow-x-auto">
          <table className="h-fit w-full min-w-[32rem] text-sm">
            <thead>
              <tr className="border-b text-left text-xs text-fd-muted-foreground">
                <th className="py-2 pr-4 font-medium">backend</th>
                <th className="py-2 pr-4 font-medium">source</th>
                <th className="py-2 font-medium">grad</th>
              </tr>
            </thead>
            <tbody>
              {k.impls.map((b) => (
                <tr key={b.name} className="border-b last:border-0">
                  <td className="py-2 pr-4 font-mono font-semibold">{b.name}</td>
                  <td className="whitespace-nowrap py-2 pr-4 font-mono text-xs text-fd-muted-foreground">
                    {b.source ?? 'reference'}
                  </td>
                  <td className="whitespace-nowrap py-2 text-xs text-fd-muted-foreground">
                    {b.forward_only ? 'forward-only' : 'forward + backward'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <Tabs items={['Use it', 'Benchmark it', 'Local']}>
          <Tab value="Use it">
            <DynamicCodeBlock lang="python" code={usageSnippet(k)} />
          </Tab>
          <Tab value="Benchmark it">
            <DynamicCodeBlock lang="python" code={benchmarkSnippet(k)} />
            <p className="mt-2 text-xs text-fd-muted-foreground">
              Each record names its backend; <code className="font-mono">report</code> prints
              latency, peak memory, and forward error, and returns the text.
            </p>
          </Tab>
          <Tab value="Local">
            <DynamicCodeBlock lang="bash" code={localSnippet(k)} />
            <p className="mt-2 text-xs text-fd-muted-foreground">
              Run the command locally, then open the local benchmark status beside the kernel name to connect,
              inspect progress, and review rows.
            </p>
          </Tab>
        </Tabs>
      </div>
    </div>
  );
}

export function Workspace({ entries }: { entries: KernelIndexEntry[] }) {
  const [query, setQuery] = useState('');
  const [activeTags, setActiveTags] = useState<string[]>([]);
  const [localOnly, setLocalOnly] = useState(false);
  const [pinned, setPinned] = useState<string[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const live = useLiveBench();
  const ready = useRef(false);

  const names = useMemo(() => new Set(entries.map((e) => e.name)), [entries]);

  // URL <-> state: ?k=pinned,csv&s=selected (replaceState keeps history clean).
  useEffect(() => {
    const params = new URLSearchParams(location.search);
    const k = (params.get('k') ?? '').split(',').filter((name) => names.has(name));
    const s = params.get('s');
    setPinned(k.slice(0, MAX_PINNED));
    setSelected(s && names.has(s) ? s : (k[0] ?? (names.has('attn') ? 'attn' : (entries[0]?.name ?? null))));
    ready.current = true;
  }, [entries, names]);

  useEffect(() => {
    if (!ready.current) return;
    const params = new URLSearchParams();
    if (pinned.length > 0) params.set('k', pinned.join(','));
    if (selected) params.set('s', selected);
    const search = params.toString();
    history.replaceState(null, '', search ? `?${search}` : location.pathname);
  }, [pinned, selected]);

  const togglePin = (name: string) =>
    setPinned((current) =>
      current.includes(name) ? current.filter((n) => n !== name) : [...current, name].slice(0, MAX_PINNED),
    );

  // Plotted = pinned (stable dash order) + the transient selection last.
  const plotted = selected && !pinned.includes(selected) ? [...pinned, selected] : pinned;
  const { kernels, loading, error } = useKernels(plotted);
  const info = kernels.find((kernel) => kernel.name === selected) ?? null;

  const tags = useMemo(() => [...new Set(entries.flatMap((e) => e.tags))].sort(), [entries]);
  const detectedStatus = live.probe?.kind === 'popcorn' ? live.status : null;
  const stateFor = (name: string) => localKernelState(detectedStatus, live.rowsByOp[name] ?? [], name);
  const q = query.toLowerCase();
  const listed = entries.filter(
    (e) =>
      !pinned.includes(e.name) &&
      (e.name.includes(q) || (e.summary ?? '').toLowerCase().includes(q)) &&
      activeTags.every((tag) => e.tags.includes(tag)) &&
      (!localOnly || stateFor(e.name) !== 'none'),
  );
  const pinnedEntries = pinned
    .map((name) => entries.find((e) => e.name === name))
    .filter((e): e is KernelIndexEntry => e !== undefined);
  const selectedEntry = entries.find((entry) => entry.name === selected);

  const Row = ({ entry, isPinned }: { entry: KernelIndexEntry; isPinned: boolean }) => {
    const localState = stateFor(entry.name);
    return (
      <div
        className={`group flex cursor-pointer flex-col gap-1.5 rounded-xl border p-3 transition-colors ${
          entry.name === selected
            ? 'border-fd-primary/50 bg-fd-accent text-fd-accent-foreground'
            : 'bg-fd-card hover:border-fd-primary/30'
        }`}
        onClick={() => setSelected(entry.name)}
      >
        <div className="flex items-center gap-2">
          <span className="truncate font-mono text-sm font-semibold">{entry.name}</span>
          {localState !== 'none' && (
            <span title={localState === 'running' ? 'Local benchmark running' : 'Local benchmark completed'}>
              <LocalStatusDot state={localState} />
            </span>
          )}
          <span className="ms-auto shrink-0 text-[11px] text-fd-muted-foreground">
            {entry.cases.toLocaleString()} cases
          </span>
          <button
            title={isPinned ? 'Unpin' : `Pin to compare (max ${MAX_PINNED})`}
            onClick={(event) => {
              event.stopPropagation();
              togglePin(entry.name);
            }}
            className={`shrink-0 rounded p-0.5 transition-opacity hover:text-fd-primary ${
              isPinned ? 'text-fd-primary' : 'text-fd-muted-foreground opacity-0 group-hover:opacity-100'
            }`}
          >
            <Pin className={`size-3.5 ${isPinned ? 'fill-current' : ''}`} />
          </button>
        </div>
        {entry.summary && <p className="line-clamp-2 text-xs text-fd-muted-foreground">{entry.summary}</p>}
        {entry.tags.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {entry.tags.map((tag) => (
              <span key={tag} className="rounded-full bg-fd-muted px-1.5 py-0.5 text-[10px] text-fd-muted-foreground">
                {tag}
              </span>
            ))}
          </div>
        )}
      </div>
    );
  };

  return (
    <main className="flex min-h-[calc(100dvh-3.5rem)] flex-col overflow-x-clip md:h-[calc(100dvh-3.5rem)] md:min-h-0 md:flex-row md:overflow-hidden">
      <div className="sticky top-14 z-20 flex items-center gap-2 border-b bg-fd-background/95 p-3 backdrop-blur md:hidden">
        <label htmlFor="kernel-select" className="sr-only">
          Kernel
        </label>
        <select
          id="kernel-select"
          value={selected ?? ''}
          onChange={(event) => setSelected(event.target.value)}
          className="min-w-0 flex-1 rounded-lg border bg-fd-card px-3 py-2 font-mono text-sm outline-none focus:ring-2 focus:ring-fd-ring"
        >
          <option value="" disabled>
            Select a kernel
          </option>
          {entries.map((entry) => (
            <option key={entry.name} value={entry.name}>
              {entry.name} · {entry.impls.length} implementation{entry.impls.length === 1 ? '' : 's'}
            </option>
          ))}
        </select>
        {selectedEntry ? (
          <button
            type="button"
            aria-label={pinned.includes(selectedEntry.name) ? 'Unpin kernel' : 'Pin kernel to compare'}
            title={pinned.includes(selectedEntry.name) ? 'Unpin' : `Pin to compare (max ${MAX_PINNED})`}
            onClick={() => togglePin(selectedEntry.name)}
            className={`inline-flex size-9 shrink-0 items-center justify-center rounded-lg border bg-fd-card ${
              pinned.includes(selectedEntry.name) ? 'text-fd-primary' : 'text-fd-muted-foreground'
            }`}
          >
            <Pin className={`size-4 ${pinned.includes(selectedEntry.name) ? 'fill-current' : ''}`} />
          </button>
        ) : null}
      </div>
      <aside className="hidden w-1/4 min-w-60 flex-col border-e md:flex">
        <div className="flex flex-col gap-2 border-b p-3">
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={`Search ${entries.length} kernels...`}
            className="w-full rounded-lg border bg-fd-card px-3 py-1.5 text-sm outline-none placeholder:text-fd-muted-foreground focus:ring-2 focus:ring-fd-ring"
          />
          <div className="flex flex-wrap gap-1">
            <button
              type="button"
              aria-pressed={localOnly}
              aria-label="Show only kernels with local benchmarks"
              onClick={() => setLocalOnly((active) => !active)}
              className={`inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[11px] transition-colors ${
                localOnly
                  ? 'border-fd-primary bg-fd-primary text-fd-primary-foreground'
                  : 'bg-fd-card text-fd-muted-foreground hover:text-fd-foreground'
              }`}
            >
              <span className="size-1.5 rounded-full bg-current" aria-hidden="true" />
              local
            </button>
            {tags.map((tag) => (
              <button
                key={tag}
                onClick={() =>
                  setActiveTags((a) => (a.includes(tag) ? a.filter((t) => t !== tag) : [...a, tag]))
                }
                className={`rounded-full border px-2 py-0.5 text-[11px] transition-colors ${
                  activeTags.includes(tag)
                    ? 'border-fd-primary bg-fd-primary text-fd-primary-foreground'
                    : 'bg-fd-card text-fd-muted-foreground hover:text-fd-foreground'
                }`}
              >
                {tag}
              </button>
            ))}
          </div>
        </div>
        <div className="flex flex-1 flex-col gap-2 overflow-y-auto p-2">
          {pinnedEntries.length > 0 && (
            <div className="flex flex-col gap-2 border-b pb-2">
              {pinnedEntries.map((entry) => (
                <Row key={entry.name} entry={entry} isPinned />
              ))}
            </div>
          )}
          {listed.map((entry) => (
            <Row key={entry.name} entry={entry} isPinned={false} />
          ))}
          {listed.length === 0 && pinnedEntries.length === 0 && (
            <p className="py-8 text-center text-sm text-fd-muted-foreground">No kernels match.</p>
          )}
        </div>
      </aside>
      <section className="flex min-w-0 flex-1 flex-col">
        <div className="border-b p-4 md:h-1/2 md:min-h-0 md:overflow-y-auto md:p-5">
          {error ? (
            <p className="flex h-full items-center justify-center text-sm text-red-500" role="alert">
              Could not load benchmark data: {error}
            </p>
          ) : loading ? (
            <p className="flex h-full items-center justify-center text-sm text-fd-muted-foreground" aria-live="polite">
              Loading benchmark data...
            </p>
          ) : info ? (
            <InfoPane key={info.name} k={info} live={live} />
          ) : (
            <p className="flex h-full items-center justify-center text-sm text-fd-muted-foreground">
              Select a kernel from the list.
            </p>
          )}
        </div>
        <div className="min-h-[36rem] p-4 md:h-1/2 md:min-h-0 md:overflow-y-auto md:p-5">
          {error ? null : loading ? (
            <div className="h-full min-h-64 animate-pulse rounded-lg bg-fd-muted/40" aria-hidden="true" />
          ) : (
            <PlotPane kernels={kernels} liveRows={live.rowsByOp} />
          )}
        </div>
      </section>
    </main>
  );
}
