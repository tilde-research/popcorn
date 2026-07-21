'use client';
import type { KernelIndexEntry } from '@/lib/kernels';
import Link from 'next/link';
import { useMemo, useState } from 'react';

export function KernelGrid({ entries }: { entries: KernelIndexEntry[] }) {
  const [query, setQuery] = useState('');
  const [active, setActive] = useState<string[]>([]);
  const tags = useMemo(() => [...new Set(entries.flatMap((e) => e.tags))].sort(), [entries]);

  const shown = entries.filter(
    (e) =>
      (e.name.includes(query.toLowerCase()) || (e.summary ?? '').toLowerCase().includes(query.toLowerCase())) &&
      active.every((tag) => e.tags.includes(tag)),
  );

  return (
    <div className="flex flex-col gap-4">
      <input
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder={`Search ${entries.length} kernels...`}
        className="w-full rounded-lg border bg-fd-card px-4 py-2.5 text-sm outline-none placeholder:text-fd-muted-foreground focus:ring-2 focus:ring-fd-ring"
      />
      <div className="flex flex-wrap gap-1.5">
        {tags.map((tag) => (
          <button
            key={tag}
            onClick={() => setActive((a) => (a.includes(tag) ? a.filter((t) => t !== tag) : [...a, tag]))}
            className={`rounded-full border px-2.5 py-0.5 text-xs transition-colors ${
              active.includes(tag)
                ? 'border-fd-primary bg-fd-primary text-fd-primary-foreground'
                : 'bg-fd-card text-fd-muted-foreground hover:text-fd-foreground'
            }`}
          >
            {tag}
          </button>
        ))}
      </div>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {shown.map((e) => (
          <Link
            key={e.name}
            href={`/kernels/${e.name}`}
            className="group flex flex-col gap-2 rounded-xl border bg-fd-card p-4 transition-colors hover:border-fd-primary/50"
          >
            <div className="flex items-baseline justify-between gap-2">
              <span className="font-mono text-sm font-semibold group-hover:text-fd-primary">{e.name}</span>
              <span className="shrink-0 text-xs text-fd-muted-foreground">
                {e.backends.length} backend{e.backends.length === 1 ? '' : 's'}
              </span>
            </div>
            <p className="line-clamp-2 text-sm text-fd-muted-foreground">{e.summary}</p>
            <div className="mt-auto flex flex-wrap gap-1">
              {e.tags.map((tag) => (
                <span key={tag} className="rounded-full bg-fd-muted px-2 py-0.5 text-[11px] text-fd-muted-foreground">
                  {tag}
                </span>
              ))}
            </div>
          </Link>
        ))}
      </div>
      {shown.length === 0 && <p className="py-12 text-center text-sm text-fd-muted-foreground">No kernels match.</p>}
    </div>
  );
}
