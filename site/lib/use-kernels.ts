'use client';
import { useEffect, useState } from 'react';
import type { Kernel } from './kernels';
import { basePath } from './shared';

const cache = new Map<string, Kernel>();
const pending = new Map<string, Promise<Kernel>>();

function load(name: string): Promise<Kernel> {
  const running = pending.get(name);
  if (running) return running;
  const promise = fetch(`${basePath}/data/${name}.json`)
    .then((response) => {
      if (!response.ok) throw new Error(`${name}: ${response.status} ${response.statusText}`);
      return response.json() as Promise<Kernel>;
    })
    .then((kernel) => {
      cache.set(name, kernel);
      return kernel;
    })
    .finally(() => pending.delete(name));
  pending.set(name, promise);
  return promise;
}

/** Fetch-and-cache one complete requested set, preserving request order. */
export function useKernels(names: string[]): { kernels: Kernel[]; loading: boolean; error: string | null } {
  const [, bump] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const missing = names.filter((name) => !cache.has(name));
  const key = names.join(',');

  useEffect(() => {
    setError(null);
    if (missing.length === 0) return;
    let alive = true;
    void Promise.all(missing.map(load))
      .then(() => alive && bump((n) => n + 1))
      .catch((reason: unknown) => alive && setError(reason instanceof Error ? reason.message : String(reason)));
    return () => {
      alive = false;
    };
  }, [key]); // eslint-disable-line react-hooks/exhaustive-deps

  const kernels = names.map((name) => cache.get(name)).filter((kernel): kernel is Kernel => kernel !== undefined);
  const loading = kernels.length !== names.length && error === null;
  return { kernels: loading ? [] : kernels, loading, error };
}
