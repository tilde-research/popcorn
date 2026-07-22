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
    .then((response) => response.json() as Promise<Kernel>)
    .then((kernel) => {
      cache.set(name, kernel);
      return kernel;
    })
    .finally(() => pending.delete(name));
  pending.set(name, promise);
  return promise;
}

/** Fetch-and-cache kernel data; returns the loaded subset, in request order. */
export function useKernels(names: string[]): Kernel[] {
  const [, bump] = useState(0);
  const missing = names.filter((name) => !cache.has(name));

  useEffect(() => {
    if (missing.length === 0) return;
    let alive = true;
    void Promise.all(missing.map(load)).then(() => alive && bump((n) => n + 1));
    return () => {
      alive = false;
    };
  }, [missing.join(',')]); // eslint-disable-line react-hooks/exhaustive-deps

  return names.map((name) => cache.get(name)).filter((kernel): kernel is Kernel => kernel !== undefined);
}
