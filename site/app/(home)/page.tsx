import Link from 'next/link';
import { Hero } from '@/components/home/hero';
import { InstallSnippet } from '@/components/home/install-snippet';
import { SwapDiff } from '@/components/home/swap-diff';
import { gitConfig } from '@/lib/shared';

const STATS = [
  { value: '96', label: 'kernels' },
  { value: '6', label: 'backends' },
  { value: '128', label: 'implementations' },
] as const;

export default function HomePage() {
  return (
    <div className="pc-home flex flex-1 flex-col">
      <Hero />

      <section className="border-t border-white/10 px-6 py-20 sm:px-10">
        <div className="mx-auto max-w-3xl">
          <h2
            className="text-2xl tracking-tight text-white sm:text-3xl"
            style={{ fontFamily: 'var(--font-display)' }}
          >
            Install and call.
          </h2>
          <p className="mt-3 max-w-xl text-stone-400">
            Replace upstream kernel calls with calls to popcorn kernels.
          </p>
          <div className="mt-8">
            <InstallSnippet />
          </div>
          <div className="mt-4">
            <SwapDiff />
          </div>
        </div>
      </section>

      <section className="border-t border-white/10 px-6 py-20 sm:px-10">
        <div className="mx-auto max-w-3xl text-center">
          <h2
            className="text-2xl tracking-tight text-white sm:text-3xl"
            style={{ fontFamily: 'var(--font-display)' }}
          >
            Explore the grid.
          </h2>
          <dl className="mt-12 flex flex-wrap items-start justify-center gap-x-16 gap-y-8">
            {STATS.map((s) => (
              <div key={s.label}>
                <dt className="font-mono text-4xl text-white tabular-nums sm:text-5xl">
                  {s.value}
                </dt>
                <dd className="mt-2 text-sm text-stone-500">{s.label}</dd>
              </div>
            ))}
          </dl>
          <Link
            href="/kernels"
            className="mt-12 inline-flex items-center justify-center border border-[color:var(--pc-butter)] bg-[color:var(--pc-butter)] px-5 py-2.5 text-sm font-medium text-[color:var(--pc-ink)] transition-colors hover:bg-transparent hover:text-[color:var(--pc-butter)]"
          >
            Open the kernel explorer
          </Link>
        </div>
      </section>

      <footer className="mt-auto border-t border-white/10 px-6 py-8 sm:px-10">
        <div className="mx-auto flex max-w-3xl flex-col gap-2 text-xs text-stone-500 sm:flex-row sm:items-center sm:justify-between">
          <p>Apache-2.0 · Tilde Research</p>
          <a
            href={`https://github.com/${gitConfig.user}/${gitConfig.repo}`}
            className="transition-colors hover:text-stone-300"
            target="_blank"
            rel="noreferrer"
          >
            github.com/{gitConfig.user}/{gitConfig.repo}
          </a>
        </div>
      </footer>
    </div>
  );
}
