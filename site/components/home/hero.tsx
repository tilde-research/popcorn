import Link from 'next/link';
import { basePath } from '@/lib/shared';
import { KernelField } from './kernel-field';

export function Hero() {
  return (
    <section className="relative flex min-h-[calc(100svh-3.5rem)] flex-col justify-center px-6 pb-20 pt-10 sm:px-10">
      <KernelField />
      <div className="relative z-10 mx-auto w-full max-w-3xl">
        <div
          role="img"
          aria-label="popcorn"
          className="pc-fade-up mb-8 h-14 max-w-full sm:h-20 md:h-24"
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={`${basePath}/popcorn-name-light.png`}
            alt=""
            width={1200}
            height={340}
            className="h-full w-auto max-w-full object-contain object-left dark:hidden"
          />
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={`${basePath}/popcorn-name-dark.png`}
            alt=""
            width={1200}
            height={340}
            className="hidden h-full w-auto max-w-full object-contain object-left dark:block"
          />
        </div>
        <h1
          aria-label="Reference-checked kernels for every workload."
          className="pc-fade-up-delay max-w-2xl text-[1.85rem] leading-[1.2] tracking-tight text-fd-foreground sm:text-[2.35rem] md:text-[2.75rem]"
          style={{ fontFamily: 'var(--font-display)' }}
        >
          <span aria-hidden>
            Reference-
            <em className="italic underline decoration-[color:var(--pc-butter)] decoration-1 underline-offset-[6px]">
              checked
            </em>{' '}
            kernels
            <br className="hidden sm:block" /> for every workload.
          </span>
        </h1>
        <div className="pc-fade-up-delay-2 mt-10 flex flex-wrap items-center gap-x-8 gap-y-4">
          <Link
            href="/docs/quickstart"
            className="inline-flex items-center justify-center border border-[color:var(--pc-butter)] bg-[color:var(--pc-butter)] px-5 py-2.5 text-sm font-medium text-[color:var(--pc-ink)] transition-colors hover:bg-transparent hover:text-[color:var(--pc-butter)]"
          >
            Get started
          </Link>
          <Link
            href="/kernels"
            className="text-sm text-fd-muted-foreground underline decoration-fd-border underline-offset-4 transition-colors hover:text-[color:var(--pc-butter)] hover:decoration-[color:var(--pc-butter)]"
          >
            Explore kernels
          </Link>
        </div>
      </div>
    </section>
  );
}
