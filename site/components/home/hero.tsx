import Link from 'next/link';
import { basePath } from '@/lib/shared';
import { KernelField } from './kernel-field';

export function Hero({ implementationCount }: { implementationCount: number }) {
  const count = implementationCount.toLocaleString('en-US');

  return (
    <section className="relative flex min-h-[calc(100svh-3.5rem)] flex-col justify-center px-6 pb-20 pt-10 sm:px-10">
      <KernelField />
      <div className="relative z-10 mx-auto w-full max-w-3xl">
        <div className="pc-fade-up relative mb-8 w-full max-w-[44rem]">
          <div role="img" aria-label="popcorn">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={`${basePath}/popcorn-name.png`}
              alt=""
              width={1200}
              height={340}
              className="block h-auto w-full dark:invert"
            />
          </div>
          <a
            href="https://tilderesearch.com"
            target="_blank"
            rel="noreferrer noopener"
            aria-label="Tilde Research"
            className="absolute bottom-[3%] right-[1%] block w-[34%] transition-opacity hover:opacity-70 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[color:var(--pc-butter)]"
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={`${basePath}/by-tilde-research.png`}
              alt=""
              width={4408}
              height={528}
              className="block h-auto w-full dark:invert"
            />
          </a>
        </div>
        <h1
          aria-label={`One interface, ${count} verified kernel implementations.`}
          className="pc-fade-up-delay max-w-2xl text-[1.85rem] leading-[1.2] tracking-tight text-fd-foreground sm:text-[2.35rem] md:text-[2.75rem]"
          style={{ fontFamily: 'var(--font-display)' }}
        >
          <span aria-hidden>
            One interface, {count}{' '}
            <span className="underline decoration-[color:var(--pc-butter)] decoration-1 underline-offset-[6px]">
              verified
            </span>
            <br /> kernel implementations.
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
