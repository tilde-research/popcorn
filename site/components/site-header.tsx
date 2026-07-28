'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Sidebar } from 'lucide-react';
import { FullSearchTrigger, SearchTrigger } from 'fumadocs-ui/layouts/shared/slots/search-trigger';
import { ThemeSwitch } from 'fumadocs-ui/layouts/shared/slots/theme-switch';
import { SidebarCollapseTrigger } from 'fumadocs-ui/layouts/notebook/slots/sidebar';
import { cn } from '@/lib/cn';
import { NavTitle } from '@/lib/layout.shared';
import { gitConfig } from '@/lib/shared';

function GitHubIcon({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" className={className} aria-hidden>
      <path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12" />
    </svg>
  );
}

const LINKS = [
  { text: 'Docs', url: '/docs' },
  { text: 'Kernels', url: '/kernels' },
] as const;

type SiteHeaderProps = {
  variant?: 'home' | 'docs';
  forceDark?: boolean;
  /** When true, reserve/show the docs sidebar collapse control. */
  showSidebarToggle?: boolean;
};

/**
 * One header for home, kernels, and docs — same order, same slot sizes.
 * [logo] [Docs] [Kernels] ····· [Search] [Theme] [GitHub] [Sidebar?]
 */
export function SiteHeader({
  variant = 'home',
  forceDark = false,
  showSidebarToggle = false,
}: SiteHeaderProps) {
  const pathname = usePathname();

  return (
    <header
      id={variant === 'docs' ? 'nd-subnav' : 'nd-nav'}
      className={cn(
        'pc-site-header sticky z-40 border-b border-fd-border/80 backdrop-blur-lg',
        variant === 'docs'
          ? '[grid-area:header] top-(--fd-docs-row-1) z-10 layout:[--fd-header-height:--spacing(14)]'
          : 'top-0 h-14',
      )}
    >
      <div className="mx-auto flex h-14 w-full max-w-(--fd-layout-width) items-center gap-3 px-4 md:gap-4 md:px-6">
        <Link href="/" className="shrink-0" aria-label="Popcorn home">
          <NavTitle forceDark={forceDark} />
        </Link>

        <nav className="flex items-center gap-3 sm:gap-4">
          {LINKS.map((link) => {
            const active = pathname === link.url || pathname.startsWith(`${link.url}/`);
            return (
              <Link
                key={link.url}
                href={link.url}
                className={cn(
                  'text-sm transition-colors',
                  active
                    ? 'font-medium text-fd-foreground'
                    : 'text-fd-muted-foreground hover:text-fd-accent-foreground',
                )}
              >
                {link.text}
              </Link>
            );
          })}
        </nav>

        <div className="ms-auto flex items-center gap-1.5 sm:gap-2">
          <FullSearchTrigger hideIfDisabled className="pc-search-trigger" />
          <SearchTrigger hideIfDisabled className="pc-icon-slot" />

          <ThemeSwitch mode="light-dark" className="pc-theme-switch" />

          <a
            href={`https://github.com/${gitConfig.user}/${gitConfig.repo}`}
            target="_blank"
            rel="noreferrer noopener"
            aria-label="GitHub"
            className="pc-icon-slot text-fd-muted-foreground transition-colors hover:text-fd-accent-foreground"
          >
            <GitHubIcon className="size-4" />
          </a>

          {/* Fixed trailing slot — empty on non-docs so GitHub never shifts */}
          <div className="pc-icon-slot max-md:hidden">
            {showSidebarToggle ? (
              <SidebarCollapseTrigger
                aria-label="Toggle sidebar"
                className="inline-flex size-full items-center justify-center text-fd-muted-foreground transition-colors hover:text-fd-accent-foreground"
              >
                <Sidebar className="size-4" />
              </SidebarCollapseTrigger>
            ) : null}
          </div>
        </div>
      </div>
    </header>
  );
}
