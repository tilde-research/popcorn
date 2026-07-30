import type { BaseLayoutProps } from 'fumadocs-ui/layouts/shared';
import { SiteHeader } from '@/components/site-header';
import { appName, basePath, gitConfig } from './shared';

/** Fixed box; the monochrome asset is inverted when the navigation is dark. */
export function NavTitle({ forceDark = false }: { forceDark?: boolean }) {
  return (
    <span
      className="nd-nav-title relative inline-flex h-6 w-[5.25rem] shrink-0 translate-y-px items-center overflow-hidden sm:h-7 sm:w-24"
      aria-label={appName}
    >
      <img
        src={`${basePath}/popcorn-name.png`}
        alt=""
        width={96}
        height={28}
        className={`absolute inset-0 h-full w-full object-contain object-left ${
          forceDark ? 'invert' : 'dark:invert'
        }`}
      />
    </span>
  );
}

type Options = {
  forceDarkNav?: boolean;
  variant?: 'home' | 'docs';
};

export function baseOptions(opts?: Options): BaseLayoutProps {
  const cinema = opts?.forceDarkNav === true;
  const variant = opts?.variant ?? 'home';

  return {
    nav: {
      title: <NavTitle forceDark={cinema} />,
      transparentMode: 'none',
      component: (
        <SiteHeader
          variant={variant}
          forceDark={cinema}
          showSidebarToggle={variant === 'docs'}
        />
      ),
    },
    // Custom SiteHeader owns search/theme/github; keep search enabled for ⌘K + triggers.
    themeSwitch: { enabled: false },
    links: [
      { text: 'Docs', url: '/docs' },
      { text: 'Kernels', url: '/kernels' },
    ],
    githubUrl: `https://github.com/${gitConfig.user}/${gitConfig.repo}`,
  };
}
