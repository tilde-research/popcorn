import type { BaseLayoutProps } from 'fumadocs-ui/layouts/shared';
import { SiteHeader } from '@/components/site-header';
import { appName, basePath, gitConfig } from './shared';

/** Fixed box — both theme PNGs share exact geometry so icons never shift. */
export function NavTitle({ forceDark = false }: { forceDark?: boolean }) {
  return (
    <span
      className="nd-nav-title relative inline-block h-5 w-[4.5rem] shrink-0 overflow-hidden"
      aria-label={appName}
    >
      {!forceDark && (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={`${basePath}/popcorn-name-light.png`}
          alt=""
          width={72}
          height={20}
          className="absolute inset-0 h-5 w-[4.5rem] object-contain object-left dark:hidden"
        />
      )}
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={`${basePath}/popcorn-name-dark.png`}
        alt=""
        width={72}
        height={20}
        className={
          forceDark
            ? 'absolute inset-0 h-5 w-[4.5rem] object-contain object-left'
            : 'absolute inset-0 hidden h-5 w-[4.5rem] object-contain object-left dark:block'
        }
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
