import type { BaseLayoutProps } from 'fumadocs-ui/layouts/shared';
import { appName, basePath, gitConfig } from './shared';

export function baseOptions(): BaseLayoutProps {
  return {
    nav: {
      title: (
        <>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={`${basePath}/popcorn-name-light.png`} alt={appName} className="h-5 w-auto dark:hidden" />
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={`${basePath}/popcorn-name-dark.png`} alt={appName} className="hidden h-5 w-auto dark:block" />
        </>
      ),
    },
    links: [
      { text: 'Docs', url: '/docs' },
      { text: 'Kernels', url: '/kernels' },
    ],
    githubUrl: `https://github.com/${gitConfig.user}/${gitConfig.repo}`,
  };
}
