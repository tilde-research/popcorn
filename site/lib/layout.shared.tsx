import type { BaseLayoutProps } from 'fumadocs-ui/layouts/shared';
import { appName, basePath, gitConfig } from './shared';

export function baseOptions(): BaseLayoutProps {
  return {
    nav: {
      title: (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={`${basePath}/popcorn-banner.png`} alt={appName} className="h-7 w-auto" />
      ),
    },
    links: [
      { text: 'Docs', url: '/docs' },
      { text: 'Kernels', url: '/kernels' },
    ],
    githubUrl: `https://github.com/${gitConfig.user}/${gitConfig.repo}`,
  };
}
