export const appName = 'Popcorn';
export const docsRoute = '/docs';
export const docsImageRoute = '/og/docs';
export const docsContentRoute = '/llms.mdx/docs';

export const gitConfig = {
  user: 'tilde-research',
  repo: 'popcorn',
  branch: 'main',
};

export const siteUrl =
  process.env.NEXT_PUBLIC_SITE_URL || `https://${gitConfig.user}.github.io/${gitConfig.repo}`;

/** Prefix for static assets fetched at runtime (GitHub project pages). */
export const basePath = process.env.NEXT_PUBLIC_BASE_PATH || '';
