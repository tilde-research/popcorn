import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createMDX } from 'fumadocs-mdx/next';

const withMDX = createMDX();

/** @type {import('next').NextConfig} */
const config = {
  output: 'export',
  reactStrictMode: true,
  devIndicators: false,
  // Docs markdown lives at the repo root (../docs), outside the Next.js project dir;
  // widen the turbopack module-resolution root so those imports resolve.
  turbopack: { root: path.join(path.dirname(fileURLToPath(import.meta.url)), '..') },
  // Set NEXT_PUBLIC_BASE_PATH=/popcorn when deploying to GitHub project pages.
  basePath: process.env.NEXT_PUBLIC_BASE_PATH || '',
};

export default withMDX(config);
