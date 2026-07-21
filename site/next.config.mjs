import { createMDX } from 'fumadocs-mdx/next';

const withMDX = createMDX();

/** @type {import('next').NextConfig} */
const config = {
  output: 'export',
  reactStrictMode: true,
  // Set NEXT_PUBLIC_BASE_PATH=/popcorn when deploying to GitHub project pages.
  basePath: process.env.NEXT_PUBLIC_BASE_PATH || '',
};

export default withMDX(config);
