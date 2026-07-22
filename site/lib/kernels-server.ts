import fs from 'node:fs/promises';
import path from 'node:path';
import type { KernelIndexEntry } from './kernels';

/** Build-time loader for the generated data (server components only). */
export async function kernelIndex(): Promise<KernelIndexEntry[]> {
  const file = path.join(process.cwd(), 'public', 'data', 'index.json');
  return JSON.parse(await fs.readFile(file, 'utf-8'));
}
