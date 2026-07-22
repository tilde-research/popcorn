import { Workspace } from '@/components/kernels/workspace';
import { kernelIndex } from '@/lib/kernels-server';
import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Kernel explorer',
  description: 'Browse every Popcorn kernel: math, backends, and measured performance.',
};

export default async function Page() {
  return <Workspace entries={await kernelIndex()} />;
}
