import { KernelGrid } from '@/components/kernels/grid';
import { kernelIndex } from '@/lib/kernels';
import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Kernel explorer',
  description: 'Browse every Popcorn kernel: math, backends, and measured performance.',
};

export default async function Page() {
  const entries = await kernelIndex();
  return (
    <main className="mx-auto w-full max-w-6xl px-4 py-10">
      <h1 className="text-2xl font-bold">Kernel explorer</h1>
      <p className="mb-6 mt-1 text-fd-muted-foreground">
        Every kernel, its backends, and benchmark data recorded on real hardware.
      </p>
      <KernelGrid entries={entries} />
    </main>
  );
}
