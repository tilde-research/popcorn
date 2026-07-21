import { PerformancePanel } from '@/components/kernels/performance';
import { kernel, kernelIndex, type Kernel } from '@/lib/kernels';
import { basePath, gitConfig } from '@/lib/shared';
import { DynamicCodeBlock } from 'fumadocs-ui/components/dynamic-codeblock';
import { Tab, Tabs } from 'fumadocs-ui/components/tabs';
import katex from 'katex';
import { Download, FileCode } from 'lucide-react';
import type { Metadata } from 'next';
import type { ReactNode } from 'react';

export const dynamicParams = false;

export async function generateStaticParams() {
  return (await kernelIndex()).map(({ name }) => ({ name }));
}

export async function generateMetadata(props: PageProps<'/kernels/[name]'>): Promise<Metadata> {
  const { name } = await props.params;
  const data = await kernel(name);
  return { title: name, description: data.summary };
}

function usageSnippet(k: Kernel): string {
  const required = k.params.filter((p) => p.default === undefined).map((p) => p.name);
  const call = required.join(', ');
  const forced = k.backends.find((b) => b.name !== 'torch')?.name;
  const lines = [`from popcorn.kernels import ${k.name}`, '', `out = ${k.name}(${call})  # auto-dispatch`];
  if (forced) lines.push(`out = ${k.name}(${call}, backend="${forced}")  # force a backend`);
  return lines.join('\n');
}

function profileSnippet(k: Kernel): string {
  const call = k.params.filter((p) => p.default === undefined).map((p) => p.name).join(', ');
  return [
    `# validate + benchmark every eligible backend, then route to the fastest`,
    `out = ${k.name}(${call}, bench=True)`,
    '',
    `# or process-wide: POPCORN_BENCH=1 python train.py`,
    `print(${k.name})  # signature and per-backend constraints`,
  ].join('\n');
}

function BackendsPanel({ k }: { k: Kernel }) {
  return (
    <table className="w-full text-sm">
      <thead>
        <tr className="border-b text-left text-xs text-fd-muted-foreground">
          <th className="py-2 pr-4 font-medium">backend</th>
          <th className="py-2 pr-4 font-medium">source</th>
          <th className="py-2 font-medium">notes</th>
        </tr>
      </thead>
      <tbody>
        {k.backends.map((b) => (
          <tr key={b.name} className="border-b last:border-0">
            <td className="py-2 pr-4 font-mono font-semibold">{b.name}</td>
            <td className="py-2 pr-4 font-mono text-xs text-fd-muted-foreground">{b.source ?? 'reference'}</td>
            <td className="py-2 text-xs text-fd-muted-foreground">
              {[
                b.forward_only ? 'forward-only' : null,
                ...Object.entries(b.supports).map(([dim, constraint]) => `${dim}: ${constraint}`),
              ]
                .filter(Boolean)
                .join(', ') || '—'}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function CodePanel({ k }: { k: Kernel }) {
  return (
    <Tabs items={['Use it', 'Profile it']}>
      <Tab value="Use it">
        <DynamicCodeBlock lang="python" code={usageSnippet(k)} />
      </Tab>
      <Tab value="Profile it">
        <DynamicCodeBlock lang="python" code={profileSnippet(k)} />
      </Tab>
    </Tabs>
  );
}

// Add a panel here and it appears on every kernel card.
const PANELS: { id: string; title: string; render: (k: Kernel) => ReactNode }[] = [
  { id: 'performance', title: 'Performance', render: (k) => <PerformancePanel rows={k.rows} /> },
  { id: 'backends', title: 'Backends', render: (k) => <BackendsPanel k={k} /> },
  { id: 'code', title: 'Code', render: (k) => <CodePanel k={k} /> },
];

export default async function Page(props: PageProps<'/kernels/[name]'>) {
  const { name } = await props.params;
  const k = await kernel(name);
  const math = k.math
    ? katex.renderToString(k.math, { displayMode: true, throwOnError: false, strict: false })
    : null;

  return (
    <main className="mx-auto w-full max-w-5xl px-4 py-10">
      <div className="flex flex-wrap items-center gap-3">
        <h1 className="font-mono text-2xl font-bold">{k.name}</h1>
        <div className="flex flex-wrap gap-1">
          {k.tags.map((tag) => (
            <span key={tag} className="rounded-full bg-fd-muted px-2 py-0.5 text-[11px] text-fd-muted-foreground">
              {tag}
            </span>
          ))}
        </div>
        <div className="ms-auto flex items-center gap-2 text-xs">
          <a
            href={`https://github.com/${gitConfig.user}/${gitConfig.repo}/blob/${gitConfig.branch}/src/popcorn/kernels/${k.name}.py`}
            className="flex items-center gap-1.5 rounded-lg border bg-fd-card px-2.5 py-1.5 transition-colors hover:bg-fd-accent"
          >
            <FileCode className="size-3.5" /> source
          </a>
          <a
            href={`${basePath}/data/${k.name}.json`}
            download={`${k.name}.json`}
            className="flex items-center gap-1.5 rounded-lg border bg-fd-card px-2.5 py-1.5 transition-colors hover:bg-fd-accent"
          >
            <Download className="size-3.5" /> data
          </a>
        </div>
      </div>
      <p className="mt-2 text-fd-muted-foreground">{k.summary}</p>
      {math && <div className="my-4 overflow-x-auto" dangerouslySetInnerHTML={{ __html: math }} />}
      {k.citations.length > 0 && (
        <p className="text-xs text-fd-muted-foreground">
          {k.citations.map((citation, index) => (
            <span key={citation.url}>
              {index > 0 && ', '}
              <a href={citation.url} className="underline hover:text-fd-foreground">
                {citation.label}
              </a>
            </span>
          ))}
        </p>
      )}
      <div className="mt-8 flex flex-col gap-6">
        {PANELS.map((panel) => (
          <section key={panel.id} className="rounded-xl border bg-fd-card/50 p-5">
            <h2 className="mb-4 text-sm font-semibold uppercase tracking-wide text-fd-muted-foreground">
              {panel.title}
            </h2>
            {panel.render(k)}
          </section>
        ))}
      </div>
    </main>
  );
}
