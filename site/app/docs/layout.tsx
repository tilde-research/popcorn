import { source } from '@/lib/source';
import { DocsLayout } from 'fumadocs-ui/layouts/notebook';
import { baseOptions } from '@/lib/layout.shared';

export default function Layout({ children }: LayoutProps<'/docs'>) {
  const options = baseOptions({ variant: 'docs' });
  return (
    <DocsLayout
      tree={source.getPageTree()}
      {...options}
      nav={{
        ...options.nav,
        mode: 'top',
      }}
    >
      {children}
    </DocsLayout>
  );
}
