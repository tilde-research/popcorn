import { HomeLayout } from 'fumadocs-ui/layouts/home';
import { baseOptions } from '@/lib/layout.shared';
import type { ReactNode } from 'react';
import 'katex/dist/katex.min.css';

export default function Layout({ children }: { children: ReactNode }) {
  return <HomeLayout {...baseOptions({ variant: 'home' })}>{children}</HomeLayout>;
}
