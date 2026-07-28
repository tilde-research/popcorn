import { HomeLayout } from 'fumadocs-ui/layouts/home';
import { baseOptions } from '@/lib/layout.shared';

export default function Layout({ children }: LayoutProps<'/'>) {
  return (
    <HomeLayout {...baseOptions({ forceDarkNav: true, variant: 'home' })} className="pc-home">
      {children}
    </HomeLayout>
  );
}
