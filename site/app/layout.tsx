import { DM_Sans, Instrument_Serif } from 'next/font/google';
import { Provider } from '@/components/provider';
import { siteUrl } from '@/lib/shared';
import type { Metadata } from 'next';
import './global.css';

const sans = DM_Sans({
  subsets: ['latin'],
  variable: '--font-sans',
});

const display = Instrument_Serif({
  subsets: ['latin'],
  weight: '400',
  variable: '--font-display',
});

export const metadata: Metadata = {
  metadataBase: new URL(siteUrl),
  title: {
    default: 'Popcorn',
    template: '%s | Popcorn',
  },
  description: 'Reference-checked, benchmark-driven kernel dispatch for PyTorch.',
};

export default function Layout({ children }: LayoutProps<'/'>) {
  return (
    <html lang="en" className={`${sans.variable} ${display.variable} ${sans.className}`} suppressHydrationWarning>
      <body className="flex flex-col min-h-screen">
        <Provider>{children}</Provider>
      </body>
    </html>
  );
}
