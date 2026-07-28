'use client';

import type { CSSProperties } from 'react';
import { basePath } from '@/lib/shared';

const MARK = `${basePath}/popcorn-logo-dark.png`;

const KERNELS = [
  { top: '12%', left: '8%', size: 56, dur: '16s', delay: '0s', pop: true, popDur: '8s', popDelay: '1.2s' },
  { top: '22%', left: '78%', size: 72, dur: '18s', delay: '-3s', pop: true, popDur: '9s', popDelay: '4s' },
  { top: '58%', left: '12%', size: 44, dur: '14s', delay: '-6s', pop: false },
  { top: '68%', left: '84%', size: 60, dur: '20s', delay: '-2s', pop: true, popDur: '7s', popDelay: '2.5s' },
  { top: '38%', left: '90%', size: 36, dur: '13s', delay: '-8s', pop: false },
  { top: '78%', left: '42%', size: 48, dur: '17s', delay: '-4s', pop: true, popDur: '10s', popDelay: '0.8s' },
  { top: '8%', left: '48%', size: 40, dur: '15s', delay: '-1s', pop: false },
] as const;

function markStyle(extra: CSSProperties = {}): CSSProperties {
  return {
    backgroundColor: 'rgba(255,255,255,0.9)',
    WebkitMaskImage: `url(${MARK})`,
    maskImage: `url(${MARK})`,
    WebkitMaskSize: 'contain',
    maskSize: 'contain',
    WebkitMaskRepeat: 'no-repeat',
    maskRepeat: 'no-repeat',
    WebkitMaskPosition: 'center',
    maskPosition: 'center',
    ...extra,
  };
}

/** Sparse drifting popcorn marks — decorative only. */
export function KernelField() {
  return (
    <div className="pointer-events-none absolute inset-0 overflow-hidden" aria-hidden>
      <div
        className="absolute inset-0"
        style={{
          background:
            'radial-gradient(ellipse 70% 55% at 50% 40%, rgba(224,168,74,0.10), transparent 70%), radial-gradient(ellipse 50% 40% at 70% 70%, rgba(255,255,255,0.035), transparent 60%)',
        }}
      />
      <div
        className="absolute left-1/2 top-[42%] h-[min(52vw,420px)] w-[min(52vw,420px)] -translate-x-1/2 -translate-y-1/2 opacity-[0.12]"
        style={markStyle()}
      />
      {KERNELS.map((k, i) => {
        const style = markStyle({
          top: k.top,
          left: k.left,
          width: k.size,
          height: k.size,
          '--pc-dur': k.dur,
          '--pc-delay': k.delay,
          ...(k.pop ? { '--pc-pop-dur': k.popDur, '--pc-pop-delay': k.popDelay } : {}),
        } as CSSProperties);

        return (
          <div
            key={i}
            className={`absolute opacity-[0.28] ${k.pop ? 'pc-kernel-pop' : 'pc-kernel'}`}
            style={style}
          />
        );
      })}
    </div>
  );
}
