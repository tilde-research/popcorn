'use client';

import { useEffect, useState } from 'react';

/** Upstream import paths mirror the `source=` targets registered in src/popcorn/kernels. */
const SWAPS = [
  {
    op: 'rms_norm',
    before: ['from fla.modules.layernorm import rms_norm', '', 'y = rms_norm(x, weight)'],
    after: ['from popcorn.kernels import rms_norm', '', 'y = rms_norm(x, weight)'],
  },
  {
    op: 'swiglu',
    before: ['from liger_kernel.transformers.functional import liger_swiglu', '', 'y = liger_swiglu(a, b)'],
    after: ['from popcorn.kernels import swiglu', '', 'y = swiglu(a, b)'],
  },
  {
    op: 'attn',
    before: ['from flash_attn_interface import flash_attn_func', '', 'o = flash_attn_func(q, k, v, causal=True)'],
    after: ['from popcorn.kernels import attn', '', 'o = attn(q, k, v, causal=True)'],
  },
  {
    op: 'cross_entropy',
    before: [
      'from unsloth.kernels.cross_entropy_loss import Fast_CrossEntropyLoss',
      '',
      'loss = Fast_CrossEntropyLoss.apply(logits, labels)',
    ],
    after: ['from popcorn.kernels import cross_entropy', '', 'loss = cross_entropy(logits, labels)'],
  },
  {
    op: 'layer_norm',
    before: [
      'from liger_kernel.transformers.functional import liger_layer_norm',
      '',
      'y = liger_layer_norm(x, weight, bias)',
    ],
    after: ['from popcorn.kernels import layer_norm', '', 'y = layer_norm(x, weight, bias)'],
  },
  {
    op: 'rope',
    before: ['from unsloth.kernels.rope_embedding import fast_rope_embedding', '', 'q, k = fast_rope_embedding(q, k, cos, sin)'],
    after: ['from popcorn.kernels import rope', '', 'q, k = rope(q, k, cos, sin)'],
  },
  {
    op: 'gla',
    before: ['from fla.ops.gla import chunk_gla', '', 'o = chunk_gla(q, k, v, g)'],
    after: ['from popcorn.kernels import gla', '', 'o = gla(q, k, v, g)'],
  },
  {
    op: 'softmax',
    before: ['from fla.ops.utils import softmax_fwd', '', 'y = softmax_fwd(x)'],
    after: ['from popcorn.kernels import softmax', '', 'y = softmax(x)'],
  },
  {
    op: 'geglu',
    before: ['from liger_kernel.transformers.functional import liger_geglu', '', 'y = liger_geglu(a, b)'],
    after: ['from popcorn.kernels import geglu', '', 'y = geglu(a, b)'],
  },
  {
    op: 'l2_norm',
    before: ['from fla.modules.l2norm import l2_norm', '', 'y = l2_norm(x)'],
    after: ['from popcorn.kernels import l2_norm', '', 'y = l2_norm(x)'],
  },
] as const;

const INTERVAL = 3600;

export function SwapDiff() {
  const [index, setIndex] = useState(0);
  const [paused, setPaused] = useState(false);

  useEffect(() => {
    if (paused || window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    const id = setInterval(() => setIndex((current) => (current + 1) % SWAPS.length), INTERVAL);
    return () => clearInterval(id);
  }, [paused]);

  const swap = SWAPS[index];
  const lines = [
    ...swap.before.map((text) => ({ sign: '-', text })),
    ...swap.after.map((text) => ({ sign: '+', text })),
  ];

  return (
    <div
      className="border border-fd-border bg-fd-card"
      onMouseEnter={() => setPaused(true)}
      onMouseLeave={() => setPaused(false)}
    >
      <div className="flex items-center justify-between border-b border-fd-border px-4 py-2.5">
        <span key={swap.op} className="pc-diff-line font-mono text-xs text-fd-muted-foreground">
          {swap.op}
        </span>
        <div className="flex gap-1.5" role="tablist" aria-label="Kernel swap examples">
          {SWAPS.map((item, i) => (
            <button
              key={item.op}
              role="tab"
              aria-selected={i === index}
              aria-label={item.op}
              onClick={() => setIndex(i)}
              className={`h-1 w-4 transition-colors ${
                i === index
                  ? 'bg-[color:var(--pc-butter)]'
                  : 'bg-fd-foreground/15 hover:bg-fd-foreground/35'
              }`}
            />
          ))}
        </div>
      </div>

      <pre className="overflow-x-auto py-3 font-mono text-[13px] leading-relaxed">
        <code>
          {lines.map((line, i) => (
            <span
              key={`${index}-${i}`}
              className={`pc-diff-line block px-4 ${
                line.sign === '-'
                  ? 'bg-red-500/[0.07] text-red-700/80 dark:text-red-300/80'
                  : 'bg-emerald-500/[0.07] text-emerald-700/90 dark:text-emerald-300/90'
              }`}
              style={{ animationDelay: `${i * 55}ms` }}
            >
              <span className="select-none opacity-50">{line.sign} </span>
              {line.text}
              {'\n'}
            </span>
          ))}
        </code>
      </pre>
    </div>
  );
}
