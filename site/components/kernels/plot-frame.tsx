'use client';
import Plotly from 'plotly.js-cartesian-dist-min';
import { useEffect, useRef } from 'react';

/** Thin imperative bridge to Plotly with theme-aware defaults. */
export function PlotFrame({ data, layout }: { data: unknown[]; layout: Record<string, unknown> }) {
  const root = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const element = root.current;
    if (!element) return;
    const styles = getComputedStyle(element);
    void Plotly.react(
      element,
      data,
      {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: styles.color, family: styles.fontFamily, size: 12 },
        margin: { l: 56, r: 16, t: 16, b: 44 },
        ...layout,
      },
      { responsive: true, displaylogo: false },
    ).then(() => Plotly.Plots.resize(element)); // container may have settled after first measure
  }, [data, layout]);

  useEffect(() => {
    const element = root.current;
    if (!element) return;
    const observer = new ResizeObserver(() => {
      if (element.classList.contains('js-plotly-plot')) void Plotly.Plots.resize(element);
    });
    observer.observe(element);
    return () => {
      observer.disconnect();
      Plotly.purge(element);
    };
  }, []);

  return <div ref={root} className="h-full min-h-64 w-full" />;
}
