'use client';
import Plotly from 'plotly.js-cartesian-dist-min';
import { useEffect, useRef } from 'react';

function displayedPlot(element: HTMLDivElement): boolean {
  if (!element.isConnected || !element.classList.contains('js-plotly-plot')) return false;
  const styles = getComputedStyle(element);
  return (
    styles.display !== 'none' &&
    styles.visibility !== 'hidden' &&
    element.offsetWidth > 0 &&
    element.offsetHeight > 0
  );
}

function rethrowUnlessHiddenResize(error: unknown): void {
  if (!(error instanceof Error) || error.message !== 'Resize must be passed a displayed plot div element.') {
    throw error;
  }
}

function resizeIfDisplayed(element: HTMLDivElement): void {
  if (!displayedPlot(element)) return;
  // Plotly rejects if a tab becomes hidden after our check but before its queued resize.
  void Plotly.Plots.resize(element).catch(rethrowUnlessHiddenResize);
}

/** Thin imperative bridge to Plotly with theme-aware defaults. */
export function PlotFrame({ data, layout }: { data: unknown[]; layout: Record<string, unknown> }) {
  const root = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const element = root.current;
    if (!element) return;
    let active = true;
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
      { displaylogo: false },
    ).then(() => {
      if (active) resizeIfDisplayed(element);
    }).catch(rethrowUnlessHiddenResize);
    return () => {
      active = false;
    };
  }, [data, layout]);

  useEffect(() => {
    const element = root.current;
    if (!element) return;
    let frame = 0;
    const observer = new ResizeObserver(() => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => resizeIfDisplayed(element));
    });
    observer.observe(element);
    return () => {
      observer.disconnect();
      cancelAnimationFrame(frame);
      Plotly.purge(element);
    };
  }, []);

  return <div ref={root} className="h-full min-h-64 w-full" />;
}
