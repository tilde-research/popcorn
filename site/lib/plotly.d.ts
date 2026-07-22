declare module 'plotly.js-cartesian-dist-min' {
  const Plotly: {
    react: (root: HTMLElement, data: unknown[], layout?: unknown, config?: unknown) => Promise<unknown>;
    purge: (root: HTMLElement) => void;
    Plots: { resize: (root: HTMLElement) => Promise<unknown> };
  };
  export default Plotly;
}
