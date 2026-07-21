declare module 'plotly.js-cartesian-dist-min' {
  const Plotly: {
    react: (root: HTMLElement, data: unknown[], layout?: unknown, config?: unknown) => Promise<unknown>;
    purge: (root: HTMLElement) => void;
  };
  export default Plotly;
}
