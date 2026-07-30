export function InstallSnippet() {
  return (
    <pre className="overflow-x-auto border border-fd-border bg-fd-card px-4 py-3 font-mono text-[13px] leading-relaxed text-fd-foreground">
      <code>
        <span className="select-none text-fd-muted-foreground">$ </span>
        {'uv pip install --torch-backend=auto popcorn'}
      </code>
    </pre>
  );
}
