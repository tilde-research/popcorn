# Popcorn docs site

Docs + kernel explorer, built with [Fumadocs](https://fumadocs.dev) and deployed to GitHub
Pages by `.github/workflows/site.yml`.

```bash
python ../scripts/export_site_data.py   # regenerate public/data/ from the package + reports
npm install
npm run dev                             # http://localhost:3000
```

- Docs pages live in `content/docs/*.mdx`.
- The kernel explorer is a single workspace at `app/kernels/` (deep-linkable via
  `?k=pinned,kernels&s=selected`): sidebar list in `components/kernels/workspace.tsx`,
  plotting in `components/kernels/plot-pane.tsx`, and shared series/metric logic in
  `components/kernels/model.ts` — add a metric to `METRICS` to add a y-axis option.
- `public/data/` is generated and gitignored; CI rebuilds it on every deploy.
