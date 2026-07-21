# Popcorn docs site

Docs + kernel explorer, built with [Fumadocs](https://fumadocs.dev) and deployed to GitHub
Pages by `.github/workflows/site.yml`.

```bash
python ../scripts/export_site_data.py   # regenerate public/data/ from the package + reports
npm install
npm run dev                             # http://localhost:3000
```

- Docs pages live in `content/docs/*.mdx`.
- The kernel explorer is `app/kernels/`; card panels are registered in
  `app/kernels/[name]/page.tsx` (`PANELS`) — add an entry there to add a panel to every card.
- `public/data/` is generated and gitignored; CI rebuilds it on every deploy.
