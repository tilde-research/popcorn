"""Self-contained HTML viewer for the reports database."""

import json
from collections.abc import Iterable

from popcorn.bench.model import Record


def render(records: Iterable[Record]) -> str:
    records = list(records)
    if not records:
        return "<!doctype html><title>popcorn reports</title><p>No report rows.</p>"
    rows = [record.to_dict() for record in records]
    data = json.dumps(rows, separators=(",", ":")).replace("</", "<\\/")
    return _TEMPLATE.replace("__DATA__", data)


_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>popcorn reports</title>
<style>
:root {
  --bg: #101216; --panel: #171a21; --line: #262b36; --text: #d7dde8; --dim: #8a93a6;
  --pass: #4ec97a; --skip: #6b7385; --fail: #ff6b6b; --crash: #c792ea; --error: #e5a35b; --accent: #7aa2f7;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text);
       font: 13px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
main { max-width: 1400px; margin: 0 auto; padding: 24px 20px 80px; }
h1 { font-size: 20px; margin: 0 0 4px; }
h2 { font-size: 14px; color: var(--dim); margin: 28px 0 10px; text-transform: uppercase; letter-spacing: .08em; }
.meta { color: var(--dim); margin-bottom: 8px; }
table { border-collapse: collapse; width: 100%; }
th, td { padding: 4px 10px; text-align: left; white-space: nowrap; }
thead th { color: var(--dim); border-bottom: 1px solid var(--line); position: sticky; top: 0;
           background: var(--bg); cursor: pointer; user-select: none; }
tbody tr { border-bottom: 1px solid var(--line); }
tbody tr.case { cursor: pointer; }
tbody tr.case:hover { background: var(--panel); }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.pill { padding: 1px 8px; border-radius: 9px; font-size: 12px; }
.pass  { color: var(--pass);  background: color-mix(in srgb, var(--pass)  12%, transparent); }
.skip  { color: var(--skip);  background: color-mix(in srgb, var(--skip)  14%, transparent); }
.fail  { color: var(--fail);  background: color-mix(in srgb, var(--fail)  12%, transparent); }
.crash { color: var(--crash); background: color-mix(in srgb, var(--crash) 12%, transparent); }
.error { color: var(--error); background: color-mix(in srgb, var(--error) 12%, transparent); }
.good { color: var(--pass); } .bad { color: #e5a35b; } .dim { color: var(--dim); }
#matrix td.fastest { box-shadow: inset 0 0 0 1px var(--accent);
                     background: color-mix(in srgb, var(--accent) 8%, transparent); }
.reason { max-width: 340px; overflow: hidden; text-overflow: ellipsis; color: var(--dim); }
#matrix td { text-align: center; }
#matrix td.cell { cursor: pointer; }
#matrix td.cell:hover { background: var(--panel); }
#matrix td:first-child { text-align: left; }
#controls { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0; align-items: center; }
select, input, button { background: var(--panel); color: var(--text); border: 1px solid var(--line);
                        border-radius: 6px; padding: 5px 8px; font: inherit; }
input { width: 260px; }
button { cursor: pointer; }
#counts span { margin-right: 14px; }
.detail td { white-space: normal; }
.detail .box { display: flex; gap: 40px; flex-wrap: wrap; padding: 8px 0 12px; }
.detail table { width: auto; }
.detail th { position: static; cursor: default; }
.matrix-wrap, .table-wrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; }
.note { color: var(--dim); margin: 8px 0; }
</style>
</head>
<body>
<main>
<h1>popcorn reports</h1>
<div class="meta" id="meta"></div>

<h2>Support matrix</h2>
<div class="note">&#10004; whole grid passes &middot; &#10004;* some cases are gated, failed, or unverified &middot; &#10008; nothing passes
&middot; numbers are median forward speedup vs torch &middot; highlighted cells are the fastest for at least one case (hover for how many) &middot; click a cell to filter</div>
<div class="matrix-wrap"><table id="matrix"></table></div>

<h2>Cases</h2>
<div class="note">&#9889; fastest measured impl for that exact case (and faster than torch)</div>
<div id="controls">
  <select id="f-op"></select>
  <select id="f-impl"></select>
  <select id="f-dtype"></select>
  <select id="f-status"></select>
  <input id="f-search" placeholder="search case / reason" type="search">
  <button id="f-clear">clear</button>
  <span id="counts"></span>
</div>
<div class="table-wrap"><table id="cases">
  <thead><tr>
    <th data-k="status">status</th><th data-k="op">op</th><th data-k="impl">impl</th>
    <th data-k="dtype">dtype</th><th data-k="case">case</th>
    <th data-k="fwd" class="num">fwd err/scale</th><th data-k="bwd" class="num">bwd err/scale</th>
    <th data-k="sf" class="num">fwd &times;</th><th data-k="sb" class="num">bwd &times;</th>
    <th data-k="mem" class="num">mem &times;</th><th data-k="reason">reason</th>
  </tr></thead>
  <tbody></tbody>
</table></div>
<div class="note" id="truncated"></div>
</main>

<script>
const DATA = __DATA__;
const SEVERITY = {crash: 0, fail: 1, error: 2, pass: 3, skip: 4};

const relerr = g => {
  const vals = Object.values(g || {});
  return vals.length ? Math.max(...vals.map(v => v.scale ? v.err / v.scale : v.err)) : null;
};
const speed = (r, d) => r.bench && r.bench[d + "_ms"] && r.bench["ref_" + d + "_ms"]
  ? r.bench["ref_" + d + "_ms"] / r.bench[d + "_ms"] : null;
const memx = r => {
  const b = r.bench || {};
  const mem = Math.max(b.fwd_mem_mb || 0, b.bwd_mem_mb || 0);
  const ref = Math.max(b.ref_fwd_mem_mb || 0, b.ref_bwd_mem_mb || 0);
  return mem ? ref / mem : null;
};
const ROWS = DATA.map(r => ({
  r, op: r.op, impl: r.impl, status: r.status, dtype: r.config.dtype,
  case: r.case, reason: r.reason || (r.bench_error ? "benchmark: " + r.bench_error : ""),
  fwd: relerr(r.fwd), bwd: relerr(r.bwd),
  sf: speed(r, "fwd"), sb: speed(r, "bwd"), mem: memx(r),
}));

// Fastest impl per case: lowest measured fwd time, but only a win when it
// also beats the torch reference (speedup > 1); otherwise torch is fastest.
const CHAMPION = {};
ROWS.forEach(v => {
  if (!v.r.bench || !v.r.bench.fwd_ms) return;
  const key = v.op + "|" + v.r.device + "|" + v.r.case_id;
  if (!CHAMPION[key] || v.r.bench.fwd_ms < CHAMPION[key].ms)
    CHAMPION[key] = {impl: v.impl, ms: v.r.bench.fwd_ms, wins: v.sf > 1};
});
ROWS.forEach(v => {
  const c = CHAMPION[v.op + "|" + v.r.device + "|" + v.r.case_id];
  v.fastest = !!c && c.wins && c.impl === v.impl;
});

// Case wins per op: the champion impl when it beats torch, torch otherwise.
const WINS = {}, MEASURED = {};
Object.entries(CHAMPION).forEach(([key, c]) => {
  const op = key.split("|", 1)[0];
  MEASURED[op] = (MEASURED[op] || 0) + 1;
  const winner = op + "|" + (c.wins ? c.impl : "torch");
  WINS[winner] = (WINS[winner] || 0) + 1;
});

const median = a => {
  const s = [...a].sort((x, y) => x - y);
  return s.length ? (s[s.length >> 1] + s[(s.length - 1) >> 1]) / 2 : null;
};

const $ = s => document.querySelector(s);
const uniq = k => [...new Set(ROWS.map(v => v[k]))].sort();
const fmt = (x, d) => x == null ? "-" : x.toExponential ? x.toExponential(1) : x;
const fx = x => x == null ? "-" : `<span class="${x >= 1 ? "good" : "bad"}">${x.toFixed(2)}&times;</span>`;
const esc = s => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/"/g, "&quot;");

const filters = {op: "", impl: "", dtype: "", status: "", search: ""};
let sortKey = null, sortDir = 1;

function fillSelect(id, key, label) {
  $(id).innerHTML = `<option value="">${label}: all</option>` +
    uniq(key).map(v => `<option>${v}</option>`).join("");
  $(id).onchange = e => { filters[key] = e.target.value; renderTable(); };
}

function renderMeta() {
  const devices = [...new Set(DATA.map(r => r.device))].join(", ");
  const latest = DATA.map(r => r.ts).sort().at(-1);
  $("#meta").textContent = `${DATA.length} case-impl rows on ${devices}` +
    ` \u00b7 torch ${DATA[0].torch} \u00b7 latest run ${latest}`;
}

function renderMatrix() {
  const ops = uniq("op"), impls = uniq("impl");
  const head = `<thead><tr><th>kernel</th><th>torch</th>${impls.map(b => `<th>${b}</th>`).join("")}</tr></thead>`;
  const body = ops.map(op => {
    const win = b => {
      const w = WINS[op + "|" + b] || 0;
      return {cls: w ? " fastest" : "", t: w ? ` title="fastest on ${w} of ${MEASURED[op]} measured cases"` : ""};
    };
    const cells = impls.map(b => {
      const group = ROWS.filter(v => v.op === op && v.impl === b);
      const passed = group.filter(v => v.status === "pass").length;
      const mark = !passed ? "&#10008;" : passed === group.length ? "&#10004;" : "&#10004;*";
      const med = median(group.filter(v => v.sf != null).map(v => v.sf));
      const x = med != null ? ` <span class="dim">${med.toFixed(2)}&times;</span>` : "";
      const w = win(b);
      return `<td class="cell ${!passed ? "fail" : "good"}${w.cls}"${w.t}` +
        ` data-op="${op}" data-impl="${b}">${group.length ? mark + x : "&#10008;"}</td>`;
    }).join("");
    const w = win("torch");
    return `<tr><td>${op}</td><td class="good${w.cls}"${w.t}>&#10004;</td>${cells}</tr>`;
  }).join("");
  $("#matrix").innerHTML = head + `<tbody>${body}</tbody>`;
  document.querySelectorAll("#matrix td.cell").forEach(td => td.onclick = () => {
    filters.op = td.dataset.op; filters.impl = td.dataset.impl;
    $("#f-op").value = filters.op; $("#f-impl").value = filters.impl;
    renderTable();
    $("#cases").scrollIntoView({behavior: "smooth"});
  });
}

function filtered() {
  const q = filters.search.toLowerCase();
  let rows = ROWS.filter(v =>
    (!filters.op || v.op === filters.op) && (!filters.impl || v.impl === filters.impl) &&
    (!filters.dtype || v.dtype === filters.dtype) && (!filters.status || v.status === filters.status) &&
    (!q || (v.op + " " + v.case + " " + v.reason).toLowerCase().includes(q)));
  const key = sortKey || "severity";
  const val = v => key === "severity" ? SEVERITY[v.status] : v[key];
  rows.sort((a, b) => {
    const x = val(a), y = val(b);
    if (x == null && y == null) return 0;
    if (x == null) return 1;
    if (y == null) return -1;
    const c = x < y ? -1 : x > y ? 1 : (b.fwd || 0) - (a.fwd || 0);
    return c * sortDir;
  });
  return rows;
}

function detailRow(r) {
  const reason = r.reason || (r.bench_error ? "benchmark: " + r.bench_error : "");
  const gauges = dir => Object.entries(r[dir] || {}).map(([name, g]) =>
    `<tr><td>${dir === "fwd" ? name : "grad " + name}</td>` +
    `<td class="num">${fmt(g.err)}</td><td class="num">${fmt(g.budget)}</td>` +
    `<td class="num">${g.scale.toExponential(1)}</td><td class="num">${fmt(g.scale ? g.err / g.scale : g.err)}</td></tr>`).join("");
  const bench = Object.entries(r.bench || {}).map(([k, v]) =>
    `<tr><td>${k}</td><td class="num">${v}</td></tr>`).join("");
  return `<div class="box">
    <div><table><thead><tr><th>gauge</th><th class="num">err</th><th class="num">budget</th>
      <th class="num">scale</th><th class="num">err/scale</th></tr></thead>
      <tbody>${gauges("fwd")}${gauges("bwd")}</tbody></table></div>
    ${bench ? `<div><table><thead><tr><th>bench</th><th class="num"></th></tr></thead><tbody>${bench}</tbody></table></div>` : ""}
    <div class="dim">case_id ${r.case_id} &middot; ${r.impl} ${r.backend_version || ""} &middot; ${r.ts}
      ${reason ? `<br>${esc(reason)}` : ""}</div>
  </div>`;
}

function renderTable() {
  const rows = filtered(), cap = 1000;
  const counts = {pass: 0, skip: 0, fail: 0, crash: 0, error: 0};
  rows.forEach(v => counts[v.status]++);
  $("#counts").innerHTML = Object.entries(counts)
    .map(([s, n]) => `<span class="${s}">${n} ${s}</span>`).join("");
  $("#cases tbody").innerHTML = rows.slice(0, cap).map((v, i) => `
    <tr class="case" data-i="${i}">
      <td><span class="pill ${v.status}">${v.status}</span></td>
      <td>${v.op}</td><td>${v.impl}</td><td>${v.dtype}</td><td>${esc(v.case)}</td>
      <td class="num">${fmt(v.fwd)}</td><td class="num">${fmt(v.bwd)}</td>
      <td class="num">${v.fastest ? "&#9889; " : ""}${fx(v.sf)}</td><td class="num">${fx(v.sb)}</td><td class="num">${fx(v.mem)}</td>
      <td class="reason" title="${esc(v.reason)}">${esc(v.reason)}</td>
    </tr>`).join("");
  $("#truncated").textContent = rows.length > cap ? `showing first ${cap} of ${rows.length} rows; narrow the filters` : "";
  document.querySelectorAll("#cases tbody tr.case").forEach(tr => tr.onclick = () => {
    const open = tr.nextElementSibling?.classList.contains("detail");
    if (open) return tr.nextElementSibling.remove();
    const row = document.createElement("tr");
    row.className = "detail";
    row.innerHTML = `<td colspan="11">${detailRow(rows[tr.dataset.i].r)}</td>`;
    tr.after(row);
  });
}

fillSelect("#f-op", "op", "op");
fillSelect("#f-impl", "impl", "impl");
fillSelect("#f-dtype", "dtype", "dtype");
fillSelect("#f-status", "status", "status");
$("#f-search").oninput = e => { filters.search = e.target.value; renderTable(); };
$("#f-clear").onclick = () => {
  Object.keys(filters).forEach(k => filters[k] = "");
  document.querySelectorAll("#controls select, #controls input").forEach(el => el.value = "");
  sortKey = null; sortDir = 1;
  renderTable();
};
document.querySelectorAll("#cases thead th").forEach(th => th.onclick = () => {
  const k = th.dataset.k;
  sortDir = sortKey === k ? -sortDir : 1;
  sortKey = k;
  renderTable();
});

renderMeta();
renderMatrix();
renderTable();
</script>
</body>
</html>
"""
