/* LRSD figures for the Mem2Gen project page.
 *   'lrsd'         -> #viz-lrsd         storyboard (Qwen2.5-1.5B): after memorization, plain fine-tuning's two-hop accuracy
 *                                        stays flat while LRSD's keeps rising; memorization, CE loss and relMSE underneath.
 *   'lrsd-models'  -> #viz-lrsd-models  the four models as small multiples.
 * Every number is computed from window.M2G_DATA.lrsd at mount and checked against its `summary` (console.warn on mismatch).
 * Plain JS, no dependencies beyond js/common.js (window.M2G).
 */
(function () {
  "use strict";
  const M = window.M2G;
  if (!M) { console.error("lrsd.js: js/common.js must be loaded first"); return; }

  const NS = "http://www.w3.org/2000/svg";
  const ARMS = ["B", "L01", "L1"];
  const NAME = { B: "Plain fine-tuning", L01: "LRSD λ = 0.1", L1: "LRSD λ = 1" };
  const SHORT = { B: "Plain", L01: "λ = 0.1", L1: "λ = 1" };
  const TOKEN = { B: "--arm-base", L01: "--arm-l01", L1: "--arm-l1" };
  const HERO_KEY = "qwen2.5-1.5b";
  let UID = 0;

  /* ------------------------------------------------------------------ helpers */
  function h(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }
  function sv(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    if (attrs) for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function stext(parent, x, y, str, cls, anchor) {
    const t = sv("text", { x: x, y: y, class: cls || "" }, parent);
    if (anchor) t.setAttribute("text-anchor", anchor);
    t.textContent = str;
    return t;
  }
  function key(arm, extra) { return h("span", "lr-key lr-bg" + arm + (extra ? " " + extra : "")); }
  const sum = (a) => a.reduce((p, c) => p + c, 0);
  const mean = (a) => sum(a) / a.length;
  const sdev = (a) => { const m = mean(a); return a.length > 1 ? Math.sqrt(sum(a.map((v) => (v - m) * (v - m))) / (a.length - 1)) : 0; };
  const f3 = (v) => v.toFixed(3);
  const f2 = (v) => v.toFixed(2);
  const pct = (v) => Math.round(v * 100);
  const signed = (v, d) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(d);
  const ramp = (t, a, b) => M.ease(M.clamp((t - a) / (b - a), 0, 1));
  const ep = (e) => Math.floor(e + 1e-6);   // the one integer epoch shown by the readout, grid counters, tooltips and aria text
  const warn = (msg) => console.warn("lrsd.js: " + msg);
  const setOp = (node, v) => { const s = v <= 0.001 ? "0" : v >= 0.999 ? "1" : v.toFixed(3); if (node.__op !== s) { node.__op = s; node.setAttribute("opacity", s); } };
  function interp(arr, e) {            // linear interpolation of a per-epoch series at fractional epoch e
    const i = Math.floor(e), j = Math.min(arr.length - 1, i + 1), w = e - i;
    const a = arr[i], b = arr[j];
    if (a == null) return b;
    if (b == null) return a;
    return a + (b - a) * w;
  }
  function rgbOf(str) {                // '#rrggbb' | 'rgb(...)' -> [r,g,b]
    str = (str || "").trim();
    if (str[0] === "#") {
      if (str.length === 4) str = "#" + str[1] + str[1] + str[2] + str[2] + str[3] + str[3];
      return [1, 3, 5].map((i) => parseInt(str.slice(i, i + 2), 16));
    }
    const m = str.match(/[\d.]+/g);
    return m ? m.slice(0, 3).map(Number) : [128, 128, 128];
  }
  function dodge(items, gap, lo, hi) { // items [{y}] -> sets .ly so neighbours are >= gap apart, inside [lo, hi]
    const it = items.slice().sort((a, b) => a.y - b.y);
    it.forEach((d) => { d.ly = d.y; });
    for (let pass = 0; pass < 4; pass++) {
      for (let i = 1; i < it.length; i++) if (it[i].ly - it[i - 1].ly < gap) {
        const mid = (it[i].ly + it[i - 1].ly) / 2;
        it[i - 1].ly = mid - gap / 2; it[i].ly = mid + gap / 2;
      }
      if (it.length) {
        const top = it[0].ly - lo; if (top < 0) it.forEach((d) => { d.ly -= top; });
        const bot = it[it.length - 1].ly - hi; if (bot > 0) it.forEach((d) => { d.ly -= bot; });
      }
    }
    return items;
  }
  const ICON = {
    play: "M4.5 2.8v10.4c0 .5.5.8 1 .5l8.3-5.2c.4-.2.4-.8 0-1L5.5 2.3c-.5-.3-1 0-1 .5z",
    pause: "M4 2.5h2.6v11H4zM9.4 2.5H12v11H9.4z",
    replay: "M8 2.5a5.5 5.5 0 1 1-5.2 3.7l1.5.5A3.9 3.9 0 1 0 8 4.1v2.1L4.9 3.3 8 .4z",
  };
  function iconBtn(label, path) {
    const b = h("button", "m2g-iconbtn lr-ibtn");
    b.type = "button";
    b.setAttribute("aria-label", label);
    const s = sv("svg", { viewBox: "0 0 16 16", "aria-hidden": "true", focusable: "false" }, b);
    b.__path = sv("path", { d: path }, s);
    return b;
  }

  /* ------------------------------------------------------------------ styles (component-scoped fallbacks) */
  const CSS = `
.lr-root{--lr-sans:var(--font-sans,Inter,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif);--lr-serif:var(--font-serif,"Source Serif 4",Georgia,serif);position:relative;font-family:var(--lr-sans);color:var(--ink);text-align:left}
.lr-root *,.lr-root *::before,.lr-root *::after{box-sizing:border-box}
.lr-card{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:18px 22px 14px}
.lr-small .lr-card{padding:14px 12px 12px;border-radius:12px}
.lr-card:focus-visible{outline:2px solid var(--focus);outline-offset:3px}
.lr-root button:focus-visible,.lr-root select:focus-visible,.lr-root input:focus-visible,.lr-root summary:focus-visible{outline:2px solid var(--focus);outline-offset:2px}
.lr-top{display:flex;align-items:center;gap:14px;margin:0 0 12px}
.lr-chips{display:flex;gap:6px;flex:1 1 auto;min-width:0;overflow-x:auto;scrollbar-width:none;padding:2px;margin:-2px}
.lr-chips::-webkit-scrollbar{display:none}
.lr-chips.lr-ovf{-webkit-mask-image:linear-gradient(90deg,transparent 0,#000 14px,#000 calc(100% - 22px),transparent);mask-image:linear-gradient(90deg,transparent 0,#000 14px,#000 calc(100% - 22px),transparent)}
.lr-chip{flex:none;appearance:none;-webkit-appearance:none;font:500 12.5px/1 var(--lr-sans);color:var(--ink-2);background:transparent;border:1px solid var(--border);border-radius:999px;padding:7px 12px 7px 10px;cursor:pointer;white-space:nowrap;transition:background-color .15s,color .15s,border-color .15s}
.lr-chip:hover{border-color:var(--axis);color:var(--ink)}
.lr-chip .lr-n{color:var(--muted);font-weight:600;margin-right:6px;font-variant-numeric:tabular-nums}
.lr-chip[aria-pressed="true"]{background:var(--ink);border-color:var(--ink);color:var(--surface)}
.lr-chip[aria-pressed="true"] .lr-n{color:inherit;opacity:.6}
.lr-readout{flex:none;font-size:13px;color:var(--muted);font-variant-numeric:tabular-nums;white-space:nowrap}
.lr-readout b{font-weight:600;color:var(--ink)}
.lr-legend{display:flex;flex-wrap:wrap;align-items:center;gap:6px 18px;font-size:13px;color:var(--ink-2);margin:0 0 6px}
.lr-li{display:inline-flex;align-items:center;white-space:nowrap}
.lr-key{display:inline-block;flex:none;width:16px;height:2px;border-radius:1px;margin-right:7px}
.lr-key.lr-k-dash{height:0;background:none;border-top:1.5px dashed var(--ink-2)}
.lr-key.lr-k-band{height:10px;width:14px;border-radius:2px;background:var(--ink);opacity:.08}
.lr-bgB{background:var(--arm-base)}.lr-bgL01{background:var(--arm-l01)}.lr-bgL1{background:var(--arm-l1)}
.lr-formula{margin-left:auto;display:inline-flex;align-items:baseline;font-size:12.5px;color:var(--ink-2);background:var(--surface-2);border:1px solid var(--border);border-radius:8px;padding:4px 10px 5px;white-space:nowrap;cursor:help;transform-origin:50% 50%;font-variant-numeric:tabular-nums}
.lr-formula .lr-fl{color:var(--muted);margin-right:6px}
.lr-formula i{font-family:var(--lr-serif);font-size:14.5px;color:var(--ink);padding:0 1px}
.lr-formula sup{font-size:9.5px;line-height:0;position:relative;top:-1px;margin-right:1px}
.lr-narrow .lr-formula{margin-left:0}
.lr-body{display:grid;grid-template-columns:minmax(0,1fr) 316px;gap:0 32px;align-items:start}
.lr-narrow .lr-body{display:block}
.lr-chart{position:relative;min-width:0}
.lr-root svg{display:block;overflow:visible}
.lr-root svg text{font-family:var(--lr-sans)}
.lr-side{position:relative;min-width:0;display:flex;flex-direction:column;gap:22px;padding-top:4px}
.lr-narrow .lr-side{margin-top:18px;padding-top:16px;border-top:1px solid var(--grid)}
.lr-narrow .lr-side{display:grid;grid-template-columns:minmax(0,1fr);gap:22px}
.lr-mid .lr-side{grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:0 32px}
.lr-root .lr-h{font-size:12px;font-weight:600;color:var(--ink-2);margin:0 0 2px;line-height:1.35}
.lr-root .lr-sub{font-size:11.5px;color:var(--muted);margin:0 0 10px;line-height:1.4}
.lr-grids{display:flex;gap:16px;border-radius:6px;width:max-content;max-width:100%}
.lr-grids:focus{outline:none}
.lr-grids:focus-visible{outline:2px solid var(--focus);outline-offset:4px}
.lr-gcell{flex:none}
.lr-gname{font-size:12px;color:var(--ink-2);margin:0 0 6px;display:flex;align-items:center;white-space:nowrap}
.lr-gcount{margin-top:6px;font-size:11.5px;color:var(--muted);line-height:1.35;font-variant-numeric:tabular-nums}
.lr-gcount b{display:block;font-size:13px;color:var(--ink);font-weight:600}
.lr-root .lr-note{font-size:12px;line-height:1.45;color:var(--ink-2);margin:10px 0 0}
.lr-gk{white-space:nowrap}
.lr-sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin:0 4px 0 1px;vertical-align:-1px}
.lr-sw0{background:color-mix(in srgb,var(--ink) 50%,var(--surface))}
.lr-sw1{background:color-mix(in srgb,var(--arm-l1) 45%,var(--surface))}
.lr-sw2{background:color-mix(in srgb,var(--arm-base) 40%,var(--surface))}
.lr-note q{font-style:italic;quotes:"\\201C" "\\201D"}
.lr-root .lr-cap{font-family:var(--lr-serif);font-size:16px;line-height:1.5;color:var(--ink);margin:14px 0 12px}
.lr-root.lr-narrow .lr-cap{margin:12px 0 0}
.lr-narrow .lr-controls{margin-top:10px}
.lr-cap strong{font-weight:600}
.lr-tick{font-size:11px;fill:var(--muted);font-variant-numeric:tabular-nums}
.lr-ptitle{font-size:12px;font-weight:600;fill:var(--ink-2)}
.lr-ptitle .lr-m{font-weight:400;fill:var(--muted)}
.lr-lab{font-size:12px;font-weight:600;fill:var(--ink);font-variant-numeric:tabular-nums}
.lr-lab2{font-size:10.5px;fill:var(--muted)}
.lr-ann{font-size:11px;fill:var(--ink-2)}
.lr-annm{font-size:11px;fill:var(--muted)}
.lr-halo{paint-order:stroke;stroke:var(--surface);stroke-width:4px;stroke-linejoin:round}
.lr-big{font-size:20px;font-weight:600;fill:var(--ink);letter-spacing:-.01em}
.lr-small .lr-big{font-size:15px}
.lr-offstub{stroke:var(--ink-2);stroke-width:1.25;fill:none;stroke-dasharray:2.5 2.5}
.lr-offstub.lr-solid{stroke-dasharray:none;stroke-linecap:round;stroke-linejoin:round}
.lr-break{stroke:var(--ink-2);stroke-width:1.25;fill:none}
.lr-breakgap{fill:var(--surface);stroke:none}
.lr-on01{fill:var(--arm-l01);font-weight:600}
.lr-offlab{font-size:12px;font-weight:600;fill:var(--ink-2);font-variant-numeric:tabular-nums}
.lr-gl{stroke:var(--grid);stroke-width:1;fill:none;shape-rendering:crispEdges}
.lr-base{stroke:var(--axis)}
.lr-line{fill:none;stroke-width:2;stroke-linecap:round;stroke-linejoin:round}
.lr-thin{stroke-width:1.5}
.lr-raw{stroke-width:1.1;stroke-opacity:.4}
.lr-k-sm{display:inline-block;flex:none;width:18px;height:10px;margin-right:7px;color:var(--ink-2)}
.lr-li-sm{color:var(--muted);font-size:12.5px}
.lr-band{stroke:none;fill-opacity:.12}
.lr-wedge{stroke:none;fill-opacity:.2}
.lr-ruler{stroke-width:1.5;stroke-dasharray:.1 3.4;stroke-linecap:round;fill:none}
.lr-wall{stroke:var(--ink-2);stroke-width:1.5;stroke-dasharray:5 4;fill:none}
.lr-evband{fill:var(--ink);fill-opacity:.05}
.lr-bracket{stroke:var(--ink);stroke-width:1.5;fill:none;stroke-linecap:round}
.lr-leader{stroke:var(--muted);stroke-width:1;fill:none}
.lr-cursor{stroke:var(--ink-2);stroke-width:1;stroke-opacity:.3;shape-rendering:crispEdges}
.lr-hair{stroke:var(--ink-2);stroke-width:1;stroke-opacity:.8;shape-rendering:crispEdges}
.lr-ring{stroke:var(--surface);stroke-width:2}
.lr-target{fill:var(--surface);stroke:var(--ink-2);stroke-width:1.5}
.lr-track{stroke:var(--grid);stroke-width:1}
.lr-sB{stroke:var(--arm-base)}.lr-sL01{stroke:var(--arm-l01)}.lr-sL1{stroke:var(--arm-l1)}
.lr-fB{fill:var(--arm-base)}.lr-fL01{fill:var(--arm-l01)}.lr-fL1{fill:var(--arm-l1)}
.lr-hit{fill:transparent;cursor:crosshair}
.lr-tip{position:absolute;z-index:6;pointer-events:none;display:none;background:var(--tip-bg,var(--surface));color:var(--ink);border:1px solid var(--border);border-radius:8px;padding:8px 10px;font-size:12px;line-height:1.4;box-shadow:var(--tip-shadow,0 6px 20px rgba(0,0,0,.14));max-width:min(340px,calc(100vw - 32px))}
.lr-tip .lr-tt{font-weight:600;color:var(--ink);margin-bottom:4px;font-size:12px}
.lr-tip table{border-collapse:collapse;font-variant-numeric:tabular-nums}
.lr-tip th{font-weight:500;color:var(--ink-2);text-align:right;padding:0 0 3px 12px;white-space:nowrap;font-size:11.5px}
.lr-tip th:first-child,.lr-tip td:first-child{text-align:left;padding-left:0}
.lr-tip td{padding:1px 0 1px 12px;text-align:right;white-space:nowrap;color:var(--ink-2)}
.lr-tip td:first-child{color:var(--muted);font-size:11.5px}
.lr-tip td b{font-weight:600;color:var(--ink)}
.lr-tip td.lr-seeds{font-size:10.5px;color:var(--muted);white-space:pre;line-height:1.25}
.lr-tip .lr-tfoot{margin-top:4px;font-size:10.5px;color:var(--muted);white-space:normal}
.lr-tip .lr-q{font-size:12px;color:var(--ink);white-space:normal;margin-bottom:3px;line-height:1.35}
.lr-tip .lr-a{font-size:11.5px;color:var(--ink-2);white-space:normal;margin-bottom:6px}
.lr-tip .lr-bc{display:flex;align-items:center;gap:6px;font-size:11px;color:var(--ink-2);margin-top:2px;white-space:nowrap}
.lr-tip .lr-bc span:first-child{width:46px;flex:none}
.lr-tip .lr-bc b{font-weight:600;color:var(--ink)}
.lr-tip .lr-bc svg{flex:none}
.lr-controls{display:flex;align-items:center;gap:12px;min-height:36px}
.lr-small .lr-controls{flex-wrap:wrap;row-gap:10px}
.lr-ibtn{flex:none}
:where(.lr-ibtn){appearance:none;-webkit-appearance:none;width:34px;height:34px;border-radius:999px;border:1px solid var(--border);background:var(--surface-2);color:var(--ink);display:inline-grid;place-items:center;cursor:pointer;padding:0}
:where(.lr-ibtn):hover{border-color:var(--axis)}
.lr-ibtn svg{width:14px;height:14px;fill:currentColor;display:block}
.lr-range{flex:1 1 auto;min-width:60px}
:where(.lr-range){appearance:none;-webkit-appearance:none;height:22px;background:transparent;margin:0;cursor:pointer;--lr-p:0%}
:where(.lr-range)::-webkit-slider-runnable-track{height:4px;border-radius:2px;background:linear-gradient(var(--ink-2),var(--ink-2)) 0 0/var(--lr-p) 100% no-repeat,var(--grid)}
:where(.lr-range)::-webkit-slider-thumb{-webkit-appearance:none;width:14px;height:14px;border-radius:50%;background:var(--ink);border:2px solid var(--surface);margin-top:-5px;box-shadow:0 0 0 1px var(--border)}
:where(.lr-range)::-moz-range-track{height:4px;border-radius:2px;background:var(--grid)}
:where(.lr-range)::-moz-range-progress{height:4px;border-radius:2px;background:var(--ink-2)}
:where(.lr-range)::-moz-range-thumb{width:12px;height:12px;border-radius:50%;background:var(--ink);border:2px solid var(--surface)}
.lr-small .lr-range{flex-basis:calc(100% - 46px)}
:where(.lr-seg){display:inline-flex;flex:none;padding:2px;border-radius:999px;background:var(--surface-2);border:1px solid var(--border)}
:where(.lr-seg) button{appearance:none;-webkit-appearance:none;border:0;background:transparent;color:var(--ink-2);font:500 12px/1 var(--lr-sans);padding:7px 10px;border-radius:999px;cursor:pointer;font-variant-numeric:tabular-nums}
:where(.lr-seg) button[aria-pressed="true"]{background:var(--surface);color:var(--ink);box-shadow:0 1px 3px rgba(0,0,0,.14)}
.lr-speedsel{display:none;flex:none;font:500 13px var(--lr-sans);color:var(--ink);background:var(--surface-2);border:1px solid var(--border);border-radius:8px;padding:6px 8px;height:34px}
.lr-small .lr-seg{display:none}.lr-small .lr-speedsel{display:inline-block}
.lr-ctl-lab{font-size:12px;color:var(--muted);flex:none}
details.lr-data{margin-top:12px;font-size:13px;color:var(--ink-2)}
details.lr-data>summary{cursor:pointer;width:max-content;max-width:100%;font-size:13px;color:var(--ink-2)}
.lr-tablewrap{overflow-x:auto;margin-top:8px;-webkit-overflow-scrolling:touch}
.lr-table{border-collapse:collapse;font-size:12.5px;font-variant-numeric:tabular-nums;width:100%;color:var(--ink)}
.lr-table caption{caption-side:top;text-align:left;font-size:12px;color:var(--muted);padding:0 0 6px}
.lr-table th,.lr-table td{padding:4px 10px;text-align:right;border-bottom:1px solid var(--grid);white-space:nowrap}
.lr-table thead th{font-size:11.5px;font-weight:600;color:var(--ink-2);border-bottom:1px solid var(--axis)}
.lr-table .lr-tl{text-align:left}
.lr-table tbody tr.lr-grp td{border-top:1px solid var(--axis)}
.lr-sr{position:absolute!important;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap;border:0}
.lr-mgrid{display:grid;gap:22px 24px}
.lr-mpanel{min-width:0;position:relative}
.lr-root .lr-mtitle{font-size:13px;font-weight:600;color:var(--ink);margin:0 0 2px;line-height:1.3}
.lr-mtitle span{font-weight:400;color:var(--muted)}
.lr-mmeta{font-size:11.5px;color:var(--muted);line-height:1.45;margin-top:2px;font-variant-numeric:tabular-nums}
.lr-mmeta .lr-mrow{display:flex;flex-wrap:wrap;align-items:center;column-gap:10px}
.lr-mmeta .lr-ml{white-space:nowrap}
.lr-mmeta .lr-mv{display:inline-flex;align-items:center;color:var(--ink-2);white-space:nowrap}
.lr-mmeta .lr-mv b{font-weight:600;color:var(--ink)}
.lr-mmeta .lr-key{width:10px;margin-right:4px}
.lr-root .lr-foot{font-size:12.5px;line-height:1.5;color:var(--ink-2);margin:16px 0 0}
.lr-mctl{display:flex;align-items:center;gap:12px;margin-top:12px;flex-wrap:wrap}
.lr-mctl details.lr-data{margin-top:0;flex:1 1 100%}
`;
  function injectStyle() {
    if (document.getElementById("lr-style")) return;
    const st = document.createElement("style");
    st.id = "lr-style";
    st.textContent = CSS;
    document.head.appendChild(st);
  }

  /* ------------------------------------------------------------------ data preparation */
  function armStats(seeds) {
    const E = seeds[0].chain.length - 1;
    const col = (k, e) => seeds.map((s) => s[k][e]);
    const st = { chain: [], sd: [], seedChain: [], ce: [null], lr: [null], rel: [], mem: [], seedIds: seeds.map((s) => s.seed) };
    for (let e = 0; e <= E; e++) {
      const c = col("chain", e);
      st.seedChain.push(c);
      st.chain.push(mean(c));
      st.sd.push(sdev(c));
      st.rel.push(mean(col("relmse", e)));
      if (e > 0) { st.ce.push(mean(col("ce", e))); st.lr.push(mean(col("lrsd", e))); }
    }
    for (let k = 0; k < seeds[0].mem.length; k++) st.mem.push(mean(col("mem", k)));
    st.seedLate = seeds.map((s) => mean(s.chain.slice(30, E + 1)));
    st.late = mean(st.chain.slice(30, E + 1));
    st.lateSd = sdev(st.seedLate);
    return st;
  }

  function prepModel(m, summ) {
    const A = {};
    ARMS.forEach((X) => { A[X] = armStats(m.arms[X]); });
    const E = m.arms.B[0].chain.length - 1;
    const nMem = A.B.mem.length, memStep = E / (nMem - 1);
    let wallK = -1;
    for (let k = 0; k < nMem; k++) if (ARMS.every((X) => A[X].mem[k] >= 0.9)) { wallK = k; break; }
    if (wallK < 0) { warn(m.key + ": no epoch with >= 90% memorization in every arm"); wallK = Math.round(10 / memStep); }
    const wall = wallK * memStep;
    // Per-epoch 3-seed means are noisy (~0.01) next to the post-memorization gains, so from the wall on the chart draws a
    // centred 5-epoch rolling mean (window truncated at epoch E), tapered in over the two epochs before the wall so the
    // line stays continuous; at the wall it equals the ruler (mean of epochs wall-2 .. wall+2).
    const smooth = (arr) => arr.map((_, e) => {
      const k = M.clamp(e - (wall - 2), 0, 2);
      return mean(arr.slice(Math.max(0, e - k), Math.min(E, e + k) + 1));
    });
    ARMS.forEach((X) => {
      const st = A[X];
      st.sm = smooth(st.chain); st.smSd = smooth(st.sd);
      st.ruler = mean(st.chain.slice(Math.max(0, wall - 2), wall + 3));
      st.memEnd = st.mem[nMem - 1];
      st.memWall = st.mem[wallK];
      const sm = summ && summ[X];
      if (!sm) { warn("summary missing for " + m.key + " " + X); return; }
      if (Math.abs(st.late - sm.late) > 1e-3) warn(`${m.key} ${X}: late ${st.late} != summary ${sm.late}`);
      if (sm.late_sd != null && Math.abs(st.lateSd - sm.late_sd) > 1e-3) warn(`${m.key} ${X}: late_sd ${st.lateSd} != summary ${sm.late_sd}`);
      if (Math.abs(st.memEnd - sm.mem) > 1e-3) warn(`${m.key} ${X}: final memorization ${st.memEnd} != summary ${sm.mem}`);
    });
    A.L01.gain = pct(A.L01.late / A.B.late - 1);
    A.L1.gain = pct(A.L1.late / A.B.late - 1);
    const pr = String(m.pair).split("-").map(Number);
    return { key: m.key, name: m.name, src: pr[0], tgt: pr[1], nL: m.n_layers, arms: A, E, wall, wallK, memStep, nQ: summ ? summ.n_q : null };
  }

  function signFlip(d) {               // exact two-sided sign-flip test on paired differences
    const n = d.length, obs = Math.abs(sum(d)) - 1e-12;
    let c = 0;
    for (let mask = 0; mask < (1 << n); mask++) {
      let s = 0;
      for (let i = 0; i < n; i++) s += (mask >> i) & 1 ? -d[i] : d[i];
      if (Math.abs(s) >= obs) c++;
    }
    return c / (1 << n);
  }

  function oracleFor(D, name) {
    try {
      const o = D.results.oracle, ci = o.cols.indexOf("prime_pat"), row = o.rows.find((r) => r[0] === name);
      return row && ci >= 0 && typeof row[ci] === "number" ? row[ci] : null;
    } catch (e) { return null; }
  }

  /* ------------------------------------------------------------------ storyboard */
  const DUR = 34;                      // seconds at 1x
  const SCENES = [
    { id: "S0", ch: 1, t0: 0, t1: 0.04 },
    { id: "S1", ch: 1, t0: 0.04, t1: 0.22, sweep: "both" },
    { id: "S2", ch: 2, t0: 0.22, t1: 0.31 },
    { id: "S3", ch: 3, t0: 0.31, t1: 0.49, sweep: "B" },
    { id: "S4a", ch: 4, t0: 0.49, t1: 0.53 },
    { id: "S4b", ch: 4, t0: 0.53, t1: 0.73, sweep: "L" },
    { id: "S5", ch: 5, t0: 0.73, t1: 0.84 },
    { id: "S6", ch: 6, t0: 0.84, t1: 1.0 },
  ];
  const CHAPTERS = [["Learn", 0], ["Memorized", 0.22], ["Plain alone", 0.31], ["With LRSD", 0.49], ["Inside", 0.73], ["Result", 0.84]];
  const KEYFRAME = [0.215, 0.305, 0.488, 0.728, 0.838, 1];   // chapter key frames (reduced motion)
  function sceneAt(t) { for (let i = SCENES.length - 1; i >= 0; i--) if (t >= SCENES[i].t0) return i; return 0; }
  function sweepEase(p) {              // linear in epoch, eased over the first and last 6%
    const a = 0.06, v = 1 / (1 - a);
    p = M.clamp(p, 0, 1);
    if (p < a) return (v * p * p) / (2 * a);
    if (p > 1 - a) return 1 - (v * (1 - p) * (1 - p)) / (2 * a);
    return v * (p - a / 2);
  }

  /* ================================================================== #viz-lrsd */
  function Hero(el, D) {
    const L = D.lrsd;
    const mRaw = L.models.find((m) => m.key === HERO_KEY) || L.models[0];
    const P = prepModel(mRaw, L.summary[mRaw.key]);
    const A = P.arms, E = P.E, W10 = P.wall;
    const ora = oracleFor(D, P.name);
    const id = "lr" + UID++;

    /* ---- per-question data (one seed) ---- */
    const Q = L.questions, N = Q.question.length;
    if (P.nQ != null && P.nQ !== N) warn(`n_q ${P.nQ} != ${N} questions`);
    const okBits = {};
    ["B", "L1"].forEach((X) => {
      okBits[X] = Q.ok[X].map((s) => { const a = new Uint8Array(N); for (let i = 0; i < N; i++) a[i] = s.charCodeAt(i) === 49 ? 1 : 0; return a; });
    });
    const lateCount = {};
    ["B", "L1"].forEach((X) => { lateCount[X] = new Array(N).fill(0); for (let e = 30; e <= E; e++) for (let i = 0; i < N; i++) lateCount[X][i] += okBits[X][e][i]; });
    const nLate = E - 30 + 1, need = Math.ceil(nLate / 2);
    const stable = (X, i) => lateCount[X][i] >= need;
    const group = (i) => (stable("B", i) && stable("L1", i) ? 0 : stable("L1", i) ? 1 : stable("B", i) ? 2 : 3);
    const order = [...Array(N).keys()].sort((a, b) => group(a) - group(b) || a - b);
    const posOf = new Array(N); order.forEach((i, p) => { posOf[i] = p; });
    const gCount = [0, 0, 0, 0]; for (let i = 0; i < N; i++) gCount[group(i)]++;
    const cands = [];
    for (let i = 0; i < N; i++) {
      if (lateCount.L1[i] !== nLate) continue;
      let never = true; for (let e = 0; e <= E; e++) if (okBits.B[e][i]) { never = false; break; }
      if (!never) continue;
      let first = 0; while (!okBits.L1[first][i]) first++;
      cands.push({ i, first });
    }
    cands.sort((a, b) => b.first - a.first || a.i - b.i);
    const ex = cands[0] || null;

    /* ---- numbers for captions (all computed) ---- */
    const r = ARMS.map((X) => A[X]);
    const ce1 = r.map((s) => s.ce[1]), ceW = r.map((s) => s.ce[W10]);
    const ceLo = f2(Math.min(...ceW)), ceHi = f2(Math.max(...ceW));
    const bMemAfter = mean(A.B.mem.slice(P.wallK + 1));
    const bCeAfter = mean(A.B.ce.slice(W10 + 1));
    const bRelAfter = mean(A.B.rel.slice(W10));
    const l1RelAfter = mean(A.L1.rel.slice(2 * W10));
    const fixed = (v) => (v < 0.01 ? v.toFixed(3) : v.toFixed(2));
    const gainAfter = {}; ARMS.forEach((X) => { gainAfter[X] = A[X].late - A[X].ruler; });
    const allUp = ARMS.every((X) => gainAfter[X] >= 0), gTxt = (X) => (allUp ? f3(gainAfter[X]) : signed(gainAfter[X], 3));
    const riseTxt = `${allUp ? "rises by" : "changes by"} ${gTxt("L01")} with λ = 0.1 and ${gTxt("L1")} with λ = 1, against ${gTxt("B")} for plain fine-tuning`;
    const memLo = Math.min(...ARMS.map((X) => A[X].memEnd)), memHi = Math.max(...ARMS.map((X) => A[X].memEnd));
    const S = {
      model: P.name, src: P.src, tgt: P.tgt, seeds: A.B.seedIds.length,
    };
    const CAP = [
      ["Setup.", `${S.model} is fine-tuned for ${E} epochs on single-hop facts only (${S.seeds} seeds per arm); the ${N} two-hop questions built from them are never trained on. LRSD adds one loss term that pulls the head entity’s layer-${S.tgt} state toward its own layer-${S.src} state.`],
      ["1 · Learn.", `Epochs 0–${W10}: every arm learns the facts, and the cross-entropy loss falls from ${f2(mean(ce1))} to ${ceLo}–${ceHi}. With λ = 1 the two states align almost at once (relMSE panel and “Inside the network”).`],
      ["2 · Memorized.", `Epoch ${W10}: memorization is at least 90% in every arm (${ARMS.map((X) => pct(A[X].memWall) + "%").join(" / ")}, mean of ${S.seeds} seeds). Dotted lines mark each arm’s two-hop level at this point (mean of epochs ${W10 - 2}–${W10 + 2}): ${f3(A.B.ruler)} plain, ${f3(A.L01.ruler)} with λ = 0.1, and already ${f3(A.L1.ruler)} with λ = 1.`],
      ["3 · Plain fine-tuning alone.", `${E - W10} more epochs: memorization ≈ ${f2(bMemAfter)}, cross-entropy loss ≈ ${f2(bCeAfter)}, and two-hop accuracy only ${f3(A.B.ruler)} → ${f3(A.B.late)} (mean of epochs 30–${E}). Its layer-${S.tgt} state stays far from its layer-${S.src} state (relMSE ≈ ${f2(bRelAfter)}).`],
      ["4 · With LRSD.", `Now the same ${E - W10} epochs with LRSD.`],
      ["4 · With LRSD.", `From epoch ${W10} (dotted) to the mean of epochs 30–${E}, two-hop accuracy ${riseTxt}. Memorization ends between ${f3(memLo)} and ${f3(memHi)} in every arm.`],
      ["5 · Inside.", `Measured on two-hop prompts that LRSD never trains on, the two states become nearly identical (relMSE ≈ ${fixed(l1RelAfter)} with λ = 1, ≈ ${f2(bRelAfter)} without LRSD). With λ = 0.1 they align slowly, over the same epochs in which its accuracy rises. This is a correlation, not a controlled test.`],
      ["6 · Result.", `Mean over epochs 30–${E}, ${S.seeds} seeds: ${f3(A.B.late)} → ${f3(A.L1.late)} (+${A.L1.gain}%) with λ = 1 and ${f3(A.L01.late)} (+${A.L01.gain}%) with λ = 0.1; memorization at epoch ${E}: ${f3(A.B.memEnd)} vs ${f3(A.L1.memEnd)}.` + (ora != null ? ` Oracle self-patching, a diagnostic upper bound, reaches ${f3(ora)}.` : "")],
    ];
    const capIndex = [0, 1, 2, 3, 4, 5, 6, 7];   // scene index -> caption index

    /* ---- DOM ---- */
    injectStyle();
    el.textContent = "";
    const root = h("div", "lr-root lr-hero"); el.appendChild(root);
    const card = h("div", "lr-card"); card.tabIndex = 0;
    card.setAttribute("role", "group");
    card.setAttribute("aria-roledescription", "animated figure");
    card.setAttribute("aria-label", `LRSD on ${P.name}: two-hop accuracy, memorization, cross-entropy loss and relMSE over ${E} epochs. Keys: space plays or pauses, arrows step one epoch, 1 to 6 jump to a chapter.`);
    root.appendChild(card);

    const top = h("div", "lr-top"); card.appendChild(top);
    const chipsEl = h("div", "lr-chips"); chipsEl.setAttribute("role", "group"); chipsEl.setAttribute("aria-label", "Chapters");
    top.appendChild(chipsEl);
    const chips = CHAPTERS.map((c, i) => {
      const b = h("button", "lr-chip"); b.type = "button";
      b.appendChild(h("span", "lr-n", String(i + 1))); b.appendChild(document.createTextNode(c[0]));
      b.setAttribute("aria-pressed", "false");
      b.addEventListener("click", () => jumpChapter(i));
      chipsEl.appendChild(b);
      return b;
    });
    const readout = h("div", "lr-readout m2g-readout"); readout.setAttribute("aria-hidden", "true"); top.appendChild(readout);

    const legend = h("div", "lr-legend m2g-legend"); card.appendChild(legend);
    ARMS.forEach((X) => { const li = h("span", "lr-li"); li.appendChild(key(X)); li.appendChild(document.createTextNode(NAME[X])); legend.appendChild(li); });
    {
      const li = h("span", "lr-li lr-li-sm"), ks = sv("svg", { class: "lr-k-sm", viewBox: "0 0 18 10", "aria-hidden": "true", focusable: "false" }, li);
      sv("path", { d: "M0 6L3 3L6 8L9 2L12 7L15 4L18 6", fill: "none", stroke: "currentColor", "stroke-width": 1, "stroke-opacity": 0.45 }, ks);
      sv("path", { d: "M0 5H18", fill: "none", stroke: "currentColor", "stroke-width": 2.5, "stroke-linecap": "round" }, ks);
      li.appendChild(document.createTextNode(`After epoch ${W10}: 5-epoch mean (faint: each epoch)`));
      legend.appendChild(li);
    }
    const formula = h("span", "lr-formula");
    formula.appendChild(h("span", "lr-fl", "LRSD"));
    const fx = [["", "loss = CE + "], ["i", "λ"], ["", " · relMSE("], ["i", "h"], ["sup", String(P.tgt)], ["", ", sg("], ["i", "h"], ["sup", String(P.src)], ["", "))"]];
    fx.forEach(([tg, tx]) => formula.appendChild(tg ? h(tg, null, tx) : document.createTextNode(tx)));
    const fTitle = `For the head-entity tokens of every single-hop training prompt: relMSE = ‖h${P.tgt} − sg(h${P.src})‖² / ‖h${P.src}‖². sg = stop-gradient: the layer-${P.src} state is a fixed target.`;
    formula.title = fTitle;
    formula.setAttribute("aria-label", `LRSD loss = CE + lambda times relMSE of h at layer ${P.tgt} and stop-gradient h at layer ${P.src}. ${fTitle}`);
    legend.appendChild(formula);

    const body = h("div", "lr-body"); card.appendChild(body);
    const chart = h("div", "lr-chart"); body.appendChild(chart);
    const svg = sv("svg", { role: "img", "aria-label": `Line charts over ${E} epochs for plain fine-tuning and LRSD at lambda 0.1 and 1: two-hop accuracy, memorization, cross-entropy loss, relMSE. Values are in the data table below.` }, chart);
    const tip = h("div", "lr-tip m2g-tip"); tip.setAttribute("aria-hidden", "true"); chart.appendChild(tip);

    const side = h("div", "lr-side"); body.appendChild(side);
    const sBox = h("div", "lr-sbox"); side.appendChild(sBox);
    sBox.appendChild(h("p", "lr-h", `Inside the network: layer ${P.tgt} vs layer ${P.src}`));
    sBox.appendChild(h("p", "lr-sub", `Head-entity state at layer ${P.tgt} (dot) and at layer ${P.src} (ring), measured on the two-hop prompts, which LRSD never trains on.`));
    const svgS = sv("svg", { role: "img", "aria-label": `relMSE between the layer-${P.tgt} and layer-${P.src} head-entity states for each arm at the current epoch` }, sBox);
    const sTip = h("div", "lr-tip m2g-tip"); sTip.setAttribute("aria-hidden", "true"); side.appendChild(sTip);

    const gBox = h("div", "lr-gbox"); side.appendChild(gBox);
    gBox.appendChild(h("p", "lr-h", "Which two-hop questions get answered"));
    gBox.appendChild(h("p", "lr-sub", `Each square is one of the ${N} two-hop questions (seed ${Q.seed}), in the same place in both grids. Shade = share of the last 5 epochs in which it was answered.`));
    const grids = h("div", "lr-grids"); grids.tabIndex = 0;
    grids.setAttribute("role", "group");
    grids.setAttribute("aria-label", `Question grids for plain fine-tuning and LRSD lambda 1. Arrow keys move between questions.`);
    gBox.appendChild(grids);
    const gridArms = ["B", "L1"], cv = {}, gcount = {};
    gridArms.forEach((X) => {
      const c = h("div", "lr-gcell"); grids.appendChild(c);
      const nm = h("div", "lr-gname"); nm.appendChild(key(X)); nm.appendChild(document.createTextNode(NAME[X])); c.appendChild(nm);
      cv[X] = h("canvas"); cv[X].setAttribute("aria-hidden", "true"); cv[X].style.display = "block"; c.appendChild(cv[X]);
      gcount[X] = h("div", "lr-gcount"); c.appendChild(gcount[X]);
    });
    const note = h("p", "lr-note"); gBox.appendChild(note);
    const live = h("div", "lr-sr"); live.setAttribute("aria-live", "polite"); gBox.appendChild(live);

    const cap = h("p", "lr-cap"); cap.setAttribute("aria-live", "polite"); card.appendChild(cap);
    // captions are measured in this hidden twin, so resizing never rewrites (and re-announces) the live caption
    const capM = h("p", "lr-cap"); capM.setAttribute("aria-hidden", "true");
    capM.style.cssText = "position:absolute;left:0;top:0;visibility:hidden;pointer-events:none;margin:0";
    root.appendChild(capM);

    const ctl = h("div", "m2g-controls lr-controls"); card.appendChild(ctl);
    const bPlay = iconBtn("Play", ICON.play); bPlay.setAttribute("aria-pressed", "false"); ctl.appendChild(bPlay);
    const range = h("input", "m2g-range lr-range");
    Object.assign(range, { type: "range", min: 0, max: 1000, step: 1, value: 0 });
    range.setAttribute("aria-label", "Position in the animation"); ctl.appendChild(range);
    const seg = h("div", "m2g-seg lr-seg"); seg.setAttribute("role", "group"); seg.setAttribute("aria-label", "Speed"); ctl.appendChild(seg);
    const speeds = [0.5, 1, 2];
    const segBtns = speeds.map((v) => { const b = h("button", null, v + "×"); b.type = "button"; b.setAttribute("aria-pressed", v === 1 ? "true" : "false"); b.addEventListener("click", () => setSpeed(v)); seg.appendChild(b); return b; });
    const sel = h("select", "lr-speedsel"); sel.setAttribute("aria-label", "Speed");
    speeds.forEach((v) => { const o = h("option", null, v + "×"); o.value = v; if (v === 1) o.selected = true; sel.appendChild(o); });
    sel.addEventListener("change", () => setSpeed(Number(sel.value)));
    ctl.appendChild(sel);
    const bReplay = iconBtn("Replay from the start", ICON.replay); ctl.appendChild(bReplay);

    const det = h("details", "m2g-data lr-data"); card.appendChild(det);
    det.appendChild(h("summary", null, "Data table"));
    det.appendChild(buildHeroTable());

    /* ---- state ---- */
    let T = 0, speed = 1, wantPlay = false, hasPlayed = false, tween = null, G = null, R = null, lastW = -1, gridKey = "", hoverE = null, focusP = -1, themeV = 0;
    const clock = M.clock(tick);

    /* ---- data table ---- */
    function buildHeroTable() {
      const wrap = h("div", "lr-tablewrap");
      const tb = h("table", "lr-table");
      tb.appendChild(h("caption", null, `${P.name}, mean of ${A.B.seedIds.length} seeds (sd across seeds). Memorization is measured every ${P.memStep} epochs; relMSE is between the layer-${P.tgt} and layer-${P.src} head-entity states on the two-hop prompts.`));
      const th = h("thead"), trh = h("tr");
      ["Epoch", "Arm", "Two-hop (mean ± sd)", "Memorization", "CE loss", "relMSE"].forEach((c, i) => { const x = h("th", i < 2 ? "lr-tl" : null, c); x.scope = "col"; trh.appendChild(x); });
      th.appendChild(trh); tb.appendChild(th);
      const tbd = h("tbody");
      for (let e = 0; e <= E; e += P.memStep) {
        ARMS.forEach((X, j) => {
          const st = A[X], tr = h("tr", j === 0 && e > 0 ? "lr-grp" : null);
          tr.appendChild(h("td", "lr-tl", j === 0 ? String(e) : ""));
          tr.appendChild(h("td", "lr-tl", NAME[X]));
          tr.appendChild(h("td", null, `${f3(st.chain[e])} ± ${f3(st.sd[e])}`));
          tr.appendChild(h("td", null, f3(st.mem[e / P.memStep])));
          tr.appendChild(h("td", null, e === 0 ? "–" : f3(st.ce[e])));
          tr.appendChild(h("td", null, f3(st.rel[e])));
          tbd.appendChild(tr);
        });
      }
      ARMS.forEach((X, j) => {
        const st = A[X], tr = h("tr", j === 0 ? "lr-grp" : null);
        tr.appendChild(h("td", "lr-tl", j === 0 ? `30–${E}` : ""));
        tr.appendChild(h("td", "lr-tl", NAME[X]));
        tr.appendChild(h("td", null, `${f3(st.late)} ± ${f3(st.lateSd)}` + (X === "B" ? "" : ` (+${st.gain}%)`)));
        tr.appendChild(h("td", null, `${f3(st.memEnd)} (ep ${E})`));
        tr.appendChild(h("td", null, f3(mean(st.ce.slice(30)))));
        tr.appendChild(h("td", null, f3(mean(st.rel.slice(30)))));
        tbd.appendChild(tr);
      });
      tb.appendChild(tbd); wrap.appendChild(tb);
      return wrap;
    }

    /* ---- geometry ---- */
    function geometry(w, small, wide, extra) {
      const g = { w, small, ml: small ? 30 : 38, gut: small ? 62 : w < 640 ? 96 : 112 };
      g.pw = Math.max(120, w - g.ml - g.gut);
      g.X = (e) => g.ml + (e / E) * g.pw;
      g.Xinv = (x) => ((x - g.ml) / g.pw) * E;
      const hA = (small ? 196 : wide ? 286 : 250) + (extra || 0), hS = small ? 34 : 38, hD = small ? 54 : 58;
      const mk = (max, ticks, fmt, hh) => ({ max, ticks, fmt, h: hh });
      const PN = {
        A: mk(0.2, [0, 0.05, 0.1, 0.15, 0.2], (v) => (v === 0 ? "0" : v.toFixed(2)), hA),
        B: mk(1, [0, 0.5, 1], (v) => String(v), hS),
        C: mk(2, [0, 1, 2], (v) => String(v), hS),
        D: mk(0.5, [0, 0.25, 0.5], (v) => String(v), hD),
      };
      let y = 0;
      ["A", "B", "C", "D"].forEach((k, i) => {
        const p = PN[k];
        const tH = k === "A" ? (small ? 36 : 22) : 17;
        p.titleY = y + (k === "A" ? 12 : 11);
        y += tH; p.top = y; y += p.h; p.bot = y; y += i === 0 ? 18 : 12;
        p.y = (v) => p.bot - (M.clamp(v, 0, p.max) / p.max) * p.h;
      });
      g.P = PN;
      g.h = PN.D.bot + 38;
      return g;
    }

    /* ---- chart construction (on layout) ---- */
    function pathOf(pts) { return pts.map((p, i) => (i ? "L" : "M") + p[0].toFixed(1) + " " + p[1].toFixed(1)).join(""); }
    function buildChart() {
      const g = G, X = g.X, PN = g.P;
      svg.textContent = "";
      svg.setAttribute("width", g.w); svg.setAttribute("height", g.h); svg.setAttribute("viewBox", `0 0 ${g.w} ${g.h}`);
      const defs = sv("defs", null, svg);
      const clip = {};
      ["B", "L"].forEach((k) => { const c = sv("clipPath", { id: id + "c" + k }, defs); clip[k] = sv("rect", { x: 0, y: -4, width: 0, height: g.h + 8 }, c); });
      const Rf = { clip, panel: {}, sub: {}, heads: {}, marks: {}, rulers: {}, rulerLab: {}, gainLab: {}, lateSeg: {}, endLab: {} };
      const x10 = X(W10), x50 = X(E);
      const gx = x50 + (g.small ? 6 : 8);

      // eval band (behind everything in A)
      Rf.evband = sv("rect", { x: X(30), y: PN.A.top, width: x50 - X(30), height: PN.A.h, class: "lr-evband", opacity: 0 }, svg);
      Rf.evLab = stext(svg, (X(30) + x50) / 2, PN.A.bot - 6, `epochs 30–${E}`, "lr-annm lr-halo", "middle"); setOp(Rf.evLab, 0);

      // panels: titles, grid, ticks
      const titles = {
        A: g.small ? ["Two-hop accuracy", "no patching"] : ["Two-hop accuracy", "exact match, no patching"],
        B: g.small ? ["Memorization", "single-hop facts"] : ["Memorization", `single-hop training facts, measured every ${P.memStep} epochs`],
        C: ["Cross-entropy loss", "single-hop training facts"],
        D: g.small ? ["relMSE", `layer ${P.tgt} vs ${P.src}, two-hop prompts`] : ["relMSE", `between the layer-${P.tgt} and layer-${P.src} head-entity states, on two-hop prompts`],
      };
      Rf.titles = {};
      ["A", "B", "C", "D"].forEach((k) => {
        const p = PN[k];
        const t = stext(svg, 0, p.titleY, titles[k][0], "lr-ptitle");
        const ts = sv("tspan", { class: "lr-m" }, t); ts.textContent = "  ·  " + titles[k][1];
        Rf.titles[k] = t;
        p.ticks.forEach((v) => {
          const yy = Math.round(p.y(v)) + 0.5;
          sv("line", { x1: g.ml, x2: x50, y1: yy, y2: yy, class: "lr-gl" + (v === 0 ? " lr-base" : "") }, svg);
          stext(svg, g.ml - 7, yy + 3.5, p.fmt(v), "lr-tick", "end");
        });
      });
      // x axis
      for (let e = 0; e <= E; e += 10) stext(svg, X(e), PN.D.bot + 16, String(e), "lr-tick", "middle");
      stext(svg, g.ml + g.pw / 2, PN.D.bot + 32, "Training epoch", "lr-annm", "middle");

      // off-scale marker (S6): the oracle sits far above this 0-0.20 axis
      Rf.off = sv("g", { opacity: 0 }, svg);
      if (ora != null) {
        if (g.small) stext(Rf.off, 0, PN.A.titleY + 16, `↑ off scale: oracle self-patching ${f3(ora)}`, "lr-annm");
        else {
          const sx = gx + 4, y0 = PN.A.top;
          sv("path", { d: `M${sx} ${y0 + 9}V${y0 - 9}`, class: "lr-offstub" }, Rf.off);
          sv("path", { d: `M${sx - 3.5} ${y0 - 5}L${sx} ${y0 - 10}L${sx + 3.5} ${y0 - 5}`, class: "lr-offstub lr-solid" }, Rf.off);
          sv("path", { d: `M${sx - 4.5} ${y0 + 3}L${sx + 4.5} ${y0 - 0.5}V${y0 + 3}L${sx - 4.5} ${y0 + 6.5}Z`, class: "lr-breakgap" }, Rf.off);
          sv("path", { d: `M${sx - 4.5} ${y0 + 3}L${sx + 4.5} ${y0 - 0.5}M${sx - 4.5} ${y0 + 6.5}L${sx + 4.5} ${y0 + 3}`, class: "lr-break" }, Rf.off);
          stext(Rf.off, sx + 10, y0 - 1, `oracle ${f3(ora)}`, "lr-offlab");
          stext(Rf.off, sx + 10, y0 + 12, "off scale ↑", "lr-lab2");
        }
      }

      // wall: one dashed segment per panel, revealed top-down through a clip
      const segs = (x) => ["A", "B", "C", "D"].map((k) => `M${x} ${PN[k].top}V${PN[k].bot}`).join("");
      Rf.segs = segs;
      { const c = sv("clipPath", { id: id + "cw" }, defs); Rf.wallClip = sv("rect", { x: 0, y: PN.A.top - 2, width: g.w, height: 0 }, c); }
      Rf.wall = sv("path", { d: segs(x10), class: "lr-wall", "clip-path": `url(#${id}cw)` }, svg);
      Rf.wallLab = sv("g", { opacity: 0 }, svg);
      stext(Rf.wallLab, x10 + 7, PN.A.top + 11, g.small ? "≥ 90% memorized" : "≥ 90% memorized in every arm", "lr-ann lr-halo");
      stext(Rf.wallLab, x10 + 7, PN.A.top + 24, `(mean of ${A.B.seedIds.length} seeds)`, "lr-annm lr-halo");

      // panel groups with per-arm clipped subgroups
      ["A", "B", "C", "D"].forEach((k) => {
        const pg = sv("g", null, svg); Rf.panel[k] = pg; Rf.sub[k] = {};
        const p = PN[k];
        ARMS.forEach((Xa) => {
          const st = A[Xa], sg = sv("g", null, pg);
          Rf.sub[k][Xa] = sg;
          const inner = sv("g", { "clip-path": `url(#${id}c${Xa === "B" ? "B" : "L"})` }, sg);
          if (k === "A") {
            // wedge between the (smoothed) curve and the arm's epoch-10 level
            const pts = [[X(W10), p.y(st.ruler)]];
            for (let e = W10; e <= E; e++) pts.push([X(e), p.y(st.sm[e])]);
            pts.push([x50, p.y(st.ruler)]);
            Rf["wedge" + Xa] = sv("path", { d: pathOf(pts) + "Z", class: "lr-wedge lr-f" + Xa, opacity: 0 }, inner);
            const band = (e0, e1) => {
              const up = [], lo = [];
              for (let e = e0; e <= e1; e++) { up.push([X(e), p.y(st.sm[e] + st.smSd[e])]); lo.push([X(e), p.y(Math.max(0, st.sm[e] - st.smSd[e]))]); }
              return pathOf(up.concat(lo.reverse())) + "Z";
            };
            sv("path", { d: band(0, W10), class: "lr-band lr-f" + Xa }, inner);
            Rf["sdLate" + Xa] = sv("path", { d: band(W10, E), class: "lr-band lr-f" + Xa, opacity: 0 }, inner);
            // faint per-epoch means under a bold 5-epoch rolling mean (identical before the wall)
            const raw = [], pre = [], post = [];
            for (let e = 0; e <= E; e++) { raw.push([X(e), p.y(st.chain[e])]); (e <= W10 ? pre : post).push([X(e), p.y(st.sm[e])]); }
            post.unshift(pre[pre.length - 1]);
            sv("path", { d: pathOf(raw.slice(Math.max(0, W10 - 3))), class: "lr-line lr-raw lr-s" + Xa }, inner);
            sv("path", { d: pathOf(pre), class: "lr-line lr-s" + Xa }, inner);
            sv("path", { d: pathOf(post), class: "lr-line lr-s" + Xa, "stroke-width": 2.5 }, inner);
          } else if (k === "B") {
            const ln = st.mem.map((v, kk) => [X(kk * P.memStep), p.y(v)]);
            sv("path", { d: pathOf(ln), class: "lr-line lr-thin lr-s" + Xa }, inner);
            Rf.marks[Xa] = st.mem.map((v, kk) => sv("circle", { cx: X(kk * P.memStep), cy: p.y(v), r: 3.5, class: "lr-ring lr-f" + Xa }, sg));
          } else {
            const arr = k === "C" ? st.ce : st.rel, ln = [];
            for (let e = 0; e <= E; e++) if (arr[e] != null) ln.push([X(e), p.y(Math.min(arr[e], p.max))]);
            sv("path", { d: pathOf(ln), class: "lr-line lr-s" + Xa }, inner);
          }
        });
      });

      // rulers (epoch-10 level) and their labels
      ARMS.forEach((Xa) => { Rf.rulers[Xa] = sv("line", { x1: x10, x2: x10, y1: PN.A.y(A[Xa].ruler), y2: PN.A.y(A[Xa].ruler), class: "lr-ruler lr-s" + Xa }, Rf.panel.A); });
      const rItems = dodge(ARMS.map((Xa) => ({ Xa, y: PN.A.y(A[Xa].ruler) })), 13, PN.A.top + 8, PN.A.bot - 4);
      Rf.rulerLabG = sv("g", { opacity: 0 }, Rf.panel.A);
      const topR = Math.min(...rItems.map((d) => d.ly));
      stext(Rf.rulerLabG, gx, topR - 13, g.small ? "ep. " + W10 : "level at epoch " + W10, "lr-lab2");
      rItems.forEach((d) => {
        if (Math.abs(d.ly - d.y) > 1) sv("path", { d: `M${x50 + 1} ${d.y}L${gx - 2} ${d.ly}`, class: "lr-leader" }, Rf.rulerLabG);
        stext(Rf.rulerLabG, gx, d.ly + 4, f3(A[d.Xa].ruler), "lr-lab");
      });

      // late-mean segments (epochs 30-50) and "gain since epoch 10" labels
      ARMS.forEach((Xa) => {
        const yy = PN.A.y(A[Xa].late);
        Rf.lateSeg[Xa] = sv("line", { x1: X(30), x2: x50, y1: yy, y2: yy, class: "lr-line lr-s" + Xa, "stroke-width": 2.5, opacity: 0 }, svg);
        const gl = sv("g", { opacity: 0 }, svg); Rf.gainLab[Xa] = gl;
        stext(gl, gx, yy + 4, signed(gainAfter[Xa], 3), "lr-lab");
        if (!g.small) stext(gl, gx, yy + 17, "vs epoch " + W10, "lr-lab2");
      });

      // S6: bracket plain -> lambda=1, end labels (level + change since the wall), gain
      const yB = PN.A.y(A.B.late), y1 = PN.A.y(A.L1.late), y01 = PN.A.y(A.L01.late);
      const bx = x50 + 7, lx = bx + 8, subDy = g.small ? 15 : 17;
      Rf.res = sv("g", { opacity: 0 }, svg);
      sv("path", { d: `M${x50 + 2} ${y1}H${bx}V${yB}H${x50 + 2}`, class: "lr-bracket" }, Rf.res);
      const endItems = dodge([
        { Xa: "L1", y: y1, t: f3(A.L1.late) },
        { Xa: "L01", y: y01, t: g.small ? f3(A.L01.late) : `${f3(A.L01.late)} (+${A.L01.gain}%)` },
        { Xa: "B", y: yB, t: f3(A.B.late) },
      ], subDy + 13, PN.A.top + 6, PN.A.bot - subDy - 2);
      const bigH = g.small ? 19 : 25, l1 = endItems[0], l01 = endItems[1];
      const bigY = l1.ly + subDy + bigH, bigFits = bigY + 8 <= l01.ly - 6;
      if (!bigFits) l1.t += ` (+${A.L1.gain}%)`;
      endItems.forEach((d) => {
        sv("line", { x1: lx, x2: lx + (g.small ? 7 : 10), y1: d.ly, y2: d.ly, class: "lr-line lr-s" + d.Xa }, Rf.res);
        const tx = lx + (g.small ? 10 : 14);
        stext(Rf.res, tx, d.ly + 4, d.t, "lr-lab");
        stext(Rf.res, tx, d.ly + 4 + subDy - 3, signed(gainAfter[d.Xa], 3) + (g.small ? "" : ` vs ep. ${W10}`), "lr-lab2");
      });
      if (bigFits) stext(Rf.res, lx, bigY, `+${A.L1.gain}%`, "lr-big");
      // memorization, on its own (0-1) scale in panel B
      stext(Rf.res, gx, PN.B.y(mean(ARMS.map((Xa) => A[Xa].memEnd))) + 4, `≈ ${f2(mean(ARMS.map((Xa) => A[Xa].memEnd)))}` + (g.small ? "" : " in every arm"), "lr-ann");

      // lambda = 0.1 highlight for S5 (epochs 10-50, drawn over the dimmed panel)
      { const ln = []; for (let e = W10; e <= E; e++) ln.push([X(e), PN.A.y(A.L01.sm[e])]); Rf.hl01 = sv("path", { d: pathOf(ln), class: "lr-line lr-sL01", "stroke-width": 3, opacity: 0 }, svg); }

      // D annotations (S5 on): plain and λ = 1 in the gutter behind colour-keyed ticks, λ = 0.1 on its own curve;
      // the "epoch 10 → 50" key sits on the panel's title row
      Rf.dAnn = sv("g", { opacity: 0 }, svg);
      {
        const p = PN.D, arrow = g.small ? "→" : " → ", tx = gx + (g.small ? 8 : 13);
        const items = dodge([
          { Xa: "B", y: p.y(A.B.rel[E]), t: g.small ? "≈ " + f2(bRelAfter) : `≈ ${f2(bRelAfter)} throughout` },
          { Xa: "L1", y: p.y(A.L1.rel[E]), t: g.small ? `→ ${fixed(l1RelAfter)}` : `${f2(A.L1.rel[W10])}${arrow}${fixed(l1RelAfter)}` },
        ], 12, p.top + 4, p.bot + 2);
        items.forEach((d) => {
          if (Math.abs(d.ly - d.y) > 1) sv("path", { d: `M${x50 + 1} ${d.y}L${gx - 2} ${d.ly}`, class: "lr-leader" }, Rf.dAnn);
          sv("line", { x1: gx, x2: gx + (g.small ? 5 : 9), y1: d.ly, y2: d.ly, class: "lr-line lr-s" + d.Xa }, Rf.dAnn);
          stext(Rf.dAnn, tx, d.ly + 4, d.t, "lr-ann").style.fontSize = g.small ? "10.5px" : "";
        });
        let hiV = 0;
        for (let e = Math.round(E - (E - W10) * 0.35); e <= E; e++) hiV = Math.max(hiV, A.L01.rel[e]);
        stext(Rf.dAnn, x50 - 2, p.y(hiV) - 5, `${f2(A.L01.rel[W10])}${arrow}${f2(A.L01.rel[E])}`, "lr-ann lr-halo lr-on01", "end");
        Rf.dHead = stext(Rf.dAnn, g.w, p.titleY, g.small ? `ep. ${W10} → ${E}` : `labels: epoch ${W10} → ${E}`, "lr-lab2", "end");
      }

      // "LRSD paused" tag (S3)
      Rf.paused = stext(svg, x10 + 8, PN.A.y(A.L1.sm[W10]) - 12, g.small ? "LRSD: on hold" : `LRSD curves held at epoch ${W10}`, "lr-annm lr-halo");
      setOp(Rf.paused, 0);

      // cursor + head dots
      Rf.cursor = sv("path", { d: segs(0), class: "lr-cursor", opacity: 0 }, svg);
      ARMS.forEach((Xa) => { Rf.heads[Xa] = sv("circle", { r: 4, cx: 0, cy: 0, class: "lr-ring lr-f" + Xa, opacity: 0 }, svg); });

      // hover layer
      Rf.hair = sv("path", { d: segs(0), class: "lr-hair", opacity: 0 }, svg);
      Rf.hdots = {};
      ARMS.forEach((Xa) => { Rf.hdots[Xa] = sv("circle", { r: 4, class: "lr-ring lr-f" + Xa, opacity: 0 }, svg); });
      const hit = sv("rect", { x: g.ml, y: PN.A.top, width: g.pw + 6, height: PN.D.bot - PN.A.top, class: "lr-hit" }, svg);
      hit.addEventListener("pointermove", onHover);
      hit.addEventListener("pointerdown", onHover);
      hit.addEventListener("pointerleave", hideHover);

      R = Rf;
      // shorten titles that do not fit
      ["A", "B", "D"].forEach((k) => {
        const t = Rf.titles[k];
        try { if (t.getComputedTextLength() > g.w) t.lastChild.textContent = ""; } catch (e) { /* not rendered */ }
      });
      try { if (Rf.titles.D.getComputedTextLength() + Rf.dHead.getComputedTextLength() + 14 > g.w) Rf.dHead.style.display = "none"; } catch (e) { /* not rendered */ }
    }

    /* ---- springs (R1) ---- */
    let SP = null;
    function buildSprings(w) {
      svgS.textContent = "";
      const small = w < 300;
      const x0 = 70, x1 = w - 50, rowH = 30, y0 = 34;
      const hgt = y0 + rowH * 3 + 18;
      svgS.setAttribute("width", w); svgS.setAttribute("height", hgt); svgS.setAttribute("viewBox", `0 0 ${w} ${hgt}`);
      const Sx = (v) => x0 + (M.clamp(v, 0, 0.5) / 0.5) * (x1 - x0);
      // mini legend
      sv("circle", { cx: 6, cy: 10, r: 5, class: "lr-target" }, svgS);
      stext(svgS, 16, 14, `layer ${P.src} (fixed target)`, "lr-annm");
      const lx2 = small ? 140 : 160;
      sv("circle", { cx: lx2, cy: 10, r: 4.5, fill: "var(--ink-2)" }, svgS).style.fill = "var(--ink-2)";
      stext(svgS, lx2 + 9, 14, `layer ${P.tgt}`, "lr-annm");
      const rows = {};
      ARMS.forEach((Xa, i) => {
        const cy = y0 + i * rowH + rowH / 2;
        const g = sv("g", null, svgS);
        sv("line", { x1: 0, x2: 12, y1: cy, y2: cy, class: "lr-line lr-s" + Xa }, g);
        stext(g, 18, cy + 4, SHORT[Xa], "lr-ann");
        sv("line", { x1: x0, x2: x1, y1: cy, y2: cy, class: "lr-track" }, g);
        const spring = sv("path", { class: "lr-line lr-s" + Xa, "stroke-width": Xa === "L1" ? 2.5 : 1.25, fill: "none" }, g);
        const gap = Xa === "B" ? sv("line", { x1: x0, x2: x0, y1: cy, y2: cy, stroke: "currentColor", "stroke-dasharray": "1 3", class: "lr-leader" }, g) : null;
        const gapLab = Xa === "B" ? stext(g, 0, cy - 7, "no LRSD term", "lr-lab2", "middle") : null;
        sv("circle", { cx: x0, cy, r: 6, class: "lr-target" }, g);
        const dot = sv("circle", { cx: x0, cy, r: 5.5, class: "lr-ring lr-f" + Xa }, g);
        const val = stext(g, w, cy + 4, "", "lr-lab", "end");
        const hitR = sv("rect", { x: 0, y: cy - rowH / 2, width: w, height: rowH, class: "lr-hit" }, g);
        hitR.style.cursor = "default";
        hitR.addEventListener("pointermove", (ev) => springTip(ev, Xa));
        hitR.addEventListener("pointerleave", () => { sTip.style.display = "none"; });
        rows[Xa] = { g, spring, gap, gapLab, dot, val, cy };
      });
      const ty = y0 + rowH * 3 + 12;
      [0, 0.25, 0.5].forEach((v) => {
        sv("line", { x1: Sx(v), x2: Sx(v), y1: ty - 9, y2: ty - 6, class: "lr-track" }, svgS);
        stext(svgS, Sx(v), ty + 3, String(v), "lr-tick", "middle");
      });
      stext(svgS, 18, ty + 3, "relMSE", "lr-lab2");
      SP = { rows, Sx, x0, x1 };
    }
    function zigzag(xa, xb, y, coils) {
      const len = xb - xa;
      if (len < 8) return "";
      const n = coils * 2, amp = 4, lead = Math.min(4, len * 0.15);
      let d = `M${xa.toFixed(1)} ${y}L${(xa + lead).toFixed(1)} ${y}`;
      const span = len - 2 * lead;
      for (let k = 1; k < n; k++) d += `L${(xa + lead + (span * k) / n).toFixed(1)} ${(y + (k % 2 ? -amp : amp)).toFixed(1)}`;
      d += `L${(xb - lead).toFixed(1)} ${y}L${xb.toFixed(1)} ${y}`;
      return d;
    }

    /* ---- question grids (R2) ---- */
    const COLS = 21, PITCH = 7, CELL = 6, ROWS = Math.ceil(N / COLS);
    const GTINT = [["--ink", 0.5], [TOKEN.L1, 0.45], [TOKEN.B, 0.4]];   // group tints (S6): both, only LRSD λ = 1, only plain
    const GW = COLS * PITCH - 1, GH = ROWS * PITCH - 1;
    function drawGrids(eB, eL, alphaL, showEx, divA, focusOn) {
      const cols = { B: rgbOf(M.css(TOKEN.B, root)), L1: rgbOf(M.css(TOKEN.L1, root)) };
      const sf = rgbOf(M.css("--surface", root) || "#ffffff"), gr = rgbOf(M.css("--grid", root) || "#e1e0d9");
      const em = sf.map((v, i) => Math.round(v + (gr[i] - v) * 0.7)), empty = `rgb(${em[0]},${em[1]},${em[2]})`, ink = M.css("--ink", root) || "#000", focus = M.css("--focus", root) || "#2a78d6";
      const tints = GTINT.map(([tok, a]) => { const c = rgbOf(M.css(tok, root)), m = sf.map((v, i) => Math.round(v + (c[i] - v) * a)); return `rgb(${m[0]},${m[1]},${m[2]})`; });
      gridArms.forEach((Xa) => {
        const e = ep(Xa === "B" ? eB : eL), alpha = Xa === "B" ? 1 : alphaL;
        const ctx = M.fitCanvas(cv[Xa], GW, GH);
        ctx.clearRect(0, 0, GW, GH);
        const e0 = Math.max(0, e - 4), span = e - e0 + 1, bits = okBits[Xa], c = cols[Xa];
        // S6: each group's squares sit on a tinted tile (1 px wider than the square, so a group reads as one block)
        if (divA > 0) {
          ctx.globalAlpha = divA;
          for (let p = 0; p < N; p++) {
            const gi = group(order[p]);
            if (gi > 2) continue;
            ctx.fillStyle = tints[gi];
            ctx.fillRect((p % COLS) * PITCH - 1, ((p / COLS) | 0) * PITCH - 1, CELL + 2, CELL + 2);
          }
        }
        for (let p = 0; p < N; p++) {
          const i = order[p], x = (p % COLS) * PITCH, y = ((p / COLS) | 0) * PITCH;
          ctx.globalAlpha = divA > 0 && group(i) < 3 ? 1 - 0.55 * divA : 1; ctx.fillStyle = empty; ctx.fillRect(x, y, CELL, CELL);
          let k = 0; for (let ee = e0; ee <= e; ee++) k += bits[ee][i];
          if (k) { ctx.globalAlpha = alpha * (k / span); ctx.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`; ctx.fillRect(x, y, CELL, CELL); }
        }
        ctx.globalAlpha = 1;
        if (showEx && ex) {
          const p = posOf[ex.i], x = (p % COLS) * PITCH, y = ((p / COLS) | 0) * PITCH;
          ctx.strokeStyle = ink; ctx.lineWidth = 1.5; ctx.strokeRect(x - 1.25, y - 1.25, CELL + 2.5, CELL + 2.5);
        }
        if (focusOn >= 0) {
          const x = (focusOn % COLS) * PITCH, y = ((focusOn / COLS) | 0) * PITCH;
          ctx.strokeStyle = focus; ctx.lineWidth = 2; ctx.strokeRect(x - 1.5, y - 1.5, CELL + 3, CELL + 3);
        }
      });
      gridArms.forEach((Xa) => {
        const e = ep(Xa === "B" ? eB : eL);
        let n = 0; const b = okBits[Xa][e]; for (let i = 0; i < N; i++) n += b[i];
        const s = `${n} / ${N}|answered at epoch ${e}`;
        if (gcount[Xa].__s !== s) {
          gcount[Xa].__s = s; gcount[Xa].textContent = "";
          gcount[Xa].appendChild(h("b", null, `${n} / ${N}`));
          gcount[Xa].appendChild(document.createTextNode(`answered at epoch ${e}`));
        }
      });
    }
    function setNote(kind) {
      if (note.__k === kind) return;
      note.__k = kind; note.textContent = "";
      if (kind === "hint") note.textContent = "Hover or focus a square to read its question.";
      else if (kind === "ex" && ex) {
        note.appendChild(document.createTextNode(`Outlined: one of ${cands.length} questions that LRSD λ = 1 answers in every epoch 30–${E} and plain fine-tuning never answers: `));
        note.appendChild(h("q", null, Q.question[ex.i]));
        note.appendChild(document.createTextNode(` → ${Q.gold[ex.i]}. LRSD λ = 1 first answers it at epoch ${ex.first}.`));
      } else if (kind === "res") {
        note.appendChild(document.createTextNode(`Tint: answered in ≥ ${need} of epochs 30–${E} by `));
        [["both", 0], ["only λ = 1", 1], ["only plain", 2]].forEach(([lab, gi], j) => {
          const it = h("span", "lr-gk"); it.appendChild(h("span", "lr-sw lr-sw" + gi)); it.appendChild(document.createTextNode(`${lab} (${gCount[gi]})`));
          note.appendChild(it); note.appendChild(document.createTextNode(j < 2 ? " · " : `; neither (${gCount[3]}).`));
        });
      }
    }

    /* ---- the frame ---- */
    function heads(t) {
      if (t < 0.04) return [0, 0];
      if (t < 0.22) { const e = W10 * sweepEase((t - 0.04) / 0.18); return [e, e]; }
      if (t < 0.31) return [W10, W10];
      if (t < 0.49) return [W10 + (E - W10) * sweepEase((t - 0.31) / 0.18), W10];
      if (t < 0.53) return [E, W10];
      if (t < 0.73) return [E, W10 + (E - W10) * sweepEase((t - 0.53) / 0.2)];
      return [E, E];
    }
    function cursorAt(t, hd) {
      if (t < 0.04) return 0;
      if (t < 0.22) return hd[1];
      if (t < 0.31) return W10;
      if (t < 0.49) return hd[0];
      if (t < 0.53) return E - (E - W10) * M.easeInOut((t - 0.49) / 0.04);
      if (t < 0.73) return hd[1];
      return E;
    }
    function render(t) {
      t = M.clamp(t, 0, 1); T = t;
      if (!R) return;
      const g = G, X = g.X, PN = g.P;
      const si = sceneAt(t), sc = SCENES[si];
      const hd = heads(t), eB = hd[0], eL = hd[1];
      const headOf = (Xa) => (Xa === "B" ? eB : eL);
      R.clip.B.setAttribute("width", Math.max(0, X(eB) + 1).toFixed(1));
      R.clip.L.setAttribute("width", Math.max(0, X(eL) + 1).toFixed(1));

      const dim = 1 - 0.65 * (ramp(t, 0.31, 0.335) - ramp(t, 0.49, 0.52));
      const foc = ramp(t, 0.73, 0.76) * (1 - ramp(t, 0.84, 0.87));
      ["A", "B", "C"].forEach((k) => setOp(R.panel[k], 1 - 0.55 * foc));
      ["A", "B", "C", "D"].forEach((k) => { setOp(R.sub[k].L01, dim); setOp(R.sub[k].L1, dim); });
      setOp(R.hl01, foc);
      setOp(R.dAnn, ramp(t, 0.745, 0.775));

      // wedges (gain over the epoch-10 level): plain from S3, LRSD from S4b; kept at half strength in S6 next to the sd bands
      const wOut = 1 - 0.5 * ramp(t, 0.84, 0.87);
      setOp(R.wedgeB, ramp(t, 0.31, 0.33) * wOut);
      setOp(R.wedgeL01, ramp(t, 0.53, 0.55) * wOut); setOp(R.wedgeL1, ramp(t, 0.53, 0.55) * wOut);
      const sdIn = 0.5 * ramp(t, 0.85, 0.88);
      ARMS.forEach((Xa) => setOp(R["sdLate" + Xa], sdIn));

      // wall drop + label
      const w = ramp(t, 0.22, 0.245);
      R.wallClip.setAttribute("height", ((PN.D.bot - PN.A.top + 4) * w).toFixed(1));
      setOp(R.wallLab, ramp(t, 0.23, 0.25));

      // memorization markers: shown once reached; pulse at the wall
      const pulseP = M.clamp((t - 0.225) / 0.03, 0, 1), pulse = pulseP > 0 && pulseP < 1 ? Math.sin(Math.PI * pulseP) : 0;
      ARMS.forEach((Xa) => {
        const e = headOf(Xa);
        R.marks[Xa].forEach((c, k) => {
          setOp(c, k * P.memStep <= e + 1e-6 ? 1 : 0);
          const rr = (3.5 * (k === P.wallK ? 1 + 0.4 * pulse : 1)).toFixed(2);
          if (c.__r !== rr) { c.__r = rr; c.setAttribute("r", rr); }
        });
      });

      // rulers wipe right, starting with the chapter-2 caption that describes them
      ARMS.forEach((Xa, i) => {
        const k = ramp(t, 0.225 + 0.008 * i, 0.25 + 0.008 * i);
        R.rulers[Xa].setAttribute("x2", M.lerp(X(W10), X(E), k));
        setOp(R.rulers[Xa], k > 0 ? 1 - 0.4 * ramp(t, 0.84, 0.87) : 0);
      });
      setOp(R.rulerLabG, ramp(t, 0.255, 0.275) * (1 - ramp(t, 0.45, 0.47)));

      // late means and gains
      const gB = ramp(t, 0.47, 0.49), gL = ramp(t, 0.71, 0.73), out = 1 - ramp(t, 0.84, 0.86);
      setOp(R.lateSeg.B, gB); setOp(R.lateSeg.L01, gL); setOp(R.lateSeg.L1, gL);
      setOp(R.gainLab.B, gB * out); setOp(R.gainLab.L01, gL * out); setOp(R.gainLab.L1, gL * out);

      // S6
      const band = ramp(t, 0.84, 0.87);
      setOp(R.evband, band); setOp(R.evLab, band);
      setOp(R.res, ramp(t, 0.86, 0.9));
      setOp(R.off, ramp(t, 0.88, 0.92));
      const divA = ramp(t, 0.9, 0.93);

      setOp(R.paused, ramp(t, 0.32, 0.345) * (1 - ramp(t, 0.49, 0.51)));

      // heads + cursor
      const cur = cursorAt(t, hd);
      const headVis = ramp(t, 0.04, 0.05) * (1 - ramp(t, 0.84, 0.86));
      ARMS.forEach((Xa) => {
        const e = headOf(Xa), c = R.heads[Xa];
        c.setAttribute("cx", X(e).toFixed(1)); c.setAttribute("cy", PN.A.y(interp(A[Xa].sm, e)).toFixed(1));
        setOp(c, headVis * (Xa === "B" ? 1 : dim));
      });
      const cx = Math.round(X(cur)) + 0.5;
      R.cursor.setAttribute("transform", `translate(${cx} 0)`);
      setOp(R.cursor, ramp(t, 0.04, 0.06) * (1 - ramp(t, 0.73, 0.75)));

      // springs
      if (SP) {
        ARMS.forEach((Xa) => {
          const row = SP.rows[Xa], v = interp(A[Xa].rel, headOf(Xa)), xd = SP.Sx(v);
          row.dot.setAttribute("cx", xd.toFixed(1));
          if (Xa === "B") {
            row.gap.setAttribute("x1", SP.x0 + 8); row.gap.setAttribute("x2", Math.max(SP.x0 + 8, xd - 8));
            row.gapLab.setAttribute("x", ((SP.x0 + xd) / 2).toFixed(1));
          } else row.spring.setAttribute("d", zigzag(SP.x0 + 6, xd - 5.5, row.cy, 6));
          const s = f3(v);
          if (row.val.textContent !== s) row.val.textContent = s;
          setOp(row.g, Xa === "B" ? 1 : dim);
        });
      }

      // grids (only when something visible changed)
      const showEx = t >= 0.53 && eL >= (ex ? ex.first : 99);
      const gk = [ep(eB), ep(eL), dim.toFixed(2), showEx, divA.toFixed(2), focusP, themeV].join("|");
      if (gk !== gridKey) { gridKey = gk; drawGrids(eB, eL, dim, showEx, divA, focusP); }
      setNote(t >= 0.86 ? "res" : t >= 0.53 ? "ex" : "hint");

      // formula pulse (S4a)
      const fp = M.clamp((t - 0.495) / 0.03, 0, 1), fk = fp > 0 && fp < 1 ? Math.sin(Math.PI * fp) : 0;
      formula.style.transform = fk ? `scale(${(1 + 0.06 * fk).toFixed(3)})` : "";
      formula.style.borderColor = fk > 0.05 ? "var(--accent)" : "";

      // text
      const ci = capIndex[si];
      if (cap.__i !== ci) {
        cap.__i = ci; cap.textContent = "";
        cap.appendChild(h("strong", null, CAP[ci][0])); cap.appendChild(document.createTextNode(" " + CAP[ci][1]));
      }
      const eRo = si === 4 ? W10 : ep(cur);        // S4a (rewind): hold the readout at the epoch LRSD resumes from
      const ro = `Epoch|${eRo}| / ${E}`;
      if (readout.__s !== ro) { readout.__s = ro; readout.textContent = "Epoch "; readout.appendChild(h("b", null, String(eRo))); readout.appendChild(document.createTextNode(` / ${E}`)); }
      const ch = si === 0 ? -1 : sc.ch - 1;       // setup: no chapter pressed yet
      chips.forEach((b, i) => { const on = i === ch ? "true" : "false"; if (b.getAttribute("aria-pressed") !== on) { b.setAttribute("aria-pressed", on); if (on === "true") centerChip(b); } });
      const rv = String(Math.round(t * 1000));
      if (range.value !== rv) range.value = rv;
      const fill = (t * 100).toFixed(1) + "%";
      range.style.setProperty("--lr-p", fill); range.style.setProperty("--fill", fill);
      const vt = (si === 0 ? "Setup" : `Chapter ${sc.ch} of 6, ${CHAPTERS[ch][0]}`) + `: plain epoch ${ep(eB)}, LRSD epoch ${ep(eL)}`;
      if (range.getAttribute("aria-valuetext") !== vt) range.setAttribute("aria-valuetext", vt);
      if (hoverE != null) drawHover();
    }
    function centerChip(b) {
      const want = b.offsetLeft - (chipsEl.clientWidth - b.offsetWidth) / 2;
      if (chipsEl.scrollWidth > chipsEl.clientWidth) chipsEl.scrollLeft = Math.max(0, want - chipsEl.offsetLeft);
    }

    /* ---- hover: chart crosshair ---- */
    let hoverX = 0;
    function onHover(ev) {
      const rect = svg.getBoundingClientRect();
      hoverX = ev.clientX - rect.left;
      hoverE = M.clamp(Math.round(G.Xinv(hoverX)), 0, E);
      drawHover();
    }
    function hideHover() {
      hoverE = null; tip.style.display = "none";
      if (R) { setOp(R.hair, 0); ARMS.forEach((Xa) => setOp(R.hdots[Xa], 0)); }
    }
    function drawHover() {
      const hd = heads(T), e = hoverE;
      const vis = { B: e <= hd[0] + 1e-6, L01: e <= hd[1] + 1e-6, L1: e <= hd[1] + 1e-6 };
      if (!vis.B && !vis.L1) { tip.style.display = "none"; setOp(R.hair, 0); ARMS.forEach((Xa) => setOp(R.hdots[Xa], 0)); return; }
      const X = G.X, PN = G.P, xx = Math.round(X(e)) + 0.5;
      R.hair.setAttribute("transform", `translate(${xx} 0)`); setOp(R.hair, 1);
      ARMS.forEach((Xa) => {
        const c = R.hdots[Xa];
        if (!vis[Xa]) { setOp(c, 0); return; }
        c.setAttribute("cx", X(e)); c.setAttribute("cy", PN.A.y(A[Xa].chain[e])); setOp(c, 1);
      });
      const km = Math.round(e / P.memStep), em = km * P.memStep;
      tip.textContent = "";
      tip.appendChild(h("div", "lr-tt", `Epoch ${e}`));
      const tb = h("table"), thr = h("tr");
      thr.appendChild(h("th"));
      ARMS.forEach((Xa) => { const c = h("th"); c.appendChild(key(Xa)); c.appendChild(document.createTextNode(SHORT[Xa])); thr.appendChild(c); });
      tb.appendChild(thr);
      const row = (label, fn, cls) => {
        const tr = h("tr"); tr.appendChild(h("td", null, label));
        ARMS.forEach((Xa) => { const td = h("td", cls || null); if (!vis[Xa]) td.textContent = "–"; else fn(td, A[Xa], Xa); tr.appendChild(td); });
        tb.appendChild(tr);
      };
      row("Two-hop", (td, st) => td.appendChild(h("b", null, f3(st.chain[e]))));
      row("± sd", (td, st) => { td.textContent = f3(st.sd[e]); });
      row(`seeds ${A.B.seedIds.join(" / ")}`, (td, st) => { td.textContent = st.seedChain[e].map(f3).join("\n"); }, "lr-seeds");
      row(`Memorization*`, (td, st) => { td.textContent = f3(st.mem[km]); });
      row("CE loss", (td, st) => { td.textContent = e === 0 ? "–" : f3(st.ce[e]); });
      row("relMSE", (td, st) => { td.textContent = f3(st.rel[e]); });
      row("LRSD loss", (td, st, Xa) => { td.textContent = Xa === "B" || e === 0 ? "–" : f3(st.lr[e]); });
      tip.appendChild(tb);
      tip.appendChild(h("div", "lr-tfoot", `* measured every ${P.memStep} epochs; value at epoch ${em}. Two-hop: this epoch, mean of ${A.B.seedIds.length} seeds (the bold lines after epoch ${W10} are 5-epoch rolling means).`));
      tip.style.display = "block";
      const tw = tip.offsetWidth, cw = chart.clientWidth;
      let left = X(e) + 16;
      if (left + tw > cw) left = X(e) - 16 - tw;
      if (left < 0) left = Math.max(0, Math.min(cw - tw, X(e) - tw / 2));
      tip.style.left = left + "px";
      tip.style.top = (PN.A.top + 4) + "px";
    }

    /* ---- hover: springs and grids ---- */
    function springTip(ev, Xa) {
      const hd = heads(T), e = Xa === "B" ? hd[0] : hd[1], v = interp(A[Xa].rel, e);
      sTip.textContent = "";
      const t1 = h("div", "lr-tt"); t1.appendChild(key(Xa)); t1.appendChild(document.createTextNode(`${NAME[Xa]} · epoch ${ep(e)}`)); sTip.appendChild(t1);
      const d = h("div"); d.appendChild(h("b", null, f3(v))); d.appendChild(document.createTextNode(` relMSE, layer ${P.tgt} vs layer ${P.src}`)); sTip.appendChild(d);
      sTip.appendChild(h("div", "lr-tfoot", `Mean of ${A.B.seedIds.length} seeds, on the two-hop prompts. 0 = identical states.` + (Xa === "B" ? " Plain fine-tuning has no term pulling them together." : "")));
      placeSideTip(ev.clientX, ev.clientY);
    }
    function placeSideTip(cx, cy) {
      sTip.style.maxWidth = Math.min(340, side.clientWidth) + "px";
      sTip.style.display = "block";
      const rs = side.getBoundingClientRect(), tw = sTip.offsetWidth, th = sTip.offsetHeight;
      let x = cx - rs.left + 14, y = cy - rs.top + 14;
      if (x + tw > rs.width) x = Math.max(0, cx - rs.left - tw - 14);
      if (x < 0) x = 0;
      if (y + th > side.scrollHeight + 40) y = cy - rs.top - th - 14;
      sTip.style.left = x + "px"; sTip.style.top = y + "px";
    }
    function barcode(Xa, i) {
      const s = sv("svg", { width: E + 1 > 60 ? 153 : (E + 1) * 3, height: 10, viewBox: `0 0 ${(E + 1) * 3} 10`, "aria-hidden": "true" });
      sv("rect", { x: 0, y: 0, width: (E + 1) * 3, height: 10, fill: "var(--grid)" }, s).style.fill = "var(--grid)";
      for (let e = 0; e <= E; e++) if (okBits[Xa][e][i]) { const r = sv("rect", { x: e * 3, y: 0, width: 3, height: 10 }, s); r.style.fill = `var(${TOKEN[Xa]})`; }
      return s;
    }
    function questionTip(p, anchorEl, cx, cy) {
      if (p < 0 || p >= N) { sTip.style.display = "none"; return; }
      const i = order[p];
      sTip.textContent = "";
      sTip.appendChild(h("div", "lr-q", Q.question[i]));
      sTip.appendChild(h("div", "lr-a", "Gold answer: " + Q.gold[i]));
      gridArms.forEach((Xa) => {
        const r = h("div", "lr-bc"); r.appendChild(h("span", null, SHORT[Xa])); r.appendChild(barcode(Xa, i));
        const k = h("span"); k.appendChild(h("b", null, String(lateCount[Xa][i]))); k.appendChild(document.createTextNode(`/${nLate}`)); r.appendChild(k); sTip.appendChild(r);
      });
      sTip.appendChild(h("div", "lr-tfoot", `Filled = answered at that epoch (0–${E}), seed ${Q.seed}. k/${nLate} = answered in k of epochs 30–${E}.`));
      if (cx == null) { const rc = anchorEl.getBoundingClientRect(); cx = rc.left + (p % COLS) * PITCH; cy = rc.top + ((p / COLS) | 0) * PITCH + 8; }
      placeSideTip(cx, cy);
      live.textContent = `Question ${p + 1} of ${N}: ${Q.question[i]} Gold answer: ${Q.gold[i]}. Answered in ${lateCount.B[i]} of epochs 30 to ${E} by plain fine-tuning, ${lateCount.L1[i]} with LRSD lambda 1.`;
    }
    gridArms.forEach((Xa) => {
      cv[Xa].addEventListener("pointermove", (ev) => {
        const rc = cv[Xa].getBoundingClientRect(), c = Math.floor((ev.clientX - rc.left) / PITCH), rr = Math.floor((ev.clientY - rc.top) / PITCH);
        const p = c >= 0 && c < COLS && rr >= 0 ? rr * COLS + c : -1;
        questionTip(p, cv[Xa], ev.clientX, ev.clientY);
      });
      cv[Xa].addEventListener("pointerleave", () => { sTip.style.display = "none"; });
    });
    grids.addEventListener("focus", () => { if (focusP < 0) focusP = 0; gridKey = ""; render(T); questionTip(focusP, cv.L1); });
    grids.addEventListener("blur", () => { focusP = -1; gridKey = ""; render(T); sTip.style.display = "none"; });
    grids.addEventListener("keydown", (ev) => {
      const mv = { ArrowRight: 1, ArrowLeft: -1, ArrowDown: COLS, ArrowUp: -COLS, Home: -N, End: N }[ev.key];
      if (mv == null) return;
      ev.preventDefault(); ev.stopPropagation();
      focusP = M.clamp(focusP + mv, 0, N - 1); gridKey = ""; render(T); questionTip(focusP, cv.L1);
    });

    /* ---- playback ---- */
    function tick(dt) {
      if (tween) {
        tween.k = Math.min(1, tween.k + dt / 0.3);
        render(M.lerp(tween.from, tween.to, M.easeInOut(tween.k)));
        if (tween.k >= 1) { tween = null; if (!wantPlay) clock.stop(); }
        return;
      }
      if (!wantPlay) { clock.stop(); return; }
      const nt = Math.min(1, T + (dt * speed) / DUR);
      render(nt);
      if (nt >= 1) { wantPlay = false; clock.stop(); setPlayUI(); }
    }
    function setPlayUI() {
      bPlay.setAttribute("aria-pressed", wantPlay ? "true" : "false");
      bPlay.setAttribute("aria-label", wantPlay ? "Pause" : T >= 1 && hasPlayed ? "Play again" : "Play");
      bPlay.__path.setAttribute("d", wantPlay ? ICON.pause : ICON.play);
      cap.setAttribute("aria-live", wantPlay ? "off" : "polite");   // announce captions only when paused / scrubbed
    }
    // The figure rests on its end frame (a poster) until it first plays; the first play starts from the beginning.
    function play() {
      if (wantPlay) return;
      if (!hasPlayed) { hasPlayed = true; tween = null; render(0); }
      else if (T >= 1) return;
      wantPlay = true; clock.start(); setPlayUI();
    }
    function pause() { wantPlay = false; if (!tween) clock.stop(); setPlayUI(); }
    function userPlay() { M.userPaused("lrsd", false); if (T >= 1) render(0); wantPlay = false; play(); }
    function userPause() { M.userPaused("lrsd", true); pause(); }
    function seekTo(t, animate) {
      t = M.clamp(t, 0, 1); hasPlayed = true;
      if (animate && !M.reducedMotion) { tween = { from: T, to: t, k: 0 }; clock.start(); }
      else { tween = null; render(t); if (!wantPlay) clock.stop(); }
    }
    function jumpChapter(i) {
      if (M.reducedMotion) { userPause(); seekTo(KEYFRAME[i], false); return; }
      M.userPaused("lrsd", false);
      seekTo(CHAPTERS[i][1], true);
      wantPlay = true; setPlayUI();
    }
    function setSpeed(v) {
      speed = v;
      segBtns.forEach((b, i) => b.setAttribute("aria-pressed", speeds[i] === v ? "true" : "false"));
      sel.value = String(v);
    }
    bPlay.addEventListener("click", () => (wantPlay ? userPause() : userPlay()));
    bReplay.addEventListener("click", () => { M.userPaused("lrsd", false); tween = null; hasPlayed = true; render(0); wantPlay = false; play(); });
    range.addEventListener("input", () => { userPause(); tween = null; hasPlayed = true; render(Number(range.value) / 1000); });
    function stepEpoch(dir) {
      const si = sceneAt(T), sc = SCENES[si];
      if (!sc.sweep) { seekTo(T + dir * 0.01, false); return; }
      const idx = sc.sweep === "B" ? 0 : 1, cur = heads(T)[idx];
      const target = dir > 0 ? Math.floor(cur + 1e-6) + 1 : Math.ceil(cur - 1e-6) - 1;
      const eAt = (tt) => heads(tt)[idx];
      if (target > eAt(sc.t1 - 1e-6) + 1e-6 || target < eAt(sc.t0) - 1e-6) { seekTo(dir > 0 ? sc.t1 : sc.t0 - 0.001, false); return; }
      let lo = sc.t0, hi = sc.t1 - 1e-6;
      for (let k = 0; k < 40; k++) { const mid = (lo + hi) / 2; if (eAt(mid) < target) lo = mid; else hi = mid; }
      seekTo(hi, false);
    }
    card.addEventListener("keydown", (ev) => {
      if (ev.target !== card) return;
      const k = ev.key;
      if (k === " " || k === "Spacebar") { ev.preventDefault(); wantPlay ? userPause() : userPlay(); }
      else if (k === "ArrowRight" || k === "ArrowLeft") { ev.preventDefault(); userPause(); stepEpoch(k === "ArrowRight" ? 1 : -1); }
      else if (k === "PageDown" || k === "]" || k === "PageUp" || k === "[") {
        ev.preventDefault();
        const ch = SCENES[sceneAt(T)].ch - 1, nx = M.clamp(ch + (k === "PageDown" || k === "]" ? 1 : -1), 0, 5);
        jumpChapter(nx);
      } else if (/^[1-6]$/.test(k)) { ev.preventDefault(); jumpChapter(Number(k) - 1); }
      else if (k === "Home") { ev.preventDefault(); userPause(); seekTo(0, false); }
      else if (k === "End") { ev.preventDefault(); userPause(); seekTo(1, false); }
    });

    /* ---- layout ---- */
    function measureNote() {
      const k = note.__k;
      note.style.minHeight = "";
      let mx = 0;
      ["hint", "ex", "res"].forEach((kind) => { note.__k = null; setNote(kind); mx = Math.max(mx, note.offsetHeight); });
      note.__k = null; note.style.minHeight = mx + "px";
      if (k) setNote(k);
    }
    function measureCaption() {
      capM.style.width = cap.clientWidth + "px";
      let mx = 0;
      CAP.forEach((c) => { capM.textContent = ""; capM.appendChild(h("strong", null, c[0])); capM.appendChild(document.createTextNode(" " + c[1])); mx = Math.max(mx, capM.offsetHeight); });
      capM.textContent = "";
      cap.style.minHeight = mx + "px";
    }
    function layout() {
      const W = Math.round(el.clientWidth || root.clientWidth || 1100);
      if (W === lastW) return;
      lastW = W;
      const narrow = W < 900, small = W < 560, mid = narrow && W >= 700;
      root.classList.toggle("lr-narrow", narrow);
      root.classList.toggle("lr-small", small);
      root.classList.toggle("lr-mid", mid);
      if (narrow) {
        if (ctl.nextSibling !== cap) card.insertBefore(ctl, cap);
        if (side.previousSibling !== cap) card.insertBefore(side, det);
      } else {
        if (cap.nextSibling !== ctl) card.insertBefore(cap, ctl);
        if (side.parentNode !== body) body.appendChild(side);
      }
      hideHover(); sTip.style.display = "none";
      G = geometry(Math.max(200, chart.clientWidth), small, !narrow);
      buildChart();
      const sw = mid ? (side.clientWidth - 32) / 2 : side.clientWidth;
      buildSprings(Math.max(240, Math.min(sw, 420)));
      measureCaption();
      measureNote();
      if (!narrow) {                     // grow panel A so the chart column matches the side column
        const extra = Math.round(M.clamp(side.offsetHeight - G.h - 4, 0, 90));
        if (extra > 6) { G = geometry(G.w, small, true, extra); buildChart(); }
      }
      chipsEl.classList.toggle("lr-ovf", chipsEl.scrollWidth > chipsEl.clientWidth + 1);
      gridKey = "";
      render(T);
    }
    const ro = "ResizeObserver" in window ? new ResizeObserver(() => layout()) : null;
    if (ro) ro.observe(el); else window.addEventListener("resize", layout);
    M.onTheme(() => { themeV++; gridKey = ""; render(T); });
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => { lastW = -1; layout(); });

    layout();
    render(1);                           // poster: the result frame, until the figure first plays
    setPlayUI();

    return {
      seek(t) { tween = null; hasPlayed = true; render(t); setPlayUI(); },
      play,
      pause,
    };
  }

  /* ================================================================== #viz-lrsd-models */
  function Models(el, D) {
    const L = D.lrsd;
    const MS = L.models.map((m) => prepModel(m, L.summary[m.key]));
    const E = MS[0].E, nS = MS[0].arms.B.seedIds.length;
    const id = "lrm" + UID++;
    // seed-paired comparisons across all models
    const test = {};
    ["L01", "L1"].forEach((X) => {
      const d = [];
      MS.forEach((m) => m.arms[X].seedIds.forEach((sd, j) => {
        const jb = m.arms.B.seedIds.indexOf(sd);
        if (jb < 0) warn(`${m.key}: seed ${sd} missing in plain arm`); else d.push(m.arms[X].seedLate[j] - m.arms.B.seedLate[jb]);
      }));
      test[X] = { n: d.length, wins: d.filter((v) => v > 0).length, p: signFlip(d) };
    });
    const llama = MS.filter((m) => /llama/i.test(m.name)).map((m) => m.arms.L1.gain);
    const qwen = MS.filter((m) => /qwen/i.test(m.name)).map((m) => m.arms.L1.gain);

    injectStyle();
    el.textContent = "";
    const root = h("div", "lr-root lr-models"); el.appendChild(root);
    const card = h("div", "lr-card"); root.appendChild(card);
    const legend = h("div", "lr-legend m2g-legend"); card.appendChild(legend);
    ARMS.forEach((X) => { const li = h("span", "lr-li"); li.appendChild(key(X)); li.appendChild(document.createTextNode(NAME[X])); legend.appendChild(li); });
    { const li = h("span", "lr-li"); li.appendChild(h("span", "lr-key lr-k-dash")); li.appendChild(document.createTextNode(`≥ 90% memorized (mean of ${nS} seeds)`)); legend.appendChild(li); }
    { const li = h("span", "lr-li"); li.appendChild(h("span", "lr-key lr-k-band")); li.appendChild(document.createTextNode(`Epochs 30–${E} (labels: their mean)`)); legend.appendChild(li); }

    const gridEl = h("div", "lr-mgrid"); card.appendChild(gridEl);
    const panels = MS.map((m) => {
      const wrap = h("div", "lr-mpanel"); gridEl.appendChild(wrap);
      const tt = h("p", "lr-mtitle", m.name + " "); tt.appendChild(h("span", null, `· layers ${m.src}→${m.tgt} of ${m.nL}`)); wrap.appendChild(tt);
      const svgM = sv("svg", { role: "img", "aria-label": `${m.name}: two-hop accuracy over ${E} epochs. Mean of epochs 30 to ${E}: plain ${f3(m.arms.B.late)}, lambda 0.1 ${f3(m.arms.L01.late)}, lambda 1 ${f3(m.arms.L1.late)}.` }, wrap);
      const meta = h("div", "lr-mmeta"); wrap.appendChild(meta);
      const l0 = h("div", "lr-mrow"); l0.appendChild(h("span", "lr-ml", "Gain over plain:")); meta.appendChild(l0);
      ["L01", "L1"].forEach((X) => { const s = h("span", "lr-mv"); s.appendChild(key(X)); s.appendChild(h("b", null, `+${m.arms[X].gain}%`)); l0.appendChild(s); });
      meta.appendChild(h("div", null, `Memorization at epoch ${E}:`));
      const l1 = h("div", "lr-mrow"); meta.appendChild(l1);
      ARMS.forEach((X) => { const s = h("span", "lr-mv"); s.appendChild(key(X)); s.appendChild(document.createTextNode(f3(m.arms[X].memEnd))); l1.appendChild(s); });
      meta.appendChild(h("div", null, `n = ${m.nQ} two-hop questions`));
      const tipM = h("div", "lr-tip m2g-tip"); tipM.setAttribute("aria-hidden", "true"); wrap.appendChild(tipM);
      return { m, wrap, svg: svgM, tip: tipM, G: null, R: null };
    });
    const both = test.L01.wins === test.L01.n && test.L1.wins === test.L1.n && test.L01.n === test.L1.n;
    const pStr = (p) => "p\u00a0≈\u00a0" + (p < 0.001 ? p.toFixed(4) : p.toFixed(3));
    const rng = (a) => (Math.min(...a) === Math.max(...a) ? `+${a[0]}%` : `+${Math.min(...a)}\u2060–\u2060${Math.max(...a)}%`);
    const foot = h("p", "lr-foot");
    foot.textContent = (both
      ? `LRSD beats plain fine-tuning in all ${test.L1.n} seed-paired comparisons at both λ (two-sided sign-flip test, ${pStr(Math.max(test.L01.p, test.L1.p))}).`
      : `LRSD beats plain fine-tuning in ${test.L01.wins}/${test.L01.n} seed-paired comparisons at λ = 0.1 (${pStr(test.L01.p)}) and ${test.L1.wins}/${test.L1.n} at λ = 1 (${pStr(test.L1.p)}; two-sided sign-flip test).`)
      + (llama.length && qwen.length ? ` The gain over plain fine-tuning is largest on Qwen (${rng(qwen)} at λ\u00a0=\u00a01); on LLaMA-3.2 it is smaller (${rng(llama)}).` : "");
    card.appendChild(foot);

    const ctl = h("div", "lr-mctl"); card.appendChild(ctl);
    const bRe = iconBtn("Replay the drawing", ICON.replay); ctl.appendChild(bRe);
    ctl.appendChild(h("span", "lr-ctl-lab", "Replay"));
    const det = h("details", "m2g-data lr-data"); ctl.appendChild(det);
    det.appendChild(h("summary", null, "Data table"));
    {
      const wrap = h("div", "lr-tablewrap"), tb = h("table", "lr-table");
      tb.appendChild(h("caption", null, `Two-hop accuracy without patching, mean of epochs 30–${E} ± sd over ${nS} seeds; gain relative to plain fine-tuning. Memorization at epoch ${E}.`));
      const th = h("thead"), tr = h("tr");
      ["Model", "Layers", "Plain", "λ = 0.1", "λ = 1", "Memorization (plain / 0.1 / 1)", "n"].forEach((c, i) => { const x = h("th", i < 2 ? "lr-tl" : null, c); x.scope = "col"; tr.appendChild(x); });
      th.appendChild(tr); tb.appendChild(th);
      const tbd = h("tbody");
      MS.forEach((m) => {
        const r = h("tr");
        r.appendChild(h("td", "lr-tl", m.name));
        r.appendChild(h("td", "lr-tl", `${m.src}→${m.tgt} of ${m.nL}`));
        ARMS.forEach((X) => r.appendChild(h("td", null, `${f3(m.arms[X].late)} ± ${f3(m.arms[X].lateSd)}` + (X === "B" ? "" : ` (+${m.arms[X].gain}%)`))));
        r.appendChild(h("td", null, ARMS.map((X) => f3(m.arms[X].memEnd)).join(" / ")));
        r.appendChild(h("td", null, String(m.nQ)));
        tbd.appendChild(r);
      });
      tb.appendChild(tbd); wrap.appendChild(tb); det.appendChild(wrap);
    }

    let T = 0, playing = false, hasPlayed = false, lastW = -1, hoverE = null;
    const DURM = 4.5;
    const clock = M.clock((dt) => {
      if (!playing) { clock.stop(); return; }
      const nt = Math.min(1, T + dt / DURM);
      render(nt);
      if (nt >= 1) { playing = false; clock.stop(); }
    });

    function buildPanel(pn, w, hPlot) {
      const m = pn.m, s = pn.svg;
      s.textContent = "";
      const g = { ml: 30, gut: 46, top: 8, h: hPlot };
      g.pw = w - g.ml - g.gut; g.bot = g.top + hPlot; g.H = g.bot + 30;
      g.X = (e) => g.ml + (e / E) * g.pw; g.Y = (v) => g.bot - (v / 0.21) * hPlot;
      s.setAttribute("width", w); s.setAttribute("height", g.H); s.setAttribute("viewBox", `0 0 ${w} ${g.H}`);
      const defs = sv("defs", null, s), cp = sv("clipPath", { id: id + m.key.replace(/\W/g, "") }, defs);
      const clip = sv("rect", { x: -4, y: -4, width: 0, height: g.H + 8 }, cp);
      const R = { clip };
      R.band = sv("rect", { x: g.X(30), y: g.top, width: g.X(E) - g.X(30), height: hPlot, class: "lr-evband", opacity: 0 }, s);
      [0, 0.05, 0.1, 0.15, 0.2].forEach((v) => {
        const yy = Math.round(g.Y(v)) + 0.5;
        sv("line", { x1: g.ml, x2: g.X(E), y1: yy, y2: yy, class: "lr-gl" + (v === 0 ? " lr-base" : "") }, s);
        stext(s, g.ml - 6, yy + 3.5, v === 0 ? "0" : v.toFixed(2), "lr-tick", "end");
      });
      for (let e = 0; e <= E; e += 10) stext(s, g.X(e), g.bot + 15, String(e), "lr-tick", "middle");
      stext(s, g.ml + g.pw / 2, g.bot + 28, "epoch", "lr-annm", "middle");
      sv("line", { x1: g.X(m.wall), x2: g.X(m.wall), y1: g.top, y2: g.bot, class: "lr-wall", "stroke-width": 1.25 }, s);
      const inner = sv("g", { "clip-path": `url(#${id}${m.key.replace(/\W/g, "")})` }, s);
      ARMS.forEach((X) => {
        const st = m.arms[X], up = [], lo = [], ln = [];
        for (let e = 0; e <= E; e++) { up.push([g.X(e), g.Y(st.chain[e] + st.sd[e])]); lo.push([g.X(e), g.Y(Math.max(0, st.chain[e] - st.sd[e]))]); ln.push([g.X(e), g.Y(st.chain[e])]); }
        sv("path", { d: pathOf2(up.concat(lo.reverse())) + "Z", class: "lr-band lr-f" + X }, inner);
        sv("path", { d: pathOf2(ln), class: "lr-line lr-s" + X, "stroke-width": 1.75 }, inner);
      });
      R.labs = sv("g", { opacity: 0 }, s);
      const gx = g.X(E) + 10;
      const items = dodge(ARMS.map((X) => ({ X, y: g.Y(m.arms[X].late) })), 12, g.top + 4, g.bot);
      items.forEach((d) => {
        const st = m.arms[d.X];
        sv("line", { x1: g.X(30), x2: g.X(E), y1: d.y, y2: d.y, class: "lr-line lr-s" + d.X, "stroke-width": 2, "stroke-opacity": 0.95 }, R.labs);
        if (Math.abs(d.ly - d.y) > 1) sv("path", { d: `M${g.X(E) + 2} ${d.y}L${gx - 2} ${d.ly}`, class: "lr-leader" }, R.labs);
        const t = stext(R.labs, gx, d.ly + 4, f3(st.late), "lr-lab");
        t.style.fontSize = "11px";
      });
      R.hair = sv("line", { x1: 0, x2: 0, y1: g.top, y2: g.bot, class: "lr-hair", opacity: 0 }, s);
      R.dots = {}; ARMS.forEach((X) => { R.dots[X] = sv("circle", { r: 3.5, class: "lr-ring lr-f" + X, opacity: 0 }, s); });
      const hit = sv("rect", { x: g.ml, y: g.top, width: g.pw + 4, height: hPlot, class: "lr-hit" }, s);
      hit.addEventListener("pointermove", (ev) => { const rc = s.getBoundingClientRect(); hoverE = M.clamp(Math.round(((ev.clientX - rc.left - g.ml) / g.pw) * E), 0, E); drawHover(pn); });
      hit.addEventListener("pointerdown", (ev) => { const rc = s.getBoundingClientRect(); hoverE = M.clamp(Math.round(((ev.clientX - rc.left - g.ml) / g.pw) * E), 0, E); drawHover(pn); });
      hit.addEventListener("pointerleave", () => { hoverE = null; drawHover(null); });
      pn.G = g; pn.R = R;
    }
    function pathOf2(pts) { return pts.map((p, i) => (i ? "L" : "M") + p[0].toFixed(1) + " " + p[1].toFixed(1)).join(""); }
    function drawHover(active) {
      const eDraw = E * Math.min(1, T / 0.7);
      panels.forEach((pn) => {
        const R = pn.R, g = pn.G;
        if (hoverE == null || hoverE > eDraw + 1e-6) { setOp(R.hair, 0); ARMS.forEach((X) => setOp(R.dots[X], 0)); pn.tip.style.display = "none"; return; }
        const xx = Math.round(g.X(hoverE)) + 0.5;
        R.hair.setAttribute("x1", xx); R.hair.setAttribute("x2", xx); setOp(R.hair, 1);
        ARMS.forEach((X) => { const c = R.dots[X]; c.setAttribute("cx", g.X(hoverE)); c.setAttribute("cy", g.Y(pn.m.arms[X].chain[hoverE])); setOp(c, 1); });
        if (pn !== active) { pn.tip.style.display = "none"; return; }
        const tp = pn.tip; tp.textContent = "";
        tp.appendChild(h("div", "lr-tt", `${pn.m.name} · epoch ${hoverE}`));
        const tb = h("table");
        ARMS.forEach((X) => {
          const st = pn.m.arms[X], tr = h("tr"), c0 = h("td"); c0.appendChild(key(X)); c0.appendChild(document.createTextNode(SHORT[X])); tr.appendChild(c0);
          const c1 = h("td"); c1.appendChild(h("b", null, f3(st.chain[hoverE]))); tr.appendChild(c1);
          tr.appendChild(h("td", null, "± " + f3(st.sd[hoverE]))); tb.appendChild(tr);
        });
        tp.appendChild(tb);
        tp.appendChild(h("div", "lr-tfoot", `Two-hop accuracy, mean ± sd of ${nS} seeds`));
        tp.style.display = "block";
        const tw = tp.offsetWidth, wr = pn.wrap.getBoundingClientRect(), cr = card.getBoundingClientRect();
        const minL = cr.left - wr.left + 8, maxR = cr.right - wr.left - 8;
        let left = g.X(hoverE) + 14;
        if (left + tw > maxR) left = g.X(hoverE) - 14 - tw;
        left = M.clamp(left, minL, Math.max(minL, maxR - tw));
        const top = pn.svg.getBoundingClientRect().top - pn.wrap.getBoundingClientRect().top;
        tp.style.left = left + "px"; tp.style.top = (top + g.top + 2) + "px";
      });
    }
    function render(t) {
      T = M.clamp(t, 0, 1);
      const eDraw = E * Math.min(1, T / 0.7), band = ramp(T, 0.7, 0.8), labs = ramp(T, 0.75, 0.9);
      panels.forEach((pn) => {
        if (!pn.R) return;
        pn.R.clip.setAttribute("width", Math.max(0, pn.G.X(eDraw) + 3));
        setOp(pn.R.band, band); setOp(pn.R.labs, labs);
      });
      if (hoverE != null) drawHover(null);
    }
    function layout() {
      const W = Math.round(el.clientWidth || 1100);
      if (W === lastW) return;
      lastW = W;
      root.classList.toggle("lr-small", W < 560);
      const cols = W >= 900 ? 4 : W >= 600 ? 2 : 1;
      gridEl.style.gridTemplateColumns = `repeat(${cols}, minmax(0, 1fr))`;
      const hPlot = cols === 4 ? 150 : cols === 2 ? 170 : 150;
      panels.forEach((pn) => buildPanel(pn, Math.max(180, pn.wrap.clientWidth), hPlot));
      render(T);
    }
    const ro = "ResizeObserver" in window ? new ResizeObserver(() => layout()) : null;
    if (ro) ro.observe(el); else window.addEventListener("resize", layout);
    bRe.addEventListener("click", () => { M.userPaused("lrsd-models", false); hasPlayed = true; render(0); playing = true; clock.start(); });
    layout();
    render(1);                           // poster: finished panels until the first scroll-in (or Replay) draws them from 0
    return {
      seek(t) { playing = false; hasPlayed = true; clock.stop(); render(t); },
      play() {
        if (playing) return;
        if (!hasPlayed) { hasPlayed = true; render(0); } else if (T >= 1) return;
        playing = true; clock.start();
      },
      pause() { playing = false; clock.stop(); },
    };
  }

  /* ------------------------------------------------------------------ registration */
  function guard(fn, name) {
    return function (el, data) {
      if (!data || !data.lrsd || !data.lrsd.models) throw new Error("M2G_DATA.lrsd is missing (" + name + ")");
      return fn(el, data);
    };
  }
  function component(factory, name) {
    let inst = null;
    return {
      mount(el, data) { inst = guard(factory, name)(el, data); },
      seek(t) { if (inst) inst.seek(t); },
      play() { if (inst) inst.play(); },
      pause() { if (inst) inst.pause(); },
    };
  }
  M.register("lrsd", component(Hero, "lrsd"));
  M.register("lrsd-models", component(Models, "lrsd-models"));
})();
