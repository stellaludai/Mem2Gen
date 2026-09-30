/* Mem2Gen project page: the self-patching explainer, "How new knowledge gets in, and how self-patching moves it".
 *
 * Registers component 'selfpatch' (mounted into #viz-selfpatch by js/common.js). Plain canvas + DOM, no dependencies.
 * Every number on screen is derived at mount from window.M2G_DATA.selfpatch (the two facts, the training log and the
 * epoch-14 self-patching map of item 10647); captions never hard-code a count.
 *
 * Storyboard: 48 s in four chapters (Inject, Ask, Patch, Scan); normalized time t in [0, 1]. Playback stretches each
 * scene to its caption's reading time (about 110 s at 1x); the scrubber shows playback time.
 * QA: window.__seek('selfpatch', t) renders frame t deterministically (no randomness, no accumulated state).
 * The tower of squares is a schematic; only ranks, counts and curves are measured.
 */
(function () {
  "use strict";
  const M = window.M2G;
  if (!M) return;

  /* ------------------------------------------------------------------ timing */
  const DUR = 48;                                   // seconds at 1x
  const CH_T = [0, 0.22, 0.36, 0.665];              // chapter starts
  const CH_NAMES = ["Inject", "Ask", "Patch", "Scan"];
  const T_LOOP = [0.07, 0.19];                      // epoch loop 0 -> demo epoch
  const T_PASS = [0.258, 0.286];                    // Ch2 forward pass
  const T_PULSE = [0.276, 0.29];                    // read-out pulse to the ladder
  const T_DROP = [0.29, 0.306];                     // answer chip drops into its slot
  const T_MAPIN = [0.671, 0.697];
  const T_FLY = [0.677, 0.704];                     // result tokens fly into their cells
  const T_DIAG = [0.705, 0.726];
  const T_DPULSE = [0.726, 0.74];
  const T_SCAN = [0.74, 0.88];
  const T_MARGIN = [0.88, 0.905];
  const T_BRACK = [0.893, 0.918];

  /* ------------------------------------------------------------------ helpers */
  const clamp = M.clamp, lerp = M.lerp, sm = M.ease, eIO = M.easeInOut;
  const seg = (t, a, b) => clamp((t - a) / (b - a), 0, 1);
  const eBack = (x) => { const c1 = 1.70158, c3 = c1 + 1; return 1 + c3 * Math.pow(x - 1, 3) + c1 * Math.pow(x - 1, 2); };
  const eExpo = (x) => (x >= 1 ? 1 : 1 - Math.pow(2, -10 * x));
  const bump = (x) => Math.sin(Math.PI * clamp(x, 0, 1));
  const fmt = (x) => x.toLocaleString("en-US");
  const ord = (r) => M.ordinal(r);
  const WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"];
  const word = (k) => WORDS[k] || String(k);
  const rankText = (v) => (v >= 0.19 ? ord(Math.round(1 / v)) : "below 5th");

  /* colour parsing (tokens may be any CSS colour) */
  let parseCtx = null;
  function rgbOf(css) {
    if (!parseCtx) parseCtx = document.createElement("canvas").getContext("2d");
    parseCtx.fillStyle = "#000";
    parseCtx.fillStyle = css;
    const s = parseCtx.fillStyle;
    if (s[0] === "#") return [1, 3, 5].map((i) => parseInt(s.slice(i, i + 2), 16));
    const m = s.match(/[\d.]+/g) || [0, 0, 0];
    return [+m[0], +m[1], +m[2]];
  }
  const rgba = (rgb, a) => `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${a})`;
  const mixRGB = (a, b, w) => a.map((c, i) => Math.round(c + (b[i] - c) * w));
  function lum(rgb) {
    const f = (c) => { c /= 255; return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); };
    return 0.2126 * f(rgb[0]) + 0.7152 * f(rgb[1]) + 0.0722 * f(rgb[2]);
  }
  const contrast = (a, b) => { const x = lum(a), y = lum(b); return (Math.max(x, y) + 0.05) / (Math.min(x, y) + 0.05); };

  function rr(ctx, x, y, w, h, r) {
    r = Math.min(r, w / 2, h / 2);
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }
  function bez(p0, p1, p2, p3, u) {
    const v = 1 - u;
    return [v * v * v * p0[0] + 3 * v * v * u * p1[0] + 3 * v * u * u * p2[0] + u * u * u * p3[0],
      v * v * v * p0[1] + 3 * v * v * u * p1[1] + 3 * v * u * u * p2[1] + u * u * u * p3[1]];
  }
  function h(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }
  /* text with every occurrence of `ent` wrapped in an entity span */
  function withEntity(parent, text, ent) {
    const parts = text.split(ent);
    parts.forEach((p, i) => {
      if (p) parent.appendChild(document.createTextNode(p));
      if (i < parts.length - 1) parent.appendChild(h("span", "sp-ent", ent));
    });
    return parent;
  }

  /* ------------------------------------------------------------------ component styles (fallbacks; page CSS wins) */
  const STYLE = `
.sp{--sp-sans:var(--font-ui,var(--font-sans,Inter,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif));--sp-serif:var(--font-body,var(--font-serif,"Source Serif 4",Georgia,serif));font-family:var(--sp-sans);color:var(--ink);max-width:1120px;margin:0 auto;text-align:left;-webkit-tap-highlight-color:transparent}
.sp *,.sp *::before,.sp *::after{box-sizing:border-box}
.sp-top{display:flex;align-items:center;justify-content:space-between;gap:8px 16px;flex-wrap:wrap;margin:0 0 10px}
.sp-rail{display:inline-flex;gap:2px;padding:3px;border-radius:999px;background:var(--surface-2);border:1px solid var(--border)}
.sp-chb{appearance:none;-webkit-appearance:none;border:0;margin:0;background:transparent;color:var(--ink-2);font:500 13px/1 var(--sp-sans);padding:9px 15px 10px;border-radius:999px;cursor:pointer;position:relative;display:inline-flex;align-items:center;justify-content:center;gap:6px;white-space:nowrap;min-height:34px}
.sp-chb:hover{color:var(--ink)}
.sp-chb .sp-n{font-variant-numeric:tabular-nums;color:var(--muted);font-weight:600}
.sp-chb[aria-current="step"]{background:var(--stage);color:var(--ink);box-shadow:0 0 0 1px var(--border),0 1px 3px rgba(0,0,0,.07)}
.sp-chb[aria-current="step"] .sp-n{color:var(--ink)}
.sp-chb .sp-bar{position:absolute;left:14px;right:14px;bottom:4px;height:2px;border-radius:2px;background:var(--ink-2);transform-origin:left center;transform:scaleX(0);opacity:.45;pointer-events:none}
.sp-badge{font-size:13px;color:var(--ink-2);font-variant-numeric:tabular-nums;white-space:nowrap}
.sp-badge b{color:var(--ink);font-weight:600}
.sp-stage{position:relative;display:grid;grid-template-columns:var(--sp-lp,272px) minmax(0,1fr);gap:var(--sp-gap,28px);padding:var(--sp-pad,20px);background:var(--stage);border:1px solid var(--border);border-radius:var(--r-l,14px);outline:none}
.sp-stage:focus-visible{box-shadow:0 0 0 2px var(--focus)}
.sp-panel{position:relative;min-width:0;display:flex;flex-direction:column;overflow:hidden}
.sp:not(.sp-narrow) .sp-panel{height:var(--sp-h,auto)}
.sp-main{position:relative;min-width:0;display:flex;justify-content:center;align-items:flex-start}
.sp-poster{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);z-index:3;appearance:none;-webkit-appearance:none;display:inline-flex;align-items:center;gap:10px;min-height:48px;padding:0 20px 0 17px;border-radius:999px;border:1px solid var(--border-strong,var(--border));background:var(--surface,var(--stage));color:var(--ink);font:600 15px/1 var(--sp-sans);white-space:nowrap;cursor:pointer;box-shadow:0 10px 30px rgba(0,0,0,.16),0 1px 3px rgba(0,0,0,.08)}
.sp-poster:hover{background:var(--hover-wash,var(--surface-2))}
.sp-poster svg{width:15px;height:15px;fill:currentColor;flex:none}
.sp-poster-d{font-weight:400;color:var(--muted);font-variant-numeric:tabular-nums}
.sp-poster[hidden]{display:none}
.sp-cv{display:block;touch-action:manipulation;-webkit-user-select:none;user-select:none}
.sp-blk{display:grid;grid-template-rows:1fr;flex:none}
.sp-blk>.sp-in{min-height:0;overflow:hidden}
.sp-card{border:1px solid var(--border);border-radius:10px;padding:9px 12px 10px;background:var(--stage);margin:0 0 10px}
.sp-card-h{display:flex;align-items:baseline;justify-content:space-between;gap:8px;min-height:16px}
.sp-eyebrow{font:600 10.5px/1.3 var(--sp-sans);letter-spacing:.07em;text-transform:uppercase;color:var(--muted)}
.sp-card-q{font-size:14px;line-height:1.42;color:var(--ink);margin-top:4px}
.sp-card-a{font-size:13px;line-height:1.4;color:var(--ink-2);margin-top:4px}
.sp-card-a b{font-weight:600;color:var(--ink)}
.sp-status{font:600 12px/1.3 var(--sp-sans);color:var(--ink-2);white-space:nowrap;display:inline-block;transform-origin:right center}
.sp-ok{color:var(--good)}.sp-bad{color:var(--bad);font-weight:700}
.sp-muted{color:var(--muted)}
.sp-facts-h,.sp-fn{display:none}
.sp-card-s{font-size:12.5px;line-height:1.4;color:var(--ink);margin-top:6px;padding-top:6px;border-top:1px solid var(--border)}
.sp-ent{font-weight:600;color:var(--entity);color:color-mix(in oklab,var(--entity) 72%,var(--ink))}
.sp-chain{font-size:13px;line-height:1.6;color:var(--ink);margin:0 0 12px;padding:0 2px}
.sp-chain .sp-rel{color:var(--muted);font-size:12px;white-space:nowrap}
.sp-bridge{border:1px solid var(--axis);border-radius:5px;padding:0 4px;white-space:nowrap}
.sp-chain.is-q .sp-bridge{border-style:dashed;color:var(--ink-2)}
.sp-btag{display:block;font-size:11.5px;color:var(--muted);line-height:1.35;margin-top:2px}
.sp-spark{margin-top:auto;padding-top:6px}
.sp-spark-h{display:flex;justify-content:space-between;align-items:baseline;margin:0 0 4px}
.sp-spark-n{font-size:12px;color:var(--ink-2);font-variant-numeric:tabular-nums}
.sp-spark canvas{display:block;width:100%}
.sp-spark-note{font-size:12px;line-height:1.4;color:var(--ink-2);margin-top:2px}
.sp-stats{margin:2px 0 12px}
.sp-stats ul{list-style:none;margin:6px 0 0;padding:0}
.sp-stats li{display:flex;align-items:baseline;gap:8px;font-size:13.5px;line-height:1.5;color:var(--ink-2)}
.sp-stats li b{font-weight:600;color:var(--ink);font-variant-numeric:tabular-nums;min-width:2.2em;text-align:right}
.sp-prov{font-size:12px;line-height:1.45;color:var(--muted);margin:8px 0 0}
.sp-sw{flex:none;width:10px;height:10px;border-radius:2px;box-shadow:inset 0 0 0 1px var(--border-strong,rgba(0,0,0,.2));transform:translateY(1px)}
.sp-cta{margin:0 0 10px}
.sp-ctabtn{appearance:none;-webkit-appearance:none;border:0;cursor:pointer;display:inline-flex;align-items:center;gap:8px;min-height:38px;padding:0 16px;white-space:nowrap;max-width:100%;border-radius:999px;background:var(--btn-bg,var(--ink));color:var(--btn-ink,var(--bg));font:500 14px/1 var(--sp-sans)}
.sp-ctabtn:hover{background:var(--btn-bg-hover,var(--ink-2))}
.sp-ctanote{font-size:12px;line-height:1.45;color:var(--muted);margin:7px 0 0}
.sp-tip{position:absolute;z-index:5;pointer-events:none;max-width:260px;background:var(--tip-bg,var(--surface));color:var(--ink);border:1px solid var(--border);border-radius:8px;padding:7px 10px;font:400 12.5px/1.4 var(--sp-sans);box-shadow:var(--tip-shadow,0 6px 20px rgba(0,0,0,.12));font-variant-numeric:tabular-nums}
.sp-tip[hidden]{display:none}
.sp-tip b{font-weight:600}
.sp-cap{font:500 15px/1.5 var(--sp-sans);color:var(--ink);margin:14px 2px 6px;max-width:62ch;min-height:4.5em;text-wrap:pretty}
.sp-cap.is-big{font-size:17.5px;font-weight:600;line-height:1.4}
.sp-ctl{display:flex;align-items:center;gap:10px 12px;flex-wrap:wrap;margin:4px 0 8px}
.sp-btns{display:flex;gap:6px;flex:none}
.sp-ib{appearance:none;-webkit-appearance:none;width:34px;height:34px;border-radius:50%;border:1px solid var(--border);background:var(--surface-2);color:var(--ink);display:inline-flex;align-items:center;justify-content:center;cursor:pointer;padding:0}
.sp-ib:hover{background:var(--hover-wash,var(--surface-2));border-color:var(--border-strong,var(--border))}
.sp-ib svg{width:14px;height:14px;fill:currentColor}
.sp-ib[disabled]{opacity:.4;cursor:default}
.sp-scrub{position:relative;flex:1 1 200px;min-width:0;height:34px;display:flex;align-items:center}
.sp-range{-webkit-appearance:none;appearance:none;width:100%;height:20px;margin:0;background:transparent;cursor:pointer}
.sp-range::-webkit-slider-runnable-track{height:4px;border-radius:2px;background:linear-gradient(90deg,var(--ink-2) 0 var(--sp-fill,0%),var(--grid) var(--sp-fill,0%) 100%)}
.sp-range::-moz-range-track{height:4px;border-radius:2px;background:var(--grid)}
.sp-range::-moz-range-progress{height:4px;border-radius:2px;background:var(--ink-2)}
.sp-range::-webkit-slider-thumb{-webkit-appearance:none;width:14px;height:14px;border-radius:50%;background:var(--ink);border:2px solid var(--bg);margin-top:-5px;box-shadow:0 0 0 1px var(--border)}
.sp-range::-moz-range-thumb{width:12px;height:12px;border-radius:50%;background:var(--ink);border:2px solid var(--bg)}
.sp-ticks{position:absolute;left:7px;right:7px;top:50%;height:0;pointer-events:none}
.sp-ticks i{position:absolute;top:5px;width:1px;height:5px;background:var(--axis)}
.sp-speed{display:inline-flex;padding:2px;border-radius:999px;background:var(--surface-2);border:1px solid var(--border);flex:none}
.sp-speed button{appearance:none;-webkit-appearance:none;border:0;background:transparent;color:var(--ink-2);font:500 12px/1 var(--sp-sans);padding:7px 9px;border-radius:999px;cursor:pointer;font-variant-numeric:tabular-nums}
.sp-speed button[aria-pressed="true"]{background:var(--stage);color:var(--ink);box-shadow:0 0 0 1px var(--border)}
.sp :focus-visible{outline:2px solid var(--focus);outline-offset:2px}
.sp-sr{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}
.sp-narrow .sp-top{flex-direction:column;align-items:stretch;gap:6px}
.sp-narrow .sp-rail{display:flex;border-radius:12px}
.sp-narrow .sp-chb{flex:1 1 0;min-height:44px;padding:0 4px;border-radius:9px;font-size:12.5px;gap:4px}
.sp-narrow .sp-chb .sp-bar{left:10px;right:10px}
.sp-narrow .sp-badge{white-space:normal;font-size:12.5px;padding:0 2px}
.sp-narrow .sp-stage{grid-template-columns:minmax(0,1fr);gap:8px}
.sp-narrow .sp-panel{height:var(--sp-slot,160px)}
.sp-narrow .sp-spark{display:none}
.sp-narrow .sp-card{padding:7px 10px 8px;margin-bottom:8px}
.sp-narrow .sp-card-q{font-size:13px;margin-top:2px}
.sp-narrow .sp-card-a{font-size:12.5px;margin-top:2px}
.sp-narrow .sp-chain{font-size:12.5px;margin-bottom:8px}
.sp-narrow .sp-qexp{display:none}
.sp-narrow .sp-card-s{margin-top:4px;padding-top:4px}
.sp-narrow .sp-cap{font-size:14.5px;margin:0 2px 8px;max-width:none}
.sp-narrow .sp-cap.is-big{font-size:16.5px}
.sp-narrow .sp-facts{border:1px solid var(--border);border-radius:10px;padding:7px 10px 8px;background:var(--stage);margin:0}
.sp-narrow .sp-blk .sp-card{margin-bottom:0}
.sp-narrow .sp-facts-h{display:flex}
.sp-narrow .sp-fact{border:0;border-radius:0;padding:0;margin:3px 0 0;background:none;font-size:12.5px;line-height:1.42}
.sp-narrow .sp-fact .sp-card-h{display:none}
.sp-narrow .sp-fact .sp-card-q,.sp-narrow .sp-fact .sp-card-a{display:inline;font-size:12.5px;line-height:1.42;margin:0}
.sp-narrow .sp-fact .sp-card-a{margin-left:.25em}
.sp-narrow .sp-fn{display:inline;font:600 10px/1 var(--sp-sans);letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin-right:.5em}
.sp-narrow .sp-card-s{margin-top:3px;padding-top:3px;border-top:0}
.sp-narrow .sp-stats{margin:0 0 8px}
.sp-narrow .sp-stats ul{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin-top:4px}
.sp-narrow .sp-stats li{display:grid;grid-template-columns:auto 1fr;align-items:center;gap:0 6px;font-size:12px;line-height:1.3}
.sp-narrow .sp-stats li b{min-width:0;text-align:left;font-size:16px}
.sp-narrow .sp-stats .sp-slab{grid-column:1 / -1}
.sp-narrow .sp-wo,.sp-narrow .sp-ctanote{display:none}
.sp-narrow .sp-prov{margin-top:6px;font-size:11.5px}
.sp-narrow .sp-cta{margin:0}
.sp-narrow .sp-ctl{gap:4px 10px;margin:2px 0 0}
.sp-narrow .sp-scrub{order:3;flex:1 1 100%;height:28px}
.sp-narrow .sp-btns{order:1}
.sp-narrow .sp-speed{order:2;margin-left:auto}
.sp-narrow .sp-ib{width:44px;height:44px}
.sp-narrow .sp-speed button{padding:10px 11px}
`;
  function injectStyle() {
    if (document.getElementById("sp-style")) return;
    const s = document.createElement("style");
    s.id = "sp-style";
    s.textContent = STYLE;
    // Prepend so the page's own shared control classes (style.css) win on equal specificity.
    document.head.insertBefore(s, document.head.firstChild);
  }

  const ICON = {
    prev: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3.5 3h1.6v10H3.5zM13 3.2v9.6L6 8z"/></svg>',
    next: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M10.9 3h1.6v10h-1.6zM3 3.2v9.6L10 8z"/></svg>',
    play: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4.5 2.6v10.8L13 8z"/></svg>',
    pause: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 3h2.8v10H4zM9.2 3H12v10H9.2z"/></svg>',
    replay: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 2.5a5.5 5.5 0 1 1-5.2 7.3l1.5-.5A3.9 3.9 0 1 0 8 4.1V6L4.6 3.3 8 .6z"/></svg>',
  };

  /* ================================================================== component state */
  let D, X, N, E, SC, EXPS;
  let host, root, stageEl, panelEl, mainEl, cv, capEl, rangeEl, playBtn, prevBtn, nextBtn, tipEl, badgeEl, sparkCv, sparkN;
  let railBtns = [], railBars = [], speedBtns = [];
  const dom = {};
  let geo = null, C = null, font = "sans-serif";
  let T = 0, speed = 1, stopAt = null, clk = null, dragging = false, mounted = false;
  let hover = null;        // {kind:'map', s, g} | {kind:'tower', col, l} | {kind:'chip'} | {kind:'spark', e}
  let pinned = null;       // {s, g} map cell pinned by click / Enter
  let lastScene = -1, lastW = 0;
  let mapCache = null, phCache = null, cacheKey = "";
  const memo = new Map();
  let memoSeq = 0;

  /* ================================================================== data */
  function derive() {
    const n = D.n_layers, G = M.decodeGrid(D.demo.grid, n);
    const u = D.demo.unpatched_rank, d0 = 1 / u;
    const same = (v) => Math.abs(v - d0) < 0.01;
    const vAt = (s, g) => G[s * n + g];
    let n1 = 0, nSame = 0, nWorse = 0, nDiag1 = 0, diagOk = true;
    const colCount = new Array(n).fill(0);
    const cum1 = new Int32Array(n * n + 1), cumSame = new Int32Array(n * n + 1), cumWorse = new Int32Array(n * n + 1);
    for (let k = 0; k < n * n; k++) {
      const s = Math.floor(k / n), g = k % n, v = G[k];
      if (v >= M.FULL) { n1++; colCount[g]++; if (s === g) nDiag1++; }
      else if (same(v)) nSame++;
      else if (v < d0 - 0.01) nWorse++;
      if (s === g && !same(v)) diagOk = false;
      cum1[k + 1] = n1; cumSame[k + 1] = nSame; cumWorse[k + 1] = nWorse;
    }
    let cStar = n;
    for (let c = n - 1; c >= 0 && colCount[c] / n < 0.1; c--) cStar = c;
    let n1Below = 0;
    for (let c = 0; c < cStar; c++) n1Below += colCount[c];
    const L = D.demo.late, Ea = D.demo.early;
    let ctrl = Math.round(0.875 * n);
    if (!(ctrl !== L.src && ctrl !== L.tgt && ctrl < n && same(vAt(L.src, ctrl)))) {
      ctrl = -1;
      for (let dd = 1; dd < n && ctrl < 0; dd++) {
        const g = L.src + dd;
        if (g < n && same(vAt(L.src, g))) ctrl = g;
      }
      if (ctrl < 0) for (let g = 0; g < n && ctrl < 0; g++) if (g !== L.tgt && same(vAt(L.src, g))) ctrl = g;
      if (ctrl < 0) ctrl = Math.min(n - 1, L.src + 3);
    }
    const rk = (v) => (v > 0 ? Math.round(1 / v) : 99);
    const out = {
      n, G, u, d0, n1, nSame, nWorse, nDiag1, colCount, cum1, cumSame, cumWorse, cStar, n1Below, n1Above: n1 - n1Below, ctrl,
      maxCol: Math.max(1, ...colCount),
      lateV: vAt(L.src, L.tgt), earlyV: vAt(Ea.src, Ea.tgt), ctrlV: vAt(L.src, ctrl),
    };
    out.lateRank = rk(out.lateV); out.earlyRank = rk(out.earlyV); out.ctrlRank = rk(out.ctrlV);
    if (!diagOk) console.warn("selfpatch: diagonal cells differ from the unpatched rank", u);
    if (out.lateRank !== 1 || out.earlyRank !== 1) console.warn("selfpatch: demo cells are not rank 1", out.lateRank, out.earlyRank);
    if (out.lateRank !== L.rank || out.earlyRank !== Ea.rank) console.warn("selfpatch: demo ranks disagree with the map");
    return out;
  }

  function buildScenes() {
    const n = N, tm = D.t_mem, L = D.demo.late, Ea = D.demo.early;
    const u = ord(X.u), tM = T_LOOP[0] + (T_LOOP[1] - T_LOOP[0]) * (tm / (E + 1));
    const wrongE = D.gen[E] < 0.5;
    const almost = X.n1 > 0 && X.n1Below / X.n1 >= 0.9;
    // Captions stay short (<= 16 words, <= ~22 for the Ch4 closers); playback also stretches each scene to its reading time.
    const statsCap = `${fmt(X.n1)} of ${fmt(n * n)} pairs rank the answer's first token 1st, ` +
      `${X.nDiag1 === 0 ? "none" : `${X.nDiag1}`} on the diagonal. ` +
      `${almost ? "Almost all" : `${fmt(X.n1Below)} of them`} write into layers 0–${X.cStar - 1}` +
      `${X.cStar <= Math.ceil(n / 2) ? ", all within the first half of the network" : ""}.`;
    const ctrlSame = X.ctrlRank === X.u;
    const list = [
      [0, 0, `The model is a stack of ${n} layers. Each token's hidden state is rewritten at every layer.`],
      [0.035, 0, `We fine-tune ${D.model} on two facts it did not know, and nothing else.`],
      [T_LOOP[0], 0, `Each epoch the two facts pass through the model; the update touches every layer.`],
      [tM, 0, `Epoch ${tm}: both facts are recalled. But where in the ${n} layers does the new knowledge sit?`],
      [T_LOOP[1], 0, `We stop at epoch ${E}, ${word(E - tm)} epochs after memorization. The model answers either fact.`],
      [0.22, 1, `Now a question that needs both facts together. It never appeared in training.`],
      [0.255, 1, `Each square is one token's hidden state at one layer (schematic). The answer is predicted at the end.`],
      [0.29, 1, `The answer's first token ranks only ${u}${wrongE ? ", and the model's own answer is wrong" : ""}.`],
      [0.33, 1, `Is the knowledge missing, or stored at a layer the question can't use?`, true],
      [0.36, 2, `Self-patching: run the same question and read ${D.head}'s hidden state after layer ${L.src}…`],
      [0.39, 2, `…then rerun the same prompt and write that state into ${D.head} at layer ${L.tgt}.`],
      [0.44, 2, `The forward pass continues from layer ${L.tgt + 1}. Only ${D.head} and the tokens after it can change.`],
      [0.475, 2, `The answer's first token now ranks ${ord(X.lateRank)}. Nothing was added: only the model's own state, moved.`],
      [0.525, 2, `An early state works too: layer ${Ea.src} → layer ${Ea.tgt} also ranks it ${ord(X.earlyRank)}.`],
      [0.58, 2, `Control: write the same layer-${L.src} state into layer ${X.ctrl} instead…`],
      [EXPS[2].sw[0], 2, ctrlSame
        ? `…and nothing changes: still ${u}. Where the state is written matters.`
        : `…and the answer's first token ranks ${rankText(X.ctrlV)}. Where the state is written matters.`],
      [0.635, 2, `Three layer pairs, three experiments. Now try every pair.`],
      [0.665, 3, `Each (read, write) pair is one cell, colored by the rank of the answer's first token.`],
      [T_DIAG[0], 3, `Writing a layer's state back into itself changes nothing: the diagonal is the unpatched model (${u}).`],
      [T_SCAN[0], 3, `Scanning all ${n} × ${n} = ${fmt(n * n)} pairs of this one epoch-${E} checkpoint.`],
      [T_SCAN[1], 3, statsCap],
      [0.94, 3, D.t_gen != null
        ? `Rank-1 cells off the diagonal: the information is stored and extractable by relocation, yet the model only uses it unaided from epoch ${D.t_gen}.`
        : `Rank-1 cells off the diagonal: the information is stored and extractable by relocation, yet the model does not use it on its own.`],
    ];
    return list.map((r, i) => ({ t0: r[0], t1: i + 1 < list.length ? list[i + 1][0] : 1, ch: r[1], cap: r[2], big: !!r[3] }));
  }

  function buildExps() {
    const L = D.demo.late, Ea = D.demo.early;
    return [
      { src: L.src, tgt: L.tgt, v: X.lateV, t0: 0.36,
        read: [0.364, 0.386], mv: [0.392, 0.436], rc: [0.442, 0.468], pl: [0.468, 0.477], sw: [0.477, 0.4895], flare: [0.482, 0.515] },
      { src: Ea.src, tgt: Ea.tgt, v: X.earlyV, t0: 0.525, reset: [0.525, 0.531],
        read: [0.531, 0.539], mv: [0.541, 0.556], rc: [0.557, 0.565], pl: [0.565, 0.568], sw: [0.568, 0.5745], flare: [0.57, 0.585] },
      { src: L.src, tgt: X.ctrl, v: X.ctrlV, t0: 0.58, reset: [0.58, 0.586], ctrl: true,
        read: [0.586, 0.594], mv: [0.596, 0.611], rc: [0.612, 0.62], pl: [0.62, 0.623], sw: [0.623, 0.633] },
    ];
  }

  /* Playback clock. The storyboard (t in [0, 1], DUR s at 1x) sets where things happen; playback stretches each scene
     to at least its caption's reading time (1.2 s + 3.3 words/s), mapping wall-clock seconds to t piecewise-linearly.
     __seek and every drawing function keep working in storyboard t. */
  let WP = [], WT = DUR;
  function buildWarp() {
    let w = 0;
    WP = SC.map((s) => {
      const words = s.cap.split(/\s+/).filter(Boolean).length;
      const sec = Math.max((s.t1 - s.t0) * DUR, 1.2 + words / 3.3);
      const r = { t0: s.t0, t1: s.t1, w0: w, w1: w + sec };
      w += sec;
      return r;
    });
    WT = w;
  }
  function wOfT(t) {
    t = clamp(t, 0, 1);
    for (let i = 0; i < WP.length; i++) {
      const p = WP[i];
      if (t < p.t1 || i === WP.length - 1) return p.w0 + (p.w1 - p.w0) * clamp((t - p.t0) / (p.t1 - p.t0), 0, 1);
    }
    return WT;
  }
  function tOfW(w) {
    if (w >= WT) return 1;
    for (let i = 0; i < WP.length; i++) {
      const p = WP[i];
      if (w < p.w1 || i === WP.length - 1) return p.t0 + (p.t1 - p.t0) * clamp((w - p.w0) / (p.w1 - p.w0), 0, 1);
    }
    return 1;
  }

  const epochAt = (t) => (t < T_LOOP[0] ? 0 : Math.min(E, Math.floor((E + 1) * seg(t, T_LOOP[0], T_LOOP[1]))));
  const sceneAt = (t) => { for (let i = SC.length - 1; i >= 0; i--) if (t >= SC[i].t0) return i; return 0; };
  const chapterAt = (t) => { for (let i = CH_T.length - 1; i >= 0; i--) if (t >= CH_T[i]) return i; return 0; };

  /* ================================================================== theme */
  function readTokens() {
    const cs = getComputedStyle(root);
    const g = (n, fb) => cs.getPropertyValue(n).trim() || fb;
    const c = {
      ink: g("--ink", "#0b0b0b"), ink2: g("--ink-2", "#52514e"), muted: g("--muted", "#898781"), grid: g("--grid", "#e1e0d9"),
      axis: g("--axis", "#c3c2b7"), stage: g("--stage", "#ffffff"), surface2: g("--surface-2", "#f6f5f2"),
      mem: g("--mem", "#1f77b4"), gen: g("--gen", "#b03f3f"), good: g("--good", "#0ca30c"), bad: g("--bad", "#d03b3b"),
      entity: g("--entity", "#1baf7a"), band: g("--band", "#e8f1f8"),
    };
    c.rgb = {};
    Object.keys(c).forEach((k) => { if (k !== "rgb") c.rgb[k] = rgbOf(c[k]); });
    c.dark = M.isDark();
    c.entText = (() => { const m = mixRGB(c.rgb.entity, c.rgb.ink, 0.28); return `rgb(${m})`; })();
    c.diag = c.dark ? rgba(c.rgb.ink, 0.55) : rgba(c.rgb.ink, 0.82);
    C = c;
    font = cs.fontFamily || "system-ui, sans-serif";
    mapCache = null; cacheKey = "";
  }
  const F = (size, w) => `${w || 400} ${size}px ${font}`;
  function chipInk(rgb) {
    const white = [255, 255, 255], ink = [11, 11, 11];
    return contrast(rgb, ink) >= contrast(rgb, white) ? "#0b0b0b" : "#ffffff";
  }

  /* ================================================================== layout */
  function computeGeo() {
    const Wr = root.clientWidth;
    if (!Wr) return false;
    const n = N, g = { n };
    const dpr = Math.min(window.devicePixelRatio || 1, 3);
    const snap = (x) => (dpr >= 2 ? Math.floor(x * 2) / 2 : Math.floor(x));
    g.wide = Wr >= 900;
    root.classList.toggle("sp-narrow", !g.wide);
    if (g.wide && capEl.previousElementSibling !== stageEl) stageEl.after(capEl);
    else if (!g.wide && capEl.previousElementSibling !== dom.top) dom.top.after(capEl);
    if (g.wide) {
      const lp = Wr >= 1040 ? 272 : 250, gap = Wr >= 1040 ? 28 : 20, pad = 20;
      stageEl.style.setProperty("--sp-lp", lp + "px");
      stageEl.style.setProperty("--sp-gap", gap + "px");
      stageEl.style.setProperty("--sp-pad", pad + "px");
      g.W = Math.floor(Math.min(Wr, 1120) - 2 * pad - 2 - lp - gap);
      const full = g.W >= 730;
      g.pitch = 13; g.cellH = 10; g.ty0 = 46; g.axisW = 30;
      const pw = full ? 16 : 12, pg = 4, hw = full ? 56 : 48, ew = full ? 40 : 34, cg = full ? 12 : 10;
      g.cols = [];
      let x = g.axisW;
      for (let i = 0; i < 4; i++) { g.cols.push({ key: "pre", i, x, w: pw }); x += pw + (i < 3 ? pg : cg); }
      g.cols.push({ key: "head", x, w: hw }); x += hw + cg;
      g.cols.push({ key: "end", x, w: ew }); x += ew;
      g.towerR = x; g.tickX = x + 5; g.gutX = x + 14; g.gutW = full ? 92 : 84;
      g.rx0 = g.gutX + g.gutW + 10; g.rw = g.W - g.rx0;
      g.towerB = g.ty0 + n * g.pitch;
      g.lad = { x: g.rx0 + 2, y0: g.ty0 - 3, pitch: 50, h: 38, lab: 40 };
      g.lad.w = Math.min(172, Math.round(g.rw * 0.42));
      g.chipW = Math.min(112, g.lad.w - g.lad.lab - 10); g.chipH = 28;
      const c = clamp(snap((g.rw - 40 - 74) / n), 6, 10.5);
      g.map = { x: g.rx0 + 40, y: 94, c, S: c * n };
      g.leg = { x: g.map.x + g.map.S + 14, y: g.map.y, w: 10, h: g.map.S };
      g.train = { x: g.rx0 + 38, y: 92, w: g.rw - 44, h: 280 };
      g.tok = { x: g.lad.x + g.lad.w + 24, y: g.lad.y0 + 3 * g.lad.pitch + 4, w: Math.min(128, g.rw - (g.lad.x - g.rx0) - g.lad.w - 30), dy: 34 };
      g.big = { x: g.lad.x + g.lad.w + 24, y: g.lad.y0 + g.lad.pitch * 0.85 };
      g.H = Math.max(g.towerB + 44, g.map.y + g.map.S + 40);
      g.bulge = 30;
    } else {
      const pad = Wr < 420 ? 8 : 14;
      stageEl.style.setProperty("--sp-pad", pad + "px");
      g.W = Math.floor(Math.min(Wr - 2 * pad - 2, 560));
      const k = g.W / 340;
      // Ch4 map (header, one-line brackets and 14 px marginal bars above it; axis titles and the legend below)
      const c = clamp(snap((g.W - 38) / n), 6, 11);
      g.map = { c, S: c * n, y: 60 };
      g.map.x = Math.round(36 + (g.W - 36 - g.map.S) / 2);
      const mapB = g.map.y + g.map.S + 30;
      g.legTitleY = mapB + 10;
      g.leg = { horiz: true, x: 6, y: mapB + 21, w: g.W - 12, h: 10 };
      const Hmap = g.leg.y + g.leg.h + 19;
      // Ch1-3 tower: rows sized so tower + ladder + result tokens take no more height than the map
      g.ty0 = 44; g.axisW = 26;
      const below = 40 + 20 + 30 + 32 + 16;
      g.pitch = clamp(Math.floor(((Hmap - g.ty0 - below) / n) * 2) / 2, 6, 11);
      g.cellH = Math.max(4, g.pitch - 2.5);
      const pw = Math.round(62 * k), hw = Math.round(84 * k), ew = Math.round(50 * k), cg = 8;
      g.cols = [{ key: "pre", i: 0, x: g.axisW, w: pw }];
      let x = g.axisW + pw + cg;
      g.cols.push({ key: "head", x, w: hw }); x += hw + cg;
      g.cols.push({ key: "end", x, w: ew }); x += ew;
      g.towerR = x; g.tickX = x + 4; g.gutX = x + 11; g.gutW = g.W - g.gutX;
      g.towerB = g.ty0 + n * g.pitch;
      const y0 = g.towerB + 40;
      g.lad = { horiz: true, y0, titleY: y0 + 8, y: y0 + 20, h: 30, pitch: g.W / 6 };
      g.lad.w = g.lad.pitch - 6;
      g.chipW = Math.min(64, g.lad.w - 6); g.chipH = 24;
      g.train = { x: 30, y: y0 + 24, w: g.W - 40, h: 46 };
      g.tokY = g.lad.y + g.lad.h + 32;
      g.tok = { w: Math.floor((g.W - 16) / 3), dy: 0 };
      g.H = Math.ceil(Math.max(Hmap, g.tokY + 16, g.train.y + g.train.h + 22));
      g.bulge = 22;
    }
    g.rowY = (l) => g.ty0 + (n - 1 - l) * g.pitch;
    g.rowCy = (l) => g.ty0 + (n - 1 - l) * g.pitch + g.cellH / 2;
    g.head = g.cols.find((c) => c.key === "head");
    g.end = g.cols.find((c) => c.key === "end");
    geo = g;
    cv.style.width = g.W + "px";
    cv.style.height = g.H + "px";
    mapCache = null; cacheKey = "";
    fitCaption();
    return true;
  }

  /* Keep the caption box at the height of the longest caption, so the page never jumps while the story plays. */
  function fitCaption() {
    if (!capEl || !capEl.clientWidth) return;
    const probe = capEl.cloneNode(false);
    probe.removeAttribute("id");
    probe.setAttribute("aria-hidden", "true");
    probe.style.cssText = `position:absolute;visibility:hidden;left:-9999px;top:0;width:${capEl.clientWidth}px;min-height:0`;
    root.appendChild(probe);
    let mx = 0;
    SC.forEach((s) => { probe.className = "sp-cap" + (s.big ? " is-big" : ""); probe.textContent = s.cap; mx = Math.max(mx, probe.offsetHeight); });
    root.removeChild(probe);
    capEl.style.minHeight = mx + "px";
  }

  /* ================================================================== map caches */
  function ensureCaches() {
    const dpr = Math.min(window.devicePixelRatio || 1, 3);
    const key = [geo.map.c, dpr, C.dark, C.grid].join("|");
    if (cacheKey === key && mapCache) return;
    cacheKey = key;
    const n = N, c = geo.map.c, S = Math.round(c * n * dpr);
    const mk = () => { const o = document.createElement("canvas"); o.width = S; o.height = S; const x = o.getContext("2d"); x.setTransform(dpr, 0, 0, dpr, 0, 0); return [o, x]; };
    const gap = c >= 8 ? 1 : 0.75;
    const [a, ax] = mk();
    const hc = new Map();
    for (let s = 0; s < n; s++) for (let g = 0; g < n; g++) {
      const v = X.G[s * n + g], key2 = Math.round(v * 255);
      if (!hc.has(key2)) hc.set(key2, M.heat(v));
      ax.fillStyle = hc.get(key2);
      ax.fillRect(g * c + gap / 2, s * c + gap / 2, c - gap, c - gap);
    }
    const [p, px] = mk();
    px.fillStyle = rgba(C.rgb.grid, C.dark ? 0.55 : 0.45);
    for (let s = 0; s < n; s++) for (let g = 0; g < n; g++) px.fillRect(g * c + gap / 2, s * c + gap / 2, c - gap, c - gap);
    mapCache = a; phCache = p;
  }

  /* ================================================================== tower state */
  function activeExp(t) {
    if (t < EXPS[0].t0 || t >= CH_T[3] + 0.016) return null;
    let i = 0;
    for (let k = 0; k < EXPS.length; k++) if (t >= EXPS[k].t0) i = k;
    return i;
  }

  /* State of one single-pair experiment at time t (pure). */
  function expState(ex, t) {
    const n = N, st = { ex, fade: 1 };
    const pr = seg(t, ex.read[0], ex.read[1]);
    const pm = seg(t, ex.mv[0], ex.mv[1]);
    const pc = seg(t, ex.rc[0], ex.rc[1]);
    const f1 = lerp(-1.5, ex.src, eIO(pr));
    if (t < ex.mv[0]) {
      st.tint = f1;
      if (pr > 0 && pr < 1) st.band = { dir: 1, front: f1 };
    } else {
      const f2 = lerp(-1.5, ex.tgt, eIO(seg(pm, 0.06, 0.86)));
      st.tint = f2;
      st.tint1 = ex.src; st.tint1A = 1 - sm(seg(pm, 0, 0.14));
      if (pm < 0.86) st.band = { dir: 1, front: f2 };
    }
    if (t >= ex.rc[0]) {
      const f3 = lerp(ex.tgt + 0.5, n + 2.5, eIO(pc));
      st.tint = Math.max(st.tint, ex.tgt);
      st.tintPre = Math.min(f3, n - 1);
      st.hatch = { from: ex.tgt + 1, to: Math.min(f3, n - 1) };
      if (pc < 1) st.band = { dir: 1, front: f3, he: true };
      st.rcTag = sm(seg(pc, 0, 0.35));
    }
    st.ring = sm(seg(t, ex.read[1] - 0.003, ex.read[1] + 0.002));
    st.socket = sm(seg(pm, 0.0, 0.2));
    st.arcP = eIO(seg(pm, 0.12, 0.88));
    st.arcA = pm > 0 ? 1 : 0;
    if (pm > 0 && pm < 1) st.cap = { p: st.arcP, a: sm(seg(pm, 0, 0.1)), lift: bump(seg(pm, 0, 0.24)) * 0.12 + bump(seg(pm, 0.86, 1)) * 0.08 };
    st.res = sm(seg(t, (ex.sw[0] + ex.sw[1]) / 2, ex.sw[1] + 0.002));   // the result, shown next to the write label
    st.old = 1 - sm(seg(pm, 0.68, 0.88));
    st.patched = sm(seg(pm, 0.86, 1));
    if (t >= ex.pl[0] && t < ex.pl[1] + 0.005) st.pulse = { p: seg(t, ex.pl[0], ex.pl[1]), a: 1 - seg(t, ex.pl[1], ex.pl[1] + 0.005) };
    return st;
  }

  function towerState(t) {
    const n = N;
    let s = { fade: 1 };
    // scene 0: one forward pass, bottom to top
    if (t > T_INTRO[0] && t < T_INTRO[1]) s.band = { dir: 1, front: lerp(-1.5, n + 3, eIO(seg(t, T_INTRO[0], T_INTRO[1]))) };
    // Ch1: epoch loop, a forward band rises and an update band falls through every layer
    if (t >= T_LOOP[0] && t < T_LOOP[1]) {
      const ef = (E + 1) * seg(t, T_LOOP[0], T_LOOP[1]), e = Math.floor(ef), f = ef - e;
      if (e < E) {
        if (f < 0.5) s.band = { dir: 1, front: lerp(-1.5, n + 3, f / 0.5) };
        else { const fr = lerp(n + 2, -3, (f - 0.5) / 0.5); s.band = { dir: -1, front: fr }; s.ticks = fr; }
      }
    }
    // Ch2: one forward pass on the two-hop question
    if (t >= T_PASS[0] && t < 0.366) {
      const fr = lerp(-1.5, n + 3, eIO(seg(t, T_PASS[0], T_PASS[1])));
      if (t < T_PASS[1]) s.band = { dir: 1, front: fr };
      s.tint = Math.min(fr, n - 1);
      s.tintA = 1 - sm(seg(t, 0.36, 0.366));
    }
    if (t >= T_PULSE[0] && t < T_PULSE[1] + 0.005) s.pulse = { p: seg(t, T_PULSE[0], T_PULSE[1]), a: 1 - seg(t, T_PULSE[1], T_PULSE[1] + 0.005) };
    // Ch3: the three experiments (the previous one fades out during the next one's reset)
    const i = activeExp(t);
    if (i != null) {
      const ex = EXPS[i];
      if (ex.reset && t < ex.reset[1]) {
        s = expState(EXPS[i - 1], EXPS[i - 1].sw[1] + 0.001);
        s.fade = 1 - sm(seg(t, ex.reset[0], ex.reset[1]));
      } else if (t >= CH_T[3]) {
        s = expState(ex, ex.sw[1] + 0.001);
        s.fade = 1 - sm(seg(t, CH_T[3], CH_T[3] + 0.015));
      } else {
        s = expState(ex, t);
      }
    }
    return s;
  }

  /* pair shown in the tower during Ch4 (diagonal fill, raster scan, then hover / pin / the demo pair) */
  function ch4Pair(t) {
    if (t < T_DIAG[0]) return null;
    const act = mapInteractive(t) && (hover && hover.kind === "map" ? hover : pinned);
    if (act) return { s: act.s, g: act.g, a: 1, arc: true };
    if (t < T_DIAG[1]) { const i = Math.min(N - 1, Math.floor(N * seg(t, T_DIAG[0], T_DIAG[1]))); return { s: i, g: i, a: 1 }; }
    if (t < T_SCAN[0]) return null;
    if (t < T_SCAN[1]) {
      const kf = N * N * seg(t, T_SCAN[0], T_SCAN[1]), k = Math.min(N * N - 1, Math.floor(kf)), s = Math.floor(k / N);
      return { s, g: k % N, gf: Math.min(N - 0.5, kf - s * N), a: 1, scan: true };
    }
    const L = D.demo.late;
    return { s: L.src, g: L.tgt, a: sm(seg(t, T_SCAN[1], T_SCAN[1] + 0.012)), arc: true };
  }
  const mapInteractive = (t) => t >= 0.94 || (!(clk && clk.running) && t >= T_DIAG[0]);
  function revealedK(t) { return t >= T_SCAN[1] ? N * N : Math.floor(N * N * seg(t, T_SCAN[0], T_SCAN[1])); }
  function cellRevealed(t, s, g) { if (s === g) return t >= T_DIAG[0] + (T_DIAG[1] - T_DIAG[0]) * ((s + 1) / N); return s * N + g < revealedK(t); }

  /* ================================================================== drawing: tower */
  /* The tower is on screen from t = 0, so the first frame is a real poster (no fade-in from an empty stage). */
  const T_INTRO = [0.004, 0.031];                   // scene 0: one forward pass rises through the stack

  function drawTower(ctx, t, A) {
    if (A <= 0.002) return;
    const g = geo, n = N;
    const dim = sm(seg(t, 0.36, 0.372));
    const st = towerState(t);
    const fade = st.fade;
    const hatchLines = [];
    for (const col of g.cols) {
      const isHE = col.key !== "pre";
      const ca = col.key === "head" ? 1 : lerp(1, C.dark ? 0.5 : 0.3, dim);
      for (let l = 0; l < n; l++) {
        let a = A * ca;
        if (a <= 0.002) continue;
        const y = g.rowY(l);
        if (col.key === "head" && st.ex && l === st.ex.tgt) a *= lerp(1, st.old, fade);
        ctx.globalAlpha = a;
        ctx.fillStyle = C.grid;
        rr(ctx, col.x, y, col.w, g.cellH, 2); ctx.fill();
        // computed hidden states
        let tint = 0;
        if (st.tint != null && l <= st.tint) tint = 1;
        if (st.tint1 != null && l <= st.tint1) tint = Math.max(tint, st.tint1A);
        if (!isHE && st.tintPre != null && l <= st.tintPre) tint = 1;
        const hatched = isHE && st.hatch && l >= st.hatch.from && l <= st.hatch.to;
        if (hatched) tint = 0;
        tint *= (st.tintA != null ? st.tintA : 1) * fade;
        if (tint > 0.002) { ctx.globalAlpha = a * tint * 0.2; ctx.fillStyle = C.ink2; rr(ctx, col.x, y, col.w, g.cellH, 2); ctx.fill(); }
        if (hatched) hatchLines.push([col, y, a * fade]);
        // moving band
        const b = st.band;
        if (b && (!b.he || isHE)) {
          const d = b.dir > 0 ? b.front - l : l - b.front;
          if (d >= -0.5 && d < 4) {
            const k = (1 - Math.max(0, d) / 4) * (d < 0 ? 1 + d * 2 : 1);
            if (b.dir > 0) { ctx.globalAlpha = a * 0.34 * k; ctx.fillStyle = C.ink2; rr(ctx, col.x, y, col.w, g.cellH, 2); ctx.fill(); }
            else { ctx.globalAlpha = a * 0.75 * k; ctx.strokeStyle = C.ink2; ctx.lineWidth = 1; rr(ctx, col.x + 0.5, y + 0.5, col.w - 1, g.cellH - 1, 2); ctx.stroke(); }
          }
        }
      }
    }
    // hatch: 45 degree lines over recomputed states
    if (hatchLines.length) {
      ctx.strokeStyle = C.ink2; ctx.lineWidth = 1;
      hatchLines.forEach(([col, y, a]) => {
        ctx.save(); ctx.globalAlpha = a * 0.75;
        rr(ctx, col.x, y, col.w, g.cellH, 2); ctx.clip();
        ctx.fillStyle = rgba(C.rgb.ink2, 0.10); ctx.fillRect(col.x, y, col.w, g.cellH);
        ctx.beginPath();
        for (let x = col.x - g.cellH; x < col.x + col.w + g.cellH; x += 4) { ctx.moveTo(x, y + g.cellH); ctx.lineTo(x + g.cellH, y); }
        ctx.stroke(); ctx.restore();
      });
    }
    ctx.globalAlpha = 1;
    // update ticks (one per layer, identical everywhere)
    if (st.ticks != null) {
      ctx.fillStyle = C.ink2;
      for (let l = 0; l < n; l++) {
        const k = clamp(1 - Math.abs(st.ticks - l) / 2.2, 0, 1);
        if (k <= 0) continue;
        ctx.globalAlpha = A * k; ctx.fillRect(g.tickX, g.rowY(l), 2, g.cellH);
      }
      ctx.globalAlpha = 1;
    }
    drawTowerFrame(ctx, t, A, dim);
    // Ch3 experiment overlays
    // connector from the end of the prompt (where the answer is read) to the answer chip while an experiment's result is
    // on screen, so the eye can follow patch -> recomputed states -> answer rank; drawn under the labels
    const tr = trailState(t);
    if (tr) {
      const cs = chipState(t);
      const [cx0, cy0] = chipXY(cs.slot, 0);
      drawTrail(ctx, g.lad.horiz ? [cx0, cy0 - g.chipH / 2 - 3] : [g.lad.x - 3, cy0], A * tr.a * cs.a);
    }
    if (st.ex) drawExpOverlay(ctx, st, A * fade);
    // Ch4 pair
    if (t >= T_DIAG[0]) {
      const p = ch4Pair(t);
      if (p) drawPair(ctx, p, A * p.a);
    }
    if (st.pulse) drawPulse(ctx, st.pulse, A);
  }

  function drawTowerFrame(ctx, t, A, dim) {
    const g = geo, n = N;
    const a0 = A;
    // axis ticks (layer 0 at the bottom)
    ctx.globalAlpha = a0;
    ctx.font = F(10.5); ctx.fillStyle = C.muted; ctx.textAlign = "right"; ctx.textBaseline = "middle";
    const ticks = [0, 8, 16, 24, n - 1].filter((v, i, arr) => arr.indexOf(v) === i && v < n);
    ticks.forEach((l) => ctx.fillText(String(l), g.axisW - 6, g.rowCy(l)));
    ctx.save(); ctx.translate(6, (g.rowCy(Math.min(8, n - 1)) + g.rowCy(Math.min(16, n - 1))) / 2); ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center"; ctx.fillText("layer", 0, 0); ctx.restore();
    // "schematic" pill
    ctx.font = F(10, 600);
    const sw = ctx.measureText("SCHEMATIC").width + 12;
    ctx.globalAlpha = a0 * 0.9;
    ctx.strokeStyle = C.axis; ctx.lineWidth = 1; rr(ctx, 0.5, 6.5, sw, 16, 8); ctx.stroke();
    ctx.fillStyle = C.muted; ctx.textAlign = "left"; ctx.fillText("SCHEMATIC", 6.5, 15);
    // column labels (appear with the question)
    const la = A * sm(seg(t, 0.23, 0.252));
    if (la > 0.002) {
      const ly = g.towerB + 13, lh = 13;
      const pre = g.cols.filter((c) => c.key === "pre");
      const px0 = pre[0].x, px1 = pre[pre.length - 1].x + pre[pre.length - 1].w, pcx = (px0 + px1) / 2;
      const pl = prefixLabel();
      ctx.globalAlpha = la * lerp(1, 0.55, dim);
      ctx.font = F(11); ctx.fillStyle = C.ink2; ctx.textAlign = "right"; ctx.textBaseline = "middle";
      ctx.fillText(pl[0], px1, ly); ctx.fillText(pl[1], px1, ly + lh);
      void pcx;
      ctx.globalAlpha = la;
      ctx.font = F(11.5, 700); ctx.fillStyle = C.entText; ctx.textAlign = "center";
      ctx.fillText(D.head, g.head.x + g.head.w / 2, ly);
      ctx.font = F(10.5); ctx.fillStyle = C.muted; ctx.fillText("head entity", g.head.x + g.head.w / 2, ly + lh);
      ctx.globalAlpha = la * lerp(1, 0.7, dim);
      ctx.font = F(11); ctx.fillStyle = C.ink2; ctx.textAlign = "left";
      ctx.fillText("end of prompt:", g.end.x, ly);
      ctx.fillText("answer read here", g.end.x, ly + lh);
      // brackets under the token groups
      ctx.strokeStyle = C.axis; ctx.lineWidth = 1; ctx.globalAlpha = la;
      ctx.beginPath(); ctx.moveTo(px0, g.towerB + 3.5); ctx.lineTo(px1, g.towerB + 3.5); ctx.stroke();
    }
    // SLC30A8 column outline (Ch3 onward)
    if (dim > 0.002) {
      ctx.globalAlpha = A * dim;
      ctx.strokeStyle = C.entity; ctx.lineWidth = 1.5;
      rr(ctx, g.head.x - 3.5, g.ty0 - 3.5, g.head.w + 7, n * g.pitch + 4, 6); ctx.stroke();
    }
    // "update" tag above the tick strip during the epoch loop
    const ua = A * sm(seg(t, T_LOOP[0], T_LOOP[0] + 0.01)) * (1 - sm(seg(t, T_LOOP[1] - 0.012, T_LOOP[1])));
    if (ua > 0.002) {
      ctx.globalAlpha = ua; ctx.font = F(10.5); ctx.fillStyle = C.muted; ctx.textAlign = "left"; ctx.textBaseline = "middle";
      ctx.fillText("↑ forward", g.gutX, g.rowCy(n - 1) + 1);
      ctx.fillText(g.wide ? "↓ update, every layer" : "↓ update", g.gutX, g.rowCy(n - 1) + 15);
    }
    ctx.globalAlpha = 1;
  }

  function prefixLabel() {
    const i = D.question.indexOf(D.head);
    const words = (i > 0 ? D.question.slice(0, i) : D.question).trim().split(/\s+/);
    if (words.length <= 4) return [words.slice(0, 2).join(" "), words.slice(2).join(" ")];
    return [words.slice(0, 2).join(" ") + " …", "… " + words.slice(-2).join(" ")];
  }

  /* text with a thin stage-coloured halo (keeps labels clean where a connector passes behind them) */
  function haloFill(ctx, str, x, y) {
    const fs = ctx.fillStyle;
    ctx.lineJoin = "round"; ctx.lineWidth = 3; ctx.strokeStyle = C.stage; ctx.strokeText(str, x, y);
    ctx.fillStyle = fs; ctx.fillText(str, x, y);
  }

  function cellRect(l) { const g = geo; return [g.head.x, g.rowY(l), g.head.w, g.cellH]; }

  function drawRing(ctx, l, a) {
    const [x, y, w, hh] = cellRect(l);
    ctx.globalAlpha = a; ctx.strokeStyle = C.ink; ctx.lineWidth = 2;
    rr(ctx, x - 3, y - 3, w + 6, hh + 6, 5); ctx.stroke();
  }
  function drawSocket(ctx, l, a, filled, side) {
    const [x, y, w, hh] = cellRect(l);
    ctx.globalAlpha = a;
    ctx.strokeStyle = C.ink; ctx.lineWidth = 1.25; ctx.setLineDash([3, 2]);
    rr(ctx, x - 2.5, y - 2.5, w + 5, hh + 5, 4); ctx.stroke(); ctx.setLineDash([]);
    // the notch: a filled wedge pointing into the cell, on the side the state arrives from
    const cy = y + hh / 2;
    ctx.fillStyle = C.ink; ctx.beginPath();
    if (side > 0) { const nx = x + w + 3; ctx.moveTo(nx + 7, cy - 5); ctx.lineTo(nx, cy); ctx.lineTo(nx + 7, cy + 5); }
    else { const nx = x - 3; ctx.moveTo(nx - 7, cy - 5); ctx.lineTo(nx, cy); ctx.lineTo(nx - 7, cy + 5); }
    ctx.closePath(); ctx.fill();
    if (filled > 0) drawPatched(ctx, l, a * filled);
  }
  function drawPatched(ctx, l, a) {
    const [x, y, w, hh] = cellRect(l);
    ctx.globalAlpha = a * 0.35; ctx.strokeStyle = C.entity; ctx.lineWidth = 3;
    rr(ctx, x - 1, y - 1, w + 2, hh + 2, 3); ctx.stroke();
    ctx.globalAlpha = a; ctx.fillStyle = C.ink;
    rr(ctx, x + w * 0.14, y, w * 0.72, hh, 3); ctx.fill();
  }
  const sideOf = (src, tgt) => (src > tgt ? -1 : 1);
  /* the copy's route: a curve hugging the SLC30A8 column (left when writing lower, right when writing higher) */
  function arcPts(src, tgt) {
    const g = geo, side = sideOf(src, tgt);
    const xe = side < 0 ? g.head.x - 6 : g.head.x + g.head.w + 6;
    const y0 = g.rowCy(src), y1 = g.rowCy(tgt);
    const B = side * Math.min(g.bulge, 6 + Math.abs(y1 - y0) * 0.11);
    return [[xe, y0], [xe + B, lerp(y0, y1, 0.08)], [xe + B, lerp(y0, y1, 0.92)], [xe, y1]];
  }
  function drawArc(ctx, src, tgt, p, a) {
    if (p <= 0 || a <= 0) return;
    const P = arcPts(src, tgt);
    ctx.globalAlpha = a * 0.8; ctx.strokeStyle = C.ink2; ctx.lineWidth = 1.25; ctx.setLineDash([4, 3]);
    ctx.beginPath();
    const steps = 40;
    for (let i = 0; i <= steps; i++) { const q = bez(P[0], P[1], P[2], P[3], (i / steps) * p); i ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1]); }
    ctx.stroke(); ctx.setLineDash([]);
  }
  /* the copied state travels inside its own column (same token, another layer), lifted above the grid */
  function drawCapsule(ctx, src, tgt, cap, a) {
    const g = geo;
    const x = g.head.x + g.head.w / 2, y = lerp(g.rowCy(src), g.rowCy(tgt), cap.p);
    const w = g.head.w * 0.72 * (1 + cap.lift), hh = (g.cellH + 2) * (1 + cap.lift);
    ctx.globalAlpha = a * cap.a * 0.3; ctx.fillStyle = C.entity;
    rr(ctx, x - w / 2 - 4, y - hh / 2 - 4, w + 8, hh + 8, 7); ctx.fill();
    ctx.globalAlpha = a * cap.a; ctx.fillStyle = C.ink;
    rr(ctx, x - w / 2, y - hh / 2, w, hh, 4); ctx.fill();
  }

  function drawExpOverlay(ctx, st, a) {
    const ex = st.ex, g = geo;
    if (a <= 0.002) return;
    const labels = [];
    drawArc(ctx, ex.src, ex.tgt, st.arcP, st.arcA * a);
    if (st.socket > 0) drawSocket(ctx, ex.tgt, st.socket * a, st.patched, sideOf(ex.src, ex.tgt));
    if (st.ring > 0) drawRing(ctx, ex.src, st.ring * a);
    if (st.cap) drawCapsule(ctx, ex.src, ex.tgt, st.cap, a);
    if (st.ring > 0) labels.push({ l: ex.src, a: st.ring * a, kind: "read" });
    if (st.socket > 0) {
      const r = Math.round(1 / Math.max(ex.v, 0.01));
      labels.push({ l: ex.tgt, a: st.socket * a, kind: "write", res: st.res || 0, resText: r === X.u ? `→ still ${ord(X.u)}` : `→ ${rankText(ex.v)}` });
    }
    // recomputed bracket + "unchanged" tag
    if (st.rcTag > 0) {
      const ra = st.rcTag * a, x = g.tickX + 1;
      const yTop = g.rowY(N - 1), yBot = g.rowY(ex.tgt + 1) + g.cellH;
      ctx.globalAlpha = ra; ctx.strokeStyle = C.ink2; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x - 3, yTop); ctx.lineTo(x, yTop); ctx.lineTo(x, yBot); ctx.lineTo(x - 3, yBot); ctx.stroke();
      labels.push({ l: (N - 1 + ex.tgt + 1) / 2, a: ra, kind: "rc" });
      const pre = g.cols.filter((c) => c.key === "pre");
      const px0 = pre[0].x, px1 = pre[pre.length - 1].x + pre[pre.length - 1].w;
      ctx.beginPath(); ctx.moveTo(px0, g.ty0 - 6.5); ctx.lineTo(px1, g.ty0 - 6.5); ctx.stroke();
      ctx.font = F(10.5); ctx.fillStyle = C.muted; ctx.textAlign = "left"; ctx.textBaseline = "middle";
      haloFill(ctx, g.wide ? "unchanged: earlier tokens can't see the patch" : "unchanged: earlier tokens can't see it", px0, g.ty0 - 15);
    }
    drawGutterLabels(ctx, labels);
  }

  function drawSweep(ctx, gf, a) {
    const g = geo, n = N, x = g.head.x, w = g.head.w;
    for (let l = 0; l < n; l++) {
      const d = gf - (l + 0.5);                      // the band leads upward with a fading tail below it
      const k = d >= -0.6 && d < 5 ? (d < 0 ? 1 + d / 0.6 : 1 - d / 5) : 0;
      if (k <= 0.01) continue;
      ctx.globalAlpha = a * 0.5 * k; ctx.fillStyle = C.ink2;
      rr(ctx, x, g.rowY(l), w, g.cellH, 2); ctx.fill();
    }
    // bracket: every target layer of the column is tried in turn
    const bx = g.tickX + 1, yTop = g.rowY(n - 1), yBot = g.rowY(0) + g.cellH;
    ctx.globalAlpha = a * 0.8; ctx.strokeStyle = C.ink2; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(bx - 3, yTop); ctx.lineTo(bx, yTop); ctx.lineTo(bx, yBot); ctx.lineTo(bx - 3, yBot); ctx.stroke();
    ctx.globalAlpha = 1;
  }

  function drawPair(ctx, p, a) {
    if (a <= 0.002) return;
    if (p.scan) {
      // raster scan: the read ring steps once per source row; the write position is a soft band sweeping the column
      // (no per-cell socket or layer number, which would change ~150 times a second)
      drawSweep(ctx, p.gf, a);
      drawRing(ctx, p.s, a);
      drawGutterLabels(ctx, [{ l: p.s, a, kind: "read" }, { l: (N - 1) / 2, a, kind: "sweep" }]);
      return;
    }
    if (p.arc && p.s !== p.g) drawArc(ctx, p.s, p.g, 1, a * 0.9);
    drawSocket(ctx, p.g, a, p.scan ? 0 : 1, sideOf(p.s, p.g));
    drawRing(ctx, p.s, a);
    if (p.s === p.g) drawGutterLabels(ctx, [{ l: p.s, a, kind: "same" }]);
    else drawGutterLabels(ctx, [{ l: p.s, a, kind: "read" }, { l: p.g, a, kind: "write" }]);
  }

  /* labels in the gutter right of the tower, with leader lines; pushed apart when they would overlap */
  function drawGutterLabels(ctx, items) {
    if (!items.length) return;
    const g = geo, lh = 12.5;
    items.forEach((it) => {
      const L = it.kind === "rc" ? String(Math.round(it.l)) : it.l;
      it.lines = it.kind === "read" ? (g.wide ? [["read", 700], [`· layer ${it.l}`, 400]] : [["read", 700], [`layer ${it.l}`, 400]])
        : it.kind === "write" ? (g.wide ? [["write", 700], [`· layer ${it.l}`, 400]] : [["write", 700], [`layer ${it.l}`, 400]])
          : it.kind === "same" ? [["read = write", 700], [`layer ${it.l}`, 400]]
            : it.kind === "sweep" ? [["write", 700], [g.wide ? "· every layer" : "every layer", 400]]
              : [["recomputed", 400]];
      it.inline = g.wide && (it.kind === "read" || it.kind === "write" || it.kind === "sweep");
      if (it.res > 0.002) it.lines.push([it.resText, 700, it.res]);
      it.h = it.inline ? lh * (it.lines.length > 2 ? 2 : 1) : it.lines.length * lh;
      it.y0 = it.kind === "rc" || it.kind === "sweep" ? g.rowCy(0) + (g.rowCy(N - 1) - g.rowCy(0)) * ((it.l) / (N - 1)) : g.rowCy(it.l);
      it.y = it.inline ? it.y0 + (it.h - lh) / 2 : it.y0;   // inline: the first line sits on the row
      void L;
    });
    items.sort((p, q) => p.y - q.y);
    const top = g.ty0 + 4, bot = g.towerB - 4, gapY = 4;
    for (let i = 1; i < items.length; i++) {
      const need = items[i - 1].y + (items[i - 1].h + items[i].h) / 2 + gapY;
      if (items[i].y < need) items[i].y = need;
    }
    const over = items[items.length - 1].y + items[items.length - 1].h / 2 - bot;
    if (over > 0) items.forEach((it) => { it.y -= over; });
    if (items[0].y - items[0].h / 2 < top) { const d = top - (items[0].y - items[0].h / 2); items.forEach((it) => { it.y += d; }); }
    items.forEach((it) => {
      ctx.globalAlpha = it.a;
      const lx = g.gutX + 2;
      if (it.kind !== "rc" && it.kind !== "sweep") {
        ctx.strokeStyle = C.ink2; ctx.lineWidth = 1;
        const ly = it.inline ? it.y - it.h / 2 + lh / 2 : it.y;
        ctx.beginPath(); ctx.moveTo(g.head.x + g.head.w + 4, it.y0); ctx.lineTo(g.towerR + 2, it.y0); ctx.lineTo(lx - 3, ly); ctx.stroke();
      }
      ctx.textBaseline = "middle"; ctx.textAlign = "left";
      if (it.inline) {
        // "write · layer 10" on one line; the result ("→ 1st") on a second line below it
        const y1 = it.y - it.h / 2 + lh / 2;
        ctx.font = F(11.5, 700); ctx.fillStyle = C.ink; haloFill(ctx, it.lines[0][0], lx, y1);
        const w0 = ctx.measureText(it.lines[0][0] + " ").width;
        ctx.font = F(11.5); ctx.fillStyle = C.ink2; haloFill(ctx, it.lines[1][0], lx + w0, y1);
        if (it.lines[2]) {
          ctx.globalAlpha = it.a * it.lines[2][2];
          ctx.font = F(12.5, 700); ctx.fillStyle = C.ink; haloFill(ctx, it.lines[2][0], lx, y1 + lh + 1);
        }
      } else {
        it.lines.forEach((ln, k) => {
          ctx.globalAlpha = it.a * (ln[2] != null ? ln[2] : 1);
          ctx.font = F(11.5, ln[1]); ctx.fillStyle = ln[1] >= 700 ? C.ink : C.ink2;
          if (it.kind === "rc") { ctx.font = F(11); ctx.fillStyle = C.muted; }
          haloFill(ctx, ln[0], lx, it.y - it.h / 2 + lh * (k + 0.5));
        });
      }
    });
    ctx.globalAlpha = 1;
  }

  /* ================================================================== drawing: ladder, chips, tokens */
  function slotC(r) {
    const g = geo, L = g.lad;
    if (L.horiz) { const i = 6 - clamp(r, 1, 6); return [(i + 0.5) * L.pitch, L.y + L.h / 2]; }
    return [L.x + L.lab + (L.w - L.lab - 4) / 2, L.y0 + (r - 1) * L.pitch + L.pitch / 2];
  }

  function drawLadder(ctx, t) {
    const a = sm(seg(t, 0.226, 0.25)) * (1 - sm(seg(t, 0.665, 0.675)));
    if (a <= 0.002) return;
    const g = geo, L = g.lad, slide = sm(seg(t, 0.665, 0.675)) * 14;
    ctx.save(); ctx.globalAlpha = a; ctx.translate(g.wide ? slide : 0, 0);
    ctx.textBaseline = "middle";
    const title = "rank of the answer's first token";
    ctx.font = F(11, 600); ctx.fillStyle = C.ink2; ctx.textAlign = "left";
    if (L.horiz) {
      const bs = bigState(t), ta = bs ? 1 - bs.a : 1;
      if (ta > 0.002) { ctx.globalAlpha = a * ta; ctx.fillText(title, 0, L.titleY); ctx.globalAlpha = a; }
    } else ctx.fillText(title, g.rx0, g.ty0 - 16);
    for (let r = 1; r <= 6; r++) {
      const [cx, cy] = slotC(r);
      const lab = r <= 5 ? ord(r) : "lower";
      ctx.strokeStyle = C.grid; ctx.lineWidth = 1; ctx.fillStyle = rgba(C.rgb.surface2, 0.6);
      if (L.horiz) {
        rr(ctx, cx - L.w / 2 + 0.5, L.y + 0.5, L.w - 1, L.h - 1, 7); ctx.fill(); ctx.stroke();
        ctx.font = F(10.5, r === 1 ? 600 : 400); ctx.fillStyle = r === 1 ? C.ink2 : C.muted; ctx.textAlign = "center";
        ctx.fillText(lab, cx, L.y + L.h + 10);
      } else {
        rr(ctx, L.x + 0.5, cy - L.h / 2 + 0.5, L.w - 1, L.h - 1, 8); ctx.fill(); ctx.stroke();
        ctx.font = F(11, r === 1 ? 600 : 400); ctx.fillStyle = r === 1 ? C.ink2 : C.muted; ctx.textAlign = "left";
        ctx.fillText(lab, L.x + 11, cy);
      }
    }
    ctx.restore();
  }

  /* where the answer chip and the "another token" chip sit (slot 1 = top rank) */
  function chipState(t) {
    const u = X.u;
    const st = { on: t >= T_DROP[0] - 0.001, slot: u, v: 1 / u, dx: 0, a: 1, grey: u > 1 ? { slot: 1, a: 1, dx: 0 } : null };
    if (!st.on) return st;
    const pd = seg(t, T_DROP[0], T_DROP[1]);
    st.slot = lerp(u - 1.6, u, eBack(pd));
    st.a = clamp(pd * 4, 0, 1);
    if (st.grey) st.grey.a = sm(seg(t, T_DROP[0] + 0.004, T_DROP[1]));
    for (let i = 0; i < EXPS.length; i++) {
      const ex = EXPS[i];
      if (t < ex.t0) break;
      const r = Math.round(1 / Math.max(ex.v, 0.01));
      // each later experiment starts from the unpatched state: undo the previous one (and keep it undone afterwards,
      // so a control that changes nothing leaves the chip in the unpatched slot)
      if (ex.reset) {
        const p = eIO(seg(t, ex.reset[0], ex.reset[1]));
        const pr = EXPS[i - 1], rPrev = Math.round(1 / Math.max(pr.v, 0.01));
        st.slot = lerp(rPrev, u, p); st.v = lerp(pr.v, 1 / u, p); st.dx = 0;
        if (st.grey) { st.grey.slot = lerp(rPrev === 1 ? u : 1, 1, p); st.grey.dx = 0; }
      }
      if (t < ex.sw[0]) continue;
      const p = eIO(seg(t, ex.sw[0], ex.sw[1]));
      if (r !== u) {
        st.slot = lerp(u, Math.min(r, 6), p); st.v = lerp(1 / u, ex.v, p);
        st.dx = bump(p) * (geo && geo.wide ? 22 : 10);
        if (st.grey && r === 1) { st.grey.slot = lerp(1, u, p); st.grey.dx = -bump(p) * (geo && geo.wide ? 22 : 10); }
      } else if (!M.reducedMotion) {
        const q = seg(t, ex.sw[0], ex.sw[1]);
        st.dx = 2.2 * Math.sin(q * Math.PI * 8) * (1 - q);
      }
    }
    const out = 1 - sm(seg(t, 0.665, 0.675));
    st.a *= out; if (st.grey) st.grey.a *= out;
    return st;
  }

  /* centre of a chip at a (fractional) ladder slot */
  function chipXY(slot, dx) {
    const g = geo, horiz = !!g.lad.horiz;
    const [cx, cy] = slotC(clamp(slot, 1, 6));
    if (slot < 1) return horiz ? [cx, cy - (1 - slot) * 30] : [cx + dx, cy - (1 - slot) * g.lad.pitch];
    if (horiz) {
      const i0 = Math.floor(slot), f = slot - i0;
      const a0 = slotC(i0), a1 = slotC(Math.min(6, i0 + 1));
      return [lerp(a0[0], a1[0], f), a0[1] - dx];
    }
    return [cx + dx, g.lad.y0 + (slot - 1) * g.lad.pitch + g.lad.pitch / 2];
  }

  function drawChip(ctx, cx, cy, w, hh, fillRGB, text, a, dashed, textCol, sc) {
    if (sc && sc !== 1) { w *= sc; hh *= sc; }
    ctx.globalAlpha = a;
    ctx.fillStyle = `rgb(${fillRGB})`;
    rr(ctx, cx - w / 2, cy - hh / 2, w, hh, hh / 2); ctx.fill();
    ctx.strokeStyle = dashed ? C.axis : rgba(C.rgb.ink2, 0.75); ctx.lineWidth = 1;
    if (dashed) ctx.setLineDash([3, 2]);
    rr(ctx, cx - w / 2 + 0.5, cy - hh / 2 + 0.5, w - 1, hh - 1, hh / 2); ctx.stroke(); ctx.setLineDash([]);
    ctx.font = F((geo.wide ? 12 : 11) * (sc || 1), 600); ctx.fillStyle = textCol || chipInk(fillRGB); ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(text, cx, cy + 0.5);
  }

  function drawChips(ctx, t) {
    const cs = chipState(t);
    if (!cs.on || cs.a <= 0.002) return;
    const g = geo, horiz = !!g.lad.horiz;
    const pos = chipXY;
    // flare behind the chip at the aha moment: a soft glow and an expanding ring in the rank-1 colour
    let chipScale = 1;
    for (const ex of EXPS) {
      if (!ex.flare) continue;
      chipScale *= 1 + 0.12 * bump(seg(t, ex.sw[0], ex.sw[1] + 0.004));
      if (M.reducedMotion) continue;
      const p = seg(t, ex.flare[0], ex.flare[1]);
      if (p <= 0 || p >= 1) continue;
      const [fx, fy] = pos(1, 0);
      const R = lerp(16, g.wide ? 104 : 72, eExpo(p));
      const hr = M.heatRGB(1), a0 = C.dark ? 0.55 : 0.45;
      const grd = ctx.createRadialGradient(fx, fy, 0, fx, fy, R);
      grd.addColorStop(0, rgba(hr, a0 * (1 - p))); grd.addColorStop(0.55, rgba(hr, a0 * 0.45 * (1 - p))); grd.addColorStop(1, rgba(hr, 0));
      ctx.globalAlpha = 1; ctx.fillStyle = grd; ctx.beginPath(); ctx.arc(fx, fy, R, 0, Math.PI * 2); ctx.fill();
      // expanding ring around the chip
      const q = seg(t, ex.flare[0], ex.flare[0] + (ex.flare[1] - ex.flare[0]) * 0.8);
      if (q > 0 && q < 1) {
        const grow = eExpo(q) * (g.wide ? 26 : 18);
        ctx.globalAlpha = 0.9 * (1 - q); ctx.strokeStyle = `rgb(${hr})`; ctx.lineWidth = 2.5;
        rr(ctx, fx - g.chipW / 2 - 3 - grow, fy - g.chipH / 2 - 3 - grow, g.chipW + 6 + 2 * grow, g.chipH + 6 + 2 * grow, g.chipH / 2 + 3 + grow);
        ctx.stroke();
      }
    }
    if (cs.grey && cs.grey.a > 0.002) {
      const [x, y] = pos(cs.grey.slot, cs.grey.dx);
      drawChip(ctx, x, y, g.chipW, g.chipH, C.rgb.surface2, g.wide ? "another token" : "other", cs.a * cs.grey.a, true, C.muted);
    }
    const [x, y] = pos(cs.slot, cs.dx);
    drawChip(ctx, x, y, g.chipW, g.chipH, M.heatRGB(cs.v), "answer", cs.a, false, null, M.reducedMotion ? 1 : chipScale);
    geo.chipBox = [x - g.chipW / 2, y - g.chipH / 2, g.chipW, g.chipH];
    ctx.globalAlpha = 1;
  }

  /* big "2nd → 1st" readout next to the ladder (wide) / centred above it (narrow) */
  function bigState(t) {
    for (let i = 0; i < EXPS.length; i++) {
      const ex = EXPS[i], next = EXPS[i + 1];
      const end = next ? next.t0 + 0.004 : CH_T[3] + 0.012;
      if (t < ex.sw[1] - 0.004 || t >= end) continue;
      const r = Math.round(1 / Math.max(ex.v, 0.01));
      const a = sm(seg(t, ex.sw[1] - 0.004, ex.sw[1] + 0.002)) * (1 - sm(seg(t, end - 0.006, end)));
      return r === X.u ? { a, text: `still ${ord(X.u)}`, sub: `no change: read ${ex.src} → write ${ex.tgt}` }
        : { a, text: `${ord(X.u)} → ${rankText(ex.v)}`, sub: `read ${ex.src} → write ${ex.tgt}` };
    }
    return null;
  }
  function drawBigLabel(ctx, t) {
    const g = geo, b = bigState(t);
    if (!b || b.a <= 0.002) return;
    ctx.globalAlpha = b.a; ctx.textBaseline = "middle";
    if (g.wide) {
      ctx.font = F(34, 700); ctx.fillStyle = C.ink; ctx.textAlign = "left";
      ctx.fillText(b.text, g.big.x, g.big.y);
      ctx.font = F(12); ctx.fillStyle = C.ink2;
      ctx.fillText(b.sub, g.big.x + 1, g.big.y + 29);
    } else {
      ctx.font = F(18, 700); ctx.fillStyle = C.ink; ctx.textAlign = "center";
      ctx.fillText(b.text, g.W / 2, g.lad.titleY - 1);
    }
    ctx.globalAlpha = 1;
  }

  function tokenRest(i) {
    const g = geo;
    if (g.wide) return [g.tok.x + g.tok.w / 2, g.tok.y + i * g.tok.dy];
    return [8 + g.tok.w * (i + 0.5), g.tokY];
  }
  function mapCellC(s, gg) { const m = geo.map; return [m.x + (gg + 0.5) * m.c, m.y + (s + 0.5) * m.c]; }

  function drawTokens(ctx, t) {
    const g = geo;
    EXPS.forEach((ex, i) => {
      const ap = sm(seg(t, ex.sw[1], ex.sw[1] + 0.006));
      if (ap <= 0.002) return;
      const f0 = T_FLY[0] + i * 0.005, f1 = T_FLY[1] - (EXPS.length - 1 - i) * 0.005;
      const fly = eIO(seg(t, f0, f1));
      if (fly >= 1) return;
      const [rx, ry] = tokenRest(i);
      const [mx, my] = mapCellC(ex.src, ex.tgt);
      const x = lerp(rx, mx, fly), y = lerp(ry, my, fly);
      const w0 = g.wide ? g.tok.w : g.tok.w - 6, h0 = g.wide ? 26 : 24;
      const w = lerp(w0, g.map.c + 4, fly), hh = lerp(h0, g.map.c + 4, fly);
      const r = Math.round(1 / Math.max(ex.v, 0.01));
      ctx.globalAlpha = ap;
      ctx.fillStyle = C.stage; rr(ctx, x - w / 2, y - hh / 2, w, hh, Math.min(hh / 2, 8)); ctx.fill();
      ctx.strokeStyle = C.ink; ctx.lineWidth = 1.5; if (ex.ctrl) ctx.setLineDash([3, 2.5]);
      rr(ctx, x - w / 2, y - hh / 2, w, hh, Math.min(hh / 2, 8)); ctx.stroke(); ctx.setLineDash([]);
      const ta = ap * (1 - clamp(fly * 3, 0, 1));
      if (ta > 0.01) {
        ctx.globalAlpha = ta;
        const sw = 10, sx = x - w / 2 + 9;
        ctx.fillStyle = M.heat(ex.v); rr(ctx, sx, y - sw / 2, sw, sw, 2); ctx.fill();
        ctx.strokeStyle = rgba(C.rgb.ink2, 0.5); ctx.lineWidth = 1; rr(ctx, sx + 0.5, y - sw / 2 + 0.5, sw - 1, sw - 1, 2); ctx.stroke();
        ctx.textBaseline = "middle"; ctx.textAlign = "left";
        ctx.font = F(g.wide ? 12 : 11, 650); ctx.fillStyle = C.ink;
        const lab = `${ex.src}→${ex.tgt}`;
        ctx.fillText(lab, sx + sw + 6, y + 0.5);
        const lw = ctx.measureText(lab).width;
        ctx.font = F(g.wide ? 12 : 11); ctx.fillStyle = C.ink2;
        ctx.fillText(`· ${r <= 5 ? ord(r) : "lower"}`, sx + sw + 6 + lw + 4, y + 0.5);
      }
    });
    ctx.globalAlpha = 1;
  }

  function drawPulse(ctx, pu, A) {
    const g = geo;
    const p0 = [g.end.x + g.end.w / 2, g.rowCy(N - 1)];
    const target = slotC(X.u);
    const p2 = g.lad.horiz ? [target[0], target[1] - g.lad.h / 2 - 2] : [g.lad.x - 3, target[1]];
    const p1 = g.lad.horiz ? [g.W - 8, p0[1]] : [lerp(p0[0], p2[0], 0.6), p0[1]];
    const at = (u) => [(1 - u) * (1 - u) * p0[0] + 2 * (1 - u) * u * p1[0] + u * u * p2[0], (1 - u) * (1 - u) * p0[1] + 2 * (1 - u) * u * p1[1] + u * u * p2[1]];
    const e = eIO(pu.p);
    ctx.globalAlpha = A * pu.a * 0.5; ctx.strokeStyle = C.ink2; ctx.lineWidth = 1.25;
    ctx.beginPath();
    for (let i = 0; i <= 30; i++) { const q = at((i / 30) * e); i ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1]); }
    ctx.stroke();
    const q = at(e);
    ctx.globalAlpha = A * pu.a; ctx.fillStyle = C.ink;
    ctx.beginPath(); ctx.arc(q[0], q[1], 3.5, 0, Math.PI * 2); ctx.fill();
    ctx.globalAlpha = 1;
  }

  /* the read-out connector of the current experiment: on once its pulse has arrived, off when the next one resets */
  function trailState(t) {
    for (let i = 0; i < EXPS.length; i++) {
      const ex = EXPS[i], next = EXPS[i + 1];
      const end = next ? next.reset[0] : CH_T[3];
      if (t < ex.pl[1] - 0.002 || t >= end + 0.006) continue;
      return { ex, a: sm(seg(t, ex.pl[1] - 0.002, ex.pl[1] + 0.003)) * (1 - sm(seg(t, end, end + 0.006))) };
    }
    return null;
  }
  function drawTrail(ctx, p2, a) {
    if (a <= 0.002) return;
    const g = geo;
    const p0 = [g.end.x + g.end.w / 2, g.rowY(N - 1) - 2];            // leaves the top of the column, above the gutter labels
    const p1 = g.lad.horiz ? [g.W - 8, p0[1]] : [lerp(p0[0], p2[0], 0.6), p0[1] - 6];
    ctx.save();
    ctx.globalAlpha = a * 0.7; ctx.strokeStyle = C.ink2; ctx.lineWidth = 1.25; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(p0[0], p0[1]);
    if (g.lad.horiz) ctx.bezierCurveTo(g.W - 6, p0[1], g.W - 6, p2[1] - 8, p2[0], p2[1]);
    else ctx.quadraticCurveTo(p1[0], p1[1], p2[0], p2[1]);
    ctx.stroke(); ctx.setLineDash([]);
    ctx.globalAlpha = a; ctx.fillStyle = C.ink;
    ctx.beginPath(); ctx.arc(p0[0], p0[1], 2.5, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.arc(p2[0], p2[1], 2.5, 0, Math.PI * 2); ctx.fill();
    ctx.restore();
  }

  /* ================================================================== drawing: training curves */
  function haloText(ctx, str, x, y, col, fnt, align) {
    ctx.font = fnt; ctx.textAlign = align || "left"; ctx.textBaseline = "middle";
    ctx.lineJoin = "round"; ctx.lineWidth = 4; ctx.strokeStyle = C.stage; ctx.strokeText(str, x, y);
    ctx.fillStyle = col; ctx.fillText(str, x, y);
  }
  /* mem / two-hop step curves of the training log, drawn up to epoch eTo (future part faint when o.future > 0) */
  function drawCurves(ctx, b, eTo, o) {
    const nE = D.mem.length - 1;
    const xs = (ep) => b.x + (ep / nE) * b.w, ys = (v) => b.y + b.h - v * b.h;
    const A0 = ctx.globalAlpha;
    // band: memorized, not yet used (to the current epoch; to t_gen once the future is revealed)
    if (eTo >= D.t_mem) {
      const e1 = Math.min(eTo, D.t_gen != null ? D.t_gen : nE);
      ctx.fillStyle = C.band; ctx.fillRect(xs(D.t_mem), b.y - 3, xs(e1) - xs(D.t_mem), b.h + 6);
      if (o.future && D.t_gen != null && D.t_gen > eTo) {
        ctx.globalAlpha = A0 * 0.45 * o.future; ctx.fillRect(xs(eTo), b.y - 3, xs(D.t_gen) - xs(eTo), b.h + 6); ctx.globalAlpha = A0;
      }
      if (o.bandLabel) {
        ctx.font = F(10.5); ctx.fillStyle = C.muted; ctx.textAlign = "left"; ctx.textBaseline = "bottom";
        ctx.fillText("memorized", xs(D.t_mem) + 1, b.y - 6);
      }
    }
    ctx.strokeStyle = C.grid; ctx.lineWidth = 1;
    [0, 0.5, 1].forEach((v) => { ctx.beginPath(); ctx.moveTo(b.x, Math.round(ys(v)) + 0.5); ctx.lineTo(b.x + b.w, Math.round(ys(v)) + 0.5); ctx.stroke(); });
    ctx.font = F(o.small ? 10 : 10.5); ctx.fillStyle = C.muted; ctx.textBaseline = "middle"; ctx.textAlign = "right";
    [0, 1].forEach((v) => ctx.fillText(String(v), b.x - 5, ys(v)));
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    (o.small ? [0, 10, 20, 30] : [0, 5, 10, 15, 20, 25, 30]).filter((ep) => ep <= nE).forEach((ep) => ctx.fillText(String(ep), xs(ep), b.y + b.h + 5));
    const step = (arr, from, to, col, lw, alpha) => {
      if (to <= from && from !== 0) return;
      ctx.globalAlpha = alpha; ctx.strokeStyle = col; ctx.lineWidth = lw; ctx.lineJoin = "round"; ctx.lineCap = "butt";
      ctx.beginPath(); ctx.moveTo(xs(from), ys(arr[from]));
      for (let i = from + 1; i <= to; i++) { ctx.lineTo(xs(i), ys(arr[i - 1])); ctx.lineTo(xs(i), ys(arr[i])); }
      ctx.stroke(); ctx.globalAlpha = A0;
    };
    if (o.future) { step(D.mem, eTo, nE, C.mem, 1.5, A0 * 0.4 * o.future); step(D.gen, eTo, nE, C.gen, 2, A0 * 0.4 * o.future); }
    step(D.mem, 0, eTo, C.mem, 2, A0);
    step(D.gen, 0, eTo, C.gen, 2.5, A0);
    // current epoch marker
    ctx.strokeStyle = C.axis; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(Math.round(xs(eTo)) + 0.5, b.y - 3); ctx.lineTo(Math.round(xs(eTo)) + 0.5, b.y + b.h + 3); ctx.stroke();
    [[D.mem, C.mem], [D.gen, C.gen]].forEach(([arr, col]) => { ctx.fillStyle = col; ctx.beginPath(); ctx.arc(xs(eTo), ys(arr[eTo]), o.small ? 2.6 : 3.4, 0, Math.PI * 2); ctx.fill(); });
    if (o.future && D.t_gen != null && D.t_gen > eTo) {
      ctx.globalAlpha = A0 * o.future;
      ctx.fillStyle = C.gen; ctx.beginPath(); ctx.arc(xs(D.t_gen), ys(D.gen[D.t_gen]), 2.6, 0, Math.PI * 2); ctx.fill();
      ctx.globalAlpha = A0;
    }
    // direct labels, right of the marker, haloed
    if (o.labels) {
      const fnt = F(o.small ? 10.5 : 11.5, 600), lx = xs(eTo) + 8;
      let ym = ys(D.mem[eTo]) - (D.mem[eTo] >= 1 ? 0 : 0), yg = ys(D.gen[eTo]);
      if (Math.abs(ym - yg) < 14) ym = yg - 14;
      haloText(ctx, o.small ? "facts" : "facts recalled", lx, ym, C.mem, fnt);
      haloText(ctx, o.small ? "two-hop" : "two-hop answer", lx, yg, C.gen, fnt);
    }
    return { xs, ys };
  }

  function drawTrain(ctx, t) {
    const a = 1 - sm(seg(t, 0.222, 0.238));
    if (a <= 0.002) return;
    const g = geo, b = g.train, e = epochAt(t);
    ctx.save(); ctx.globalAlpha = a;
    const rec = Math.round(2 * D.mem[e]);
    ctx.textBaseline = "middle";
    if (g.wide) {
      ctx.font = F(11, 600); ctx.fillStyle = C.ink2; ctx.textAlign = "left";
      ctx.fillText("training log", g.rx0, g.ty0 - 16);
      ctx.font = F(11); ctx.fillStyle = C.ink2; ctx.textAlign = "right";
      ctx.fillText("facts recalled", g.rx0 + g.rw, g.ty0 - 16);
      ctx.font = F(26, 650); ctx.fillStyle = C.ink;
      ctx.fillText(`${rec} / 2`, g.rx0 + g.rw, g.ty0 + 12);
      ctx.save(); ctx.translate(g.rx0 + 6, b.y + b.h / 2); ctx.rotate(-Math.PI / 2);
      ctx.font = F(10.5); ctx.fillStyle = C.muted; ctx.textAlign = "center"; ctx.fillText("accuracy", 0, 0); ctx.restore();
      ctx.font = F(10.5); ctx.fillStyle = C.muted; ctx.textAlign = "center";
      ctx.fillText("fine-tuning epoch", b.x + b.w / 2, b.y + b.h + 30);
    } else {
      ctx.font = F(11, 600); ctx.fillStyle = C.ink2; ctx.textAlign = "left";
      ctx.fillText("training log", 0, g.lad.titleY);
      ctx.font = F(11); ctx.textAlign = "right";
      ctx.fillText(`epoch ${e} · facts recalled ${rec}/2`, g.W, g.lad.titleY);
    }
    drawCurves(ctx, b, e, { labels: true, bandLabel: g.wide, small: !g.wide });
    ctx.restore();
  }

  const SPARK = { h: 96, x: 18, y: 14, r: 8, b: 24 };
  function drawSpark(t) {
    if (!geo || !geo.wide || !sparkCv) return;
    const w = sparkCv.parentNode.clientWidth;
    if (!w) return;
    const ctx = M.fitCanvas(sparkCv, w, SPARK.h);
    ctx.clearRect(0, 0, w, SPARK.h);
    const fut = sm(seg(t, 0.945, 0.97));
    const b = { x: SPARK.x, y: SPARK.y, w: w - SPARK.x - SPARK.r, h: SPARK.h - SPARK.y - SPARK.b };
    const { xs, ys } = drawCurves(ctx, b, E, { labels: false, small: true, future: fut });
    const fnt = F(10.5, 600), lx = xs(E) + 7;
    haloText(ctx, "facts recalled", lx, ys(D.mem[E]) - 7, C.mem, fnt);
    haloText(ctx, "two-hop answer", lx, ys(D.gen[E]) - 7, C.gen, fnt);
    if (hover && hover.kind === "spark") {
      const x = Math.round(xs(hover.e)) + 0.5;
      ctx.strokeStyle = C.ink2; ctx.lineWidth = 1; ctx.setLineDash([2, 2]);
      ctx.beginPath(); ctx.moveTo(x, b.y - 3); ctx.lineTo(x, b.y + b.h + 3); ctx.stroke(); ctx.setLineDash([]);
    }
  }

  /* ================================================================== drawing: map */
  function drawMap(ctx, t) {
    const aF = sm(seg(t, T_MAPIN[0], T_MAPIN[1]));
    if (aF <= 0.002) return;
    const g = geo, m = g.map, n = N, c = m.c;
    ensureCaches();
    ctx.save(); ctx.globalAlpha = aF;
    ctx.drawImage(phCache, m.x, m.y, m.S, m.S);
    const dpr = mapCache.width / m.S;
    const k = revealedK(t), rows = Math.floor(k / n), rem = k % n;
    const blit = (gx, gy, gw, gh) => ctx.drawImage(mapCache, gx * c * dpr, gy * c * dpr, gw * c * dpr, gh * c * dpr, m.x + gx * c, m.y + gy * c, gw * c, gh * c);
    if (rows > 0) blit(0, 0, n, rows);
    if (rem > 0) blit(0, rows, rem, 1);
    // diagonal = the unpatched model
    const nd = t >= T_DIAG[1] ? n : Math.floor(n * seg(t, T_DIAG[0], T_DIAG[1]));
    const dp = bump(seg(t, T_DPULSE[0], T_DPULSE[1]));
    for (let i = 0; i < nd; i++) {
      const x = m.x + i * c, y = m.y + i * c, v = X.G[i * n + i];
      ctx.fillStyle = M.heat(v); ctx.fillRect(x + 0.5, y + 0.5, c - 1, c - 1);
      ctx.strokeStyle = C.diag; ctx.lineWidth = 1.1 + dp * 1.2;
      ctx.strokeRect(x + 1, y + 1, c - 2, c - 2);
    }
    // frame
    ctx.strokeStyle = C.axis; ctx.lineWidth = 1; ctx.strokeRect(m.x - 0.5, m.y - 0.5, m.S + 1, m.S + 1);
    // axes
    ctx.font = F(10.5); ctx.fillStyle = C.muted; ctx.textBaseline = "middle";
    const ticks = [0, 8, 16, 24, n - 1];
    ctx.textAlign = "right"; ticks.forEach((s) => ctx.fillText(String(s), m.x - 5, m.y + (s + 0.5) * c));
    ctx.textAlign = "center"; ticks.forEach((gg) => ctx.fillText(String(gg), m.x + (gg + 0.5) * c, m.y + m.S + 10));
    ctx.fillStyle = C.ink2; ctx.font = F(11);
    ctx.fillText("target layer (state written)", m.x + m.S / 2, m.y + m.S + 26);
    ctx.save(); ctx.translate(g.wide ? m.x - 30 : 7, m.y + m.S / 2); ctx.rotate(-Math.PI / 2);
    ctx.fillText("source layer (state read)", 0, 0); ctx.restore();
    // scan cursor
    if (t >= T_SCAN[0] && t < T_SCAN[1]) {
      const kk = Math.min(n * n - 1, k), s = Math.floor(kk / n), gg = kk % n;
      ctx.strokeStyle = C.ink; ctx.lineWidth = 1.5; ctx.strokeRect(m.x + gg * c - 1, m.y + s * c - 1, c + 2, c + 2);
    }
    // rings left by the three experiments
    EXPS.forEach((ex, i) => {
      const f0 = T_FLY[0] + i * 0.005, f1 = T_FLY[1] - (EXPS.length - 1 - i) * 0.005;
      const ra = sm(seg(eIO(seg(t, f0, f1)), 0.85, 1));
      if (ra <= 0) return;
      ctx.globalAlpha = aF * ra; ctx.strokeStyle = C.ink; ctx.lineWidth = 1.5;
      if (ex.ctrl) ctx.setLineDash([2.5, 2]);
      ctx.strokeRect(m.x + ex.tgt * c - 2, m.y + ex.src * c - 2, c + 4, c + 4); ctx.setLineDash([]);
    });
    ctx.globalAlpha = aF;
    // hover / pin crosshair
    const act = mapInteractive(t) && (hover && hover.kind === "map" ? hover : pinned);
    if (act) {
      ctx.strokeStyle = C.ink; ctx.lineWidth = 1;
      ctx.strokeRect(m.x - 0.5, m.y + act.s * c - 0.5, m.S + 1, c + 1);
      ctx.setLineDash([3, 2]); ctx.strokeRect(m.x + act.g * c - 0.5, m.y - 0.5, c + 1, m.S + 1); ctx.setLineDash([]);
      ctx.lineWidth = 2; ctx.strokeRect(m.x + act.g * c - 1.5, m.y + act.s * c - 1.5, c + 3, c + 3);
    }
    drawMargins(ctx, t, aF);
    drawLegend(ctx, t, aF);
    drawMapHeader(ctx, t, aF);
    ctx.restore();
  }

  function drawMargins(ctx, t, aF) {
    const g = geo, m = g.map, n = N, c = m.c;
    const gp = eIO(seg(t, T_MARGIN[0], T_MARGIN[1]));
    if (gp <= 0) return;
    const sh = g.wide ? 24 : 14, base = m.y - 5;
    ctx.globalAlpha = aF * sm(seg(t, T_MARGIN[0], T_MARGIN[0] + 0.008));
    ctx.strokeStyle = C.axis; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(m.x, base + 0.5); ctx.lineTo(m.x + m.S, base + 0.5); ctx.stroke();
    ctx.fillStyle = rgba(C.rgb.ink2, 0.6);
    for (let gg = 0; gg < n; gg++) {
      const hh = (X.colCount[gg] / X.maxCol) * sh * gp;
      if (hh > 0) ctx.fillRect(m.x + gg * c + 1, base - hh, Math.max(1, c - 2), hh);
    }
    if (g.wide) {
      ctx.font = F(10); ctx.fillStyle = C.muted; ctx.textAlign = "left"; ctx.textBaseline = "middle";
      ctx.fillText("rank-1 cells", m.x + m.S + 8, base - sh / 2 - 5);
      ctx.fillText("per column", m.x + m.S + 8, base - sh / 2 + 7);
    }
    // brackets
    const ba = aF * sm(seg(t, T_BRACK[0], T_BRACK[1]));
    if (ba <= 0.002) return;
    ctx.globalAlpha = ba;
    const yb = base - sh - 7;
    const br = (g0, g1, l1, l2) => {
      const x0 = m.x + g0 * c + 1, x1 = m.x + g1 * c - 1;
      ctx.strokeStyle = C.ink2; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x0, yb + 4); ctx.lineTo(x0, yb); ctx.lineTo(x1, yb); ctx.lineTo(x1, yb + 4); ctx.stroke();
      const cx = clamp((x0 + x1) / 2, m.x + 40, m.x + m.S - 40);
      ctx.textAlign = "center"; ctx.textBaseline = "middle";
      if (g.wide) {
        ctx.font = F(11, 650); ctx.fillStyle = C.ink; ctx.fillText(l2, cx, yb - 8);
        ctx.font = F(11); ctx.fillStyle = C.ink2; ctx.fillText(l1, cx, yb - 21);
      } else {
        ctx.font = F(10.5, 650); ctx.fillStyle = C.ink; ctx.fillText(l2, cx, yb - 8);
      }
    };
    if (g.wide) {
      br(0, X.cStar, `written into layers 0–${X.cStar - 1}`, `${fmt(X.n1Below)} of ${fmt(X.n1)} rank-1 cells`);
      if (X.cStar < n) br(X.cStar, n, `layers ${X.cStar}–${n - 1}`, `${fmt(X.n1Above)} of ${fmt(X.n1)}`);
    } else {
      br(0, X.cStar, "", `layers 0–${X.cStar - 1}: ${fmt(X.n1Below)} of ${fmt(X.n1)}`);
      if (X.cStar < n) br(X.cStar, n, "", `${X.cStar}–${n - 1}: ${fmt(X.n1Above)}`);
    }
  }

  function legendPos(v) {
    const L = geo.leg;
    return L.horiz ? L.x + v * L.w : L.y + (1 - v) * L.h;
  }
  function drawLegend(ctx, t, aF) {
    const g = geo, L = g.leg;
    const a = aF * sm(seg(t, 0.676, 0.7));
    if (a <= 0.002) return;
    ctx.globalAlpha = a;
    const steps = 96;
    for (let i = 0; i < steps; i++) {
      const v0 = i / steps, v1 = (i + 1) / steps;
      ctx.fillStyle = M.heat((v0 + v1) / 2);
      if (L.horiz) ctx.fillRect(L.x + v0 * L.w, L.y, (v1 - v0) * L.w + 0.6, L.h);
      else ctx.fillRect(L.x, L.y + (1 - v1) * L.h, L.w, (v1 - v0) * L.h + 0.6);
    }
    ctx.strokeStyle = C.axis; ctx.lineWidth = 1;
    ctx.strokeRect(L.x - 0.5, L.y - 0.5, L.w + 1, L.h + 1);
    ctx.font = F(10.5); ctx.fillStyle = C.ink2; ctx.textBaseline = "middle";
    M.RANK_TICKS.forEach(([v, lab]) => {
      const p = legendPos(v);
      ctx.beginPath();
      if (L.horiz) { ctx.moveTo(p, L.y + L.h); ctx.lineTo(p, L.y + L.h + 4); ctx.stroke(); ctx.textAlign = "center"; ctx.fillText(lab, clamp(p, L.x + 8, L.x + L.w - 8), L.y + L.h + 12); }
      else { ctx.moveTo(L.x + L.w, p); ctx.lineTo(L.x + L.w + 4, p); ctx.stroke(); ctx.textAlign = "left"; ctx.fillText(lab, L.x + L.w + 7, p); }
    });
    ctx.fillStyle = C.ink2; ctx.font = F(11, 600);
    if (L.horiz) { ctx.textAlign = "left"; ctx.fillText("rank of the answer's first token", 0, g.legTitleY); }
    else {
      ctx.save(); ctx.translate(L.x + L.w + 38, L.y + L.h / 2); ctx.rotate(Math.PI / 2);
      ctx.textAlign = "center"; ctx.font = F(11); ctx.fillText("rank of the answer's first token", 0, 0); ctx.restore();
    }
    // marker at the shown pair's rank
    const p = ch4Pair(t);
    if (p && !p.scan && t >= T_DIAG[0]) {
      const v = X.G[p.s * N + p.g];
      if (cellRevealed(t, p.s, p.g) || p.arc) {
        const q = legendPos(v);
        ctx.fillStyle = C.ink; ctx.beginPath();
        if (L.horiz) { ctx.moveTo(q, L.y - 1); ctx.lineTo(q - 5, L.y - 8); ctx.lineTo(q + 5, L.y - 8); }
        else { ctx.moveTo(L.x - 1, q); ctx.lineTo(L.x - 8, q - 5); ctx.lineTo(L.x - 8, q + 5); }
        ctx.closePath(); ctx.fill();
      }
    }
  }

  function drawMapHeader(ctx, t, aF) {
    const g = geo, m = g.map, n = N;
    let text = null, bold = null;
    if (t >= T_SCAN[0] && t < T_SCAN[1]) {
      const k = revealedK(t);
      bold = `pairs scanned ${fmt(k)} / ${fmt(n * n)}`; text = ` · ranked 1st: ${fmt(X.cum1[k])}`;
    } else if (t >= T_SCAN[1]) {
      if (g.wide) { bold = `${fmt(X.n1)} of ${fmt(n * n)} pairs`; text = " rank the answer's first token 1st"; }
      else {
        const p = ch4Pair(t);
        if (p) { const v = X.G[p.s * n + p.g]; bold = `read ${p.s} → write ${p.g}: ${rankText(v)}`; text = p.s === p.g ? " (no patch)" : ` · unpatched ${ord(X.u)}`; }
      }
    } else if (t >= T_DIAG[0] && !g.wide) {
      const p = ch4Pair(t);
      if (p) { bold = `read ${p.s} → write ${p.g}`; text = " (same layer: no change)"; }
    }
    if (!bold) return;
    const y = g.wide ? 14 : 12;
    ctx.globalAlpha = aF; ctx.textBaseline = "middle"; ctx.textAlign = "left";
    ctx.font = F(g.wide ? 12 : 11.5, 650); ctx.fillStyle = C.ink;
    const x0 = g.wide ? m.x : 0;
    ctx.fillText(bold, x0, y);
    const w = ctx.measureText(bold).width;
    ctx.font = F(g.wide ? 12 : 11.5); ctx.fillStyle = C.ink2; ctx.fillText(text, x0 + w, y);
  }

  /* ================================================================== render */
  function render() {
    if (!geo || !C) return;
    const t = T;
    const ctx = M.fitCanvas(cv, geo.W, geo.H);
    ctx.clearRect(0, 0, geo.W, geo.H);
    const towerA = geo.wide ? 1 : 1 - sm(seg(t, 0.665, 0.678));
    drawTower(ctx, t, towerA);
    drawTrain(ctx, t);
    drawLadder(ctx, t);
    drawMap(ctx, t);
    drawChips(ctx, t);
    drawBigLabel(ctx, t);
    drawTokens(ctx, t);
    ctx.globalAlpha = 1;
    drawSpark(t);
    syncDom(t);
  }

  /* ================================================================== DOM */
  function set(el, key, val) {
    const k = el.__spId || (el.__spId = "e" + (++memoSeq));
    const mk = k + key;
    if (memo.get(mk) === val) return;
    memo.set(mk, val);
    if (key === "text") el.textContent = val;
    else if (key.startsWith("--")) el.style.setProperty(key, val);
    else if (key.startsWith("@")) { if (val == null) el.removeAttribute(key.slice(1)); else el.setAttribute(key.slice(1), val); }
    else el.style[key] = val;
  }
  function blk(el, f, o, ty) {
    set(el, "gridTemplateRows", `${Math.max(0, f).toFixed(3)}fr`);
    set(el, "opacity", o.toFixed(3));
    set(el, "visibility", f <= 0.001 || o <= 0.001 ? "hidden" : "visible");
    if (ty != null) set(el, "transform", `translateY(${ty.toFixed(1)}px)`);
  }

  /* left panel (wide) / card slot (narrow): blocks collapse and expand through grid-template-rows fractions */
  function syncPanel(t) {
    const wide = geo.wide;
    const e = t < CH_T[1] ? epochAt(t) : E;
    // panel blocks
    const ask = eIO(seg(t, 0.222, 0.246)), scan = eIO(seg(t, 0.665, 0.69));
    const st = eIO(seg(t, T_SCAN[1], T_SCAN[1] + 0.02)), cta = eIO(seg(t, 0.94, 0.955));
    blk(dom.bFacts, 1 - ask, 1);
    // scene 1 ("We fine-tune ... on two facts"): the fact cards are outlined for a moment
    const hl = bump(seg(t, 0.036, 0.066));
    const ring = hl > 0.004 && C ? `0 0 0 1.5px ${rgba(C.rgb.ink2, (0.85 * hl).toFixed(3))}` : "none";
    dom.factCards.forEach((c) => set(c, "boxShadow", wide ? ring : "none"));
    set(dom.facts, "boxShadow", wide ? "none" : ring);
    blk(dom.bChain, wide ? 1 - scan : 0, 1 - sm(seg(t, 0.665, 0.675)));
    dom.chainParts.forEach((p, i) => {
      const o = wide ? sm(seg(t, 0.05 + i * 0.004, 0.058 + i * 0.004)) : 1;
      set(p, "opacity", o.toFixed(3));
    });
    set(dom.chain, "@class", "sp-chain" + (t >= 0.228 ? " is-q" : ""));
    set(dom.btag, "opacity", sm(seg(t, 0.232, 0.25)).toFixed(3));
    if (wide) {
      blk(dom.bQ, ask * (1 - st), sm(seg(t, 0.232, 0.255)) * (1 - sm(seg(t, T_SCAN[1], T_SCAN[1] + 0.01))));
      blk(dom.bStats, st, sm(seg(t, T_SCAN[1] + 0.006, T_SCAN[1] + 0.026)));
    } else {
      // narrow: the stats replace the question card when the map comes in, and count up while the map is scanned
      blk(dom.bQ, ask * (1 - scan), sm(seg(t, 0.232, 0.255)) * (1 - sm(seg(t, 0.665, 0.675))));
      blk(dom.bStats, scan, sm(seg(t, 0.672, 0.69)));
    }
    let cnt = [X.n1, X.nSame, X.nWorse];
    if (!wide && t < T_SCAN[1]) {
      const k = revealedK(t), nd = t >= T_DIAG[1] ? N : Math.floor(N * seg(t, T_DIAG[0], T_DIAG[1]));
      const diagAhead = Math.max(0, nd - Math.ceil(k / (N + 1)));   // diagonal cells shown before the raster reaches them
      cnt = [X.cum1[k], X.cumSame[k] + diagAhead, X.cumWorse[k]];
    }
    dom.statN.forEach((b, i) => set(b, "text", fmt(cnt[i])));
    blk(dom.bCta, cta, sm(seg(t, 0.945, 0.965)));
    set(dom.ctaBtn, "@tabindex", cta > 0.5 ? null : "-1");
    blk(dom.bSpark, wide ? ask : 0, sm(seg(t, 0.236, 0.258)));
    // statuses
    const both = t >= 0.035 && D.mem[e] >= 1 - 1e-9;
    const tM = SC[3].t0, pop = bump(seg(t, tM, tM + 0.008));
    dom.factStatus.forEach((s) => {
      set(s, "visibility", both ? "visible" : "hidden");
      set(s, "transform", `scale(${(1 + 0.15 * pop).toFixed(3)})`);
    });
    const wrong = D.gen[E] < 0.5 ? sm(seg(t, T_DROP[0], T_DROP[1])) : 0;
    set(dom.qStatus, "opacity", wrong.toFixed(3));
  }
  /* Size the panel to the tallest state it must hold, so nothing is clipped and the stage never changes height.
     Narrow: the card slot above the canvas. Wide: the left column, at least as tall as the canvas. */
  function fitPanel() {
    if (!geo) return;
    const probes = [0.2, 0.35, 0.7, 0.93, 1];
    // CTA: one line; shorter label if the full one does not fit
    dom.ctaLab.textContent = "See this map change over training";
    if (dom.ctaBtn.offsetWidth > panelEl.clientWidth + 0.5) dom.ctaLab.textContent = "See it change over training";
    panelEl.style.height = "auto";
    let mx = 0;
    probes.forEach((t) => { syncPanel(t); mx = Math.max(mx, panelEl.scrollHeight); });
    panelEl.style.height = "";
    if (geo.wide) {
      stageEl.style.removeProperty("--sp-slot");
      if (mx > geo.H) { geo.H = Math.ceil(mx); cv.style.height = geo.H + "px"; }
      stageEl.style.setProperty("--sp-h", geo.H + "px");
    } else {
      stageEl.style.setProperty("--sp-slot", Math.ceil(mx + 2) + "px");
    }
  }

  function syncDom(t) {
    const wide = geo.wide;
    const si = sceneAt(t), sc = SC[si], ch = chapterAt(t);
    if (si !== lastScene) {
      lastScene = si;
      capEl.textContent = sc.cap;
      capEl.classList.toggle("is-big", sc.big);
    }
    set(rangeEl, "@aria-valuetext", `Chapter ${ch + 1}, ${CH_NAMES[ch]}: ${sc.cap}`);
    railBtns.forEach((b, i) => set(b, "@aria-current", i === ch ? "step" : null));
    railBars.forEach((b, i) => {
      const t0 = CH_T[i], t1 = CH_T[i + 1] != null ? CH_T[i + 1] : 1;
      set(b, "transform", `scaleX(${i === ch ? seg(t, t0, t1).toFixed(3) : 0})`);
    });
    const e = t < CH_T[1] ? epochAt(t) : E;
    const rec = Math.round(2 * D.mem[e]);
    set(dom.badgeE, "text", String(e));
    set(dom.badgeRec, "text", wide ? "" : ` · facts recalled ${rec}/2`);
    set(dom.badgeFt, "display", wide ? "" : "none");
    syncPanel(t);
    set(dom.sparkN, "text", `epoch ${E} / ${D.mem.length - 1}`);
    set(dom.sparkNote, "opacity", sm(seg(t, 0.95, 0.97)).toFixed(3));
    // scrubber + buttons
    const wf = wOfT(t) / WT;
    if (!dragging) { set(rangeEl, "value", String(Math.round(wf * 1000))); rangeEl.value = String(Math.round(wf * 1000)); }
    set(rangeEl, "--sp-fill", (wf * 100).toFixed(2) + "%");
    set(rangeEl, "--fill", (wf * 100).toFixed(2) + "%");
    const running = clk && clk.running;
    const icon = running ? "pause" : t >= 1 ? "replay" : "play";
    if (playBtn.__icon !== icon) { playBtn.__icon = icon; playBtn.innerHTML = ICON[icon]; }
    set(playBtn, "@aria-label", running ? "Pause" : t >= 1 ? "Replay" : "Play");
    set(playBtn, "@aria-pressed", running ? "true" : "false");
    set(capEl, "@aria-live", running ? "off" : "polite");
    set(dom.poster, "@hidden", t <= 0 && !running ? null : "");
  }

  /* ================================================================== playback */
  function startClock(until) {
    stopAt = until == null ? null : until;
    if (!clk) clk = M.clock(tick);
    clk.start();
    render();
  }
  function tick(dt) {
    T = tOfW(wOfT(T) + dt * speed);
    if (stopAt != null && T >= stopAt) { T = stopAt; stopAt = null; clk.stop(); }
    if (T >= 1) { T = 1; clk.stop(); }
    render();
  }
  function play() { if (T >= 1 || !mounted) return; startClock(null); }
  function pause() { if (clk) clk.stop(); stopAt = null; if (mounted) render(); }
  function seek(t) { T = clamp(+t || 0, 0, 1); if (mounted) render(); }

  function userToggle() {
    if (clk && clk.running) { pause(); M.userPaused("selfpatch", true); return; }
    if (T >= 1) T = 0;
    M.userPaused("selfpatch", false);
    startClock(null);
  }
  /* jump to a scene: playing -> continue from its start; paused -> play just that scene; reduced motion -> its end state */
  function goScene(i) {
    i = clamp(i, 0, SC.length - 1);
    const s = SC[i], running = clk && clk.running;
    hover = null;
    if (M.reducedMotion && !running) { T = Math.max(s.t0, s.t1 - 0.0005); render(); return; }
    T = s.t0;
    // playing a single scene (stopAt set): the jump plays the new scene instead of snapping back to the old stop
    if (running) { if (stopAt != null) stopAt = Math.max(s.t0, s.t1 - 0.0005); render(); return; }
    M.userPaused("selfpatch", true);
    startClock(Math.max(s.t0, s.t1 - 0.0005));
  }
  function goChapter(c) {
    const running = clk && clk.running;
    const t0 = CH_T[c], t1 = CH_T[c + 1] != null ? CH_T[c + 1] : 1;
    hover = null;
    if (M.reducedMotion && !running) { T = Math.max(t0, t1 - 0.0005); render(); return; }
    T = t0;
    if (running) { if (stopAt != null) stopAt = Math.max(t0, t1 - 0.0005); render(); return; }
    startClock(Math.max(t0, t1 - 0.0005));
  }
  function stepScene(d) {
    const si = sceneAt(T);
    const s = SC[si];
    if (d < 0) {
      const into = wOfT(T) - wOfT(s.t0);
      goScene(into > 1.2 || (clk && clk.running && into > 1.2) ? si : si - 1);
    } else goScene(si + 1);
  }

  /* ================================================================== interaction */
  function localXY(ev) {
    const r = cv.getBoundingClientRect();
    return [ev.clientX - r.left, ev.clientY - r.top];
  }
  function hitTest(x, y) {
    const g = geo, t = T, m = g.map;
    if (t >= T_DIAG[0] && mapInteractive(t) && x >= m.x && x < m.x + m.S && y >= m.y && y < m.y + m.S) {
      const s = Math.floor((y - m.y) / m.c), gg = Math.floor((x - m.x) / m.c);
      if (cellRevealed(t, s, gg)) return { kind: "map", s, g: gg };
    }
    if (g.chipBox && t >= T_DROP[1] && t < 0.665) {
      const [bx, by, bw, bh] = g.chipBox;
      if (x >= bx && x <= bx + bw && y >= by && y <= by + bh) return { kind: "chip" };
    }
    const towerVisible = g.wide || t < 0.668;
    if (towerVisible && t > 0.02 && y >= g.ty0 && y < g.towerB) {
      const l = N - 1 - Math.floor((y - g.ty0) / g.pitch);
      const col = g.cols.find((c) => x >= c.x - 1 && x <= c.x + c.w + 1);
      if (col && l >= 0 && l < N) return { kind: "tower", col: col.key, l };
    }
    return null;
  }
  function tipHTML(hv) {
    const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
    if (hv.kind === "map") {
      const v = X.G[hv.s * N + hv.g];
      if (hv.s === hv.g) return `<b>Layer ${hv.s} onto itself</b>: no change, which is the unpatched model (${ord(X.u)}).`;
      const worse = v < X.d0 - 0.01;
      return `<b>Read layer ${hv.s} → write layer ${hv.g}</b>: answer's first token ranked <b>${rankText(v)}</b> (unpatched: ${ord(X.u)})${worse ? " (worse than no patch)" : ""}.`;
    }
    if (hv.kind === "chip") return `The answer chip stands for the first token of “${esc(D.answer)}”.`;
    if (hv.kind === "spark") {
      const e = hv.e;
      return `<b>Epoch ${e}</b> · facts recalled ${Math.round(2 * D.mem[e])}/2 · two-hop answer ${D.gen[e] >= 0.5 ? "✓ correct" : "✗ wrong"} (training log)`;
    }
    if (hv.kind === "tower") {
      if (hv.col === "head") return `Hidden state of “${esc(D.head)}” after layer ${hv.l} (schematic); all of its sub-tokens are patched together.`;
      if (hv.col === "end") return `Hidden state at the end of the prompt after layer ${hv.l} (schematic); the answer's first token is predicted here.`;
      return `Hidden state of a question token before “${esc(D.head)}” after layer ${hv.l} (schematic).`;
    }
    return "";
  }
  function showTip(hv, cx, cy) {
    if (!hv) { tipEl.hidden = true; return; }
    tipEl.innerHTML = tipHTML(hv);
    tipEl.hidden = false;
    const sr = stageEl.getBoundingClientRect(), cr = cv.getBoundingClientRect();
    let x = cr.left - sr.left + cx + 14, y = cr.top - sr.top + cy + 16;
    const tw = tipEl.offsetWidth, th = tipEl.offsetHeight;
    if (x + tw > sr.width - 6) x = Math.max(6, cr.left - sr.left + cx - tw - 14);
    if (y + th > sr.height - 6) y = Math.max(6, cr.top - sr.top + cy - th - 14);
    tipEl.style.left = x + "px"; tipEl.style.top = y + "px";
  }
  const sameHover = (a, b) => (a === b) || (a && b && a.kind === b.kind && a.s === b.s && a.g === b.g && a.l === b.l && a.col === b.col && a.e === b.e);

  function onMove(ev) {
    const [x, y] = localXY(ev);
    const hv = hitTest(x, y);
    if (!sameHover(hv, hover)) { hover = hv; if (!(clk && clk.running)) render(); }
    showTip(hv, x, y);
  }
  function onLeave() { hover = null; tipEl.hidden = true; if (!(clk && clk.running)) render(); if (pinned) pinTip(); }
  function pinTip() {
    if (!pinned) return;
    const [cx, cy] = mapCellC(pinned.s, pinned.g);
    showTip({ kind: "map", s: pinned.s, g: pinned.g }, cx, cy);
  }
  function onDown(ev) {
    const [x, y] = localXY(ev);
    const hv = hitTest(x, y);
    if (hv && hv.kind === "map") {
      pinned = pinned && pinned.s === hv.s && pinned.g === hv.g ? null : { s: hv.s, g: hv.g };
      render();
      if (pinned) { announce(); if (ev.pointerType !== "mouse") pinTip(); } else if (ev.pointerType !== "mouse") tipEl.hidden = true;
    } else if (ev.pointerType !== "mouse") {
      hover = hv; showTip(hv, x, y); render();
    }
  }
  function announce() {
    if (!pinned) return;
    const tmp = document.createElement("div"); tmp.innerHTML = tipHTML({ kind: "map", s: pinned.s, g: pinned.g });
    dom.live.textContent = tmp.textContent;
  }
  function onSparkMove(ev) {
    const r = sparkCv.getBoundingClientRect(), x = ev.clientX - r.left;
    const nE = D.mem.length - 1, w = r.width;
    const e = clamp(Math.round(((x - SPARK.x) / (w - SPARK.x - SPARK.r)) * nE), 0, T >= 0.945 ? nE : E);
    const hv = { kind: "spark", e };
    if (!sameHover(hv, hover)) { hover = hv; drawSpark(T); }
    const sr = stageEl.getBoundingClientRect();
    tipEl.innerHTML = tipHTML(hv); tipEl.hidden = false;
    let tx = r.left - sr.left + x + 12, ty = r.top - sr.top - tipEl.offsetHeight - 6;
    if (tx + tipEl.offsetWidth > sr.width - 6) tx = sr.width - 6 - tipEl.offsetWidth;
    tipEl.style.left = tx + "px"; tipEl.style.top = Math.max(4, ty) + "px";
  }
  function onKey(ev) {
    const k = ev.key;
    if (ev.target !== stageEl) return;
    if (pinned && mapInteractive(T) && ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(k) && !ev.shiftKey) {
      const d = { ArrowLeft: [0, -1], ArrowRight: [0, 1], ArrowUp: [-1, 0], ArrowDown: [1, 0] }[k];
      pinned = { s: clamp(pinned.s + d[0], 0, N - 1), g: clamp(pinned.g + d[1], 0, N - 1) };
      render(); announce(); pinTip(); ev.preventDefault(); return;
    }
    if (k === "Escape" && pinned) { pinned = null; tipEl.hidden = true; render(); ev.preventDefault(); return; }
    if (k === "Enter" && mapInteractive(T) && T >= T_DIAG[0]) {
      pinned = pinned ? null : { s: D.demo.late.src, g: D.demo.late.tgt };
      render(); if (pinned) { announce(); pinTip(); } else tipEl.hidden = true;
      ev.preventDefault(); return;
    }
    if (k === " " || k === "k" || k === "K") { userToggle(); ev.preventDefault(); return; }
    if (k === "ArrowRight" || k === "ArrowLeft") {
      if (ev.shiftKey) { pause(); M.userPaused("selfpatch", true); seek(tOfW(wOfT(T) + (k === "ArrowRight" ? 1 : -1))); }
      else stepScene(k === "ArrowRight" ? 1 : -1);
      ev.preventDefault(); return;
    }
    if (k === "Home" || k === "End") { pause(); M.userPaused("selfpatch", true); seek(k === "Home" ? 0 : 1); ev.preventDefault(); return; }
    if (/^[1-4]$/.test(k)) { goChapter(+k - 1); ev.preventDefault(); }
  }

  /* ================================================================== build */
  function build() {
    host.innerHTML = "";
    root = h("div", "sp");
    // top: chapter rail + badge
    const top = h("div", "sp-top");
    const rail = h("div", "sp-rail");
    rail.setAttribute("role", "group");
    rail.setAttribute("aria-label", "Chapters");
    CH_NAMES.forEach((name, i) => {
      const b = h("button", "sp-chb");
      b.type = "button";
      b.innerHTML = `<span class="sp-n">${i + 1}</span><span>${name}</span><span class="sp-bar"></span>`;
      b.setAttribute("aria-label", `Chapter ${i + 1}: ${name}`);
      b.addEventListener("click", () => goChapter(i));
      railBtns.push(b); railBars.push(b.querySelector(".sp-bar"));
      rail.appendChild(b);
    });
    badgeEl = h("div", "sp-badge m2g-readout");
    dom.badgeFt = h("span", null, " · fine-tuned");
    badgeEl.append(D.model, dom.badgeFt, " · epoch ");
    dom.badgeE = h("b", null, "0");
    badgeEl.append(dom.badgeE, ` / ${D.mem.length - 1}`);
    dom.badgeRec = h("span");
    badgeEl.append(dom.badgeRec);
    top.append(rail, badgeEl);

    // stage
    stageEl = h("div", "sp-stage");
    stageEl.tabIndex = 0;
    stageEl.setAttribute("role", "group");
    stageEl.setAttribute("aria-roledescription", "animation");
    stageEl.setAttribute("aria-label", `Self-patching explainer on one ${D.model} example. Space plays or pauses, arrow keys step through scenes, 1 to 4 jump to a chapter; in the final map, Enter pins a cell and arrow keys move it.`);
    panelEl = h("div", "sp-panel");
    const mkBlk = (inner) => { const b = h("div", "sp-blk"), i = h("div", "sp-in"); i.appendChild(inner); b.appendChild(i); panelEl.appendChild(b); return b; };

    // facts (wide: two cards; narrow: one compact card under a shared header)
    const facts = h("div", "sp-facts");
    dom.facts = facts;
    dom.factCards = []; dom.factStatus = [];
    const fh = h("div", "sp-card-h sp-facts-h");
    const fst = h("span", "sp-status");
    fst.append(h("span", "sp-ok", "✓"), document.createTextNode(" both recalled"));
    fh.append(h("span", "sp-eyebrow", `Training data · ${D.facts.length} facts`), fst);
    facts.appendChild(fh);
    dom.factStatus.push(fst);
    D.facts.forEach((f, i) => {
      const card = h("div", "sp-card sp-fact");
      const hd = h("div", "sp-card-h");
      hd.append(h("span", "sp-eyebrow", `Fact ${i + 1} · training data`));
      const stt = h("span", "sp-status");
      stt.append(h("span", "sp-ok", "✓"), document.createTextNode(" recalled"));
      hd.append(stt);
      const q = h("div", "sp-card-q");
      q.appendChild(h("span", "sp-fn", `Fact ${i + 1}`));
      withEntity(q, f.prompt, D.head);
      const a = h("div", "sp-card-a");
      a.append("→ ", h("b", null, f.answer));
      card.append(hd, q, a);
      facts.appendChild(card);
      dom.factCards.push(card); dom.factStatus.push(stt);
    });
    dom.bFacts = mkBlk(facts);

    // chain
    const chain = h("div", "sp-chain");
    const parts = [h("span", "sp-ent", D.head), h("span", "sp-rel", `—${D.relations[0]}→`), h("span", "sp-bridge", D.bridge),
      h("span", "sp-rel", `—${D.relations[1]}→`), h("span", null, D.answer)];
    parts.forEach((p, i) => { if (i) chain.appendChild(document.createTextNode(" ")); chain.appendChild(p); });
    dom.btag = h("span", "sp-btag", `${D.bridge} is never named in the question`);
    chain.appendChild(dom.btag);
    dom.chain = chain; dom.chainParts = parts;
    dom.bChain = mkBlk(chain);

    // question
    const qc = h("div", "sp-card");
    const qh = h("div", "sp-card-h");
    qh.append(h("span", "sp-eyebrow", "Two-hop question · never trained"));
    const qq = withEntity(h("div", "sp-card-q"), D.question, D.head);
    const qa = h("div", "sp-card-a sp-qexp");
    qa.append("expected: ", h("b", null, D.answer));
    dom.qStatus = h("div", "sp-card-s");
    dom.qStatus.append(h("span", "sp-muted", "model's answer: "), h("span", "sp-bad", "✗"), document.createTextNode(" wrong"), h("span", "sp-muted", " (training log)"));
    qc.append(qh, qq, qa, dom.qStatus);
    dom.bQ = mkBlk(qc);

    // stats (Ch4)
    const stats = h("div", "sp-stats");
    stats.append(h("div", "sp-eyebrow", `All ${fmt(N * N)} pairs, epoch ${E}`));
    const ul = h("ul");
    dom.sw = []; dom.statN = [];
    [[X.n1, "rank it 1st", "", 1], [X.nSame, `leave it ${ord(X.u)}`, ", as with no patch", X.d0], [X.nWorse, "push it lower", "", 1 / 3]].forEach(([k, lab, more, v]) => {
      const li = h("li"), sw = h("i", "sp-sw"), num = h("b", null, fmt(k)), txt = h("span", "sp-slab", lab);
      if (more) txt.appendChild(h("span", "sp-wo", more));
      sw.__v = v; dom.sw.push(sw); dom.statN.push(num);
      li.append(sw, num, txt);
      ul.appendChild(li);
    });
    stats.appendChild(ul);
    stats.appendChild(h("p", "sp-prov", "Map cells recovered from the paper's per-epoch figures."));
    dom.bStats = mkBlk(stats);

    // CTA
    const cta = h("div", "sp-cta");
    dom.ctaBtn = h("button", "sp-ctabtn");
    dom.ctaBtn.type = "button";
    dom.ctaLab = h("span", null, "See this map change over training");
    dom.ctaBtn.append(dom.ctaLab, h("span", "sp-arr", "↓"));
    dom.ctaBtn.lastChild.setAttribute("aria-hidden", "true");
    dom.ctaBtn.addEventListener("click", handoff);
    cta.append(dom.ctaBtn, h("p", "sp-ctanote", "Picking the best cell needs the answer, so self-patching is a diagnostic, not a fix."));
    dom.bCta = mkBlk(cta);

    // sparkline (wide)
    const sp = h("div", "sp-spark");
    const sph = h("div", "sp-spark-h");
    sph.append(h("span", "sp-eyebrow", "Training log"));
    sparkN = h("span", "sp-spark-n");
    dom.sparkN = sparkN;
    sph.append(sparkN);
    sparkCv = h("canvas");
    sparkCv.setAttribute("role", "img");
    sparkCv.setAttribute("aria-label", `Training log: both facts recalled from epoch ${D.t_mem}; the two-hop answer is wrong until epoch ${D.t_gen}.`);
    dom.sparkNote = h("div", "sp-spark-note");
    dom.sparkNote.append(h("span", "sp-ok", "✓"), document.createTextNode(` two-hop answer correct from epoch ${D.t_gen}`));
    sp.append(sph, sparkCv, dom.sparkNote);
    dom.bSpark = mkBlk(sp);
    dom.bSpark.classList.add("sp-spark-blk");
    dom.bSpark.style.marginTop = "auto";
    sparkCv.addEventListener("pointermove", onSparkMove);
    sparkCv.addEventListener("pointerleave", () => { if (hover && hover.kind === "spark") hover = null; tipEl.hidden = true; drawSpark(T); });

    mainEl = h("div", "sp-main");
    cv = h("canvas", "sp-cv");
    cv.setAttribute("aria-hidden", "true");
    mainEl.appendChild(cv);
    // poster: before the first play, a play affordance over the first frame
    dom.poster = h("button", "sp-poster");
    dom.poster.type = "button";
    const mm = Math.floor(WT / 60), ss = Math.round(WT - 60 * mm);
    dom.poster.innerHTML = `${ICON.play}<span>Play the explainer</span><span class="sp-poster-d">${mm}:${String(ss).padStart(2, "0")}</span>`;
    dom.poster.setAttribute("aria-label", `Play the self-patching explainer (${mm} min ${ss} s)`);
    dom.poster.hidden = true;
    dom.poster.addEventListener("click", () => { userToggle(); stageEl.focus({ preventScroll: true }); });
    mainEl.appendChild(dom.poster);
    tipEl = h("div", "m2g-tip sp-tip");
    tipEl.setAttribute("role", "tooltip");
    tipEl.hidden = true;
    dom.live = h("div", "sp-sr");
    dom.live.setAttribute("aria-live", "polite");
    stageEl.append(panelEl, mainEl, tipEl, dom.live);

    // caption
    capEl = h("p", "sp-cap");
    capEl.setAttribute("aria-live", "polite");

    // controls
    const ctl = h("div", "m2g-controls sp-ctl");
    const btns = h("div", "sp-btns");
    const ib = (name, label) => { const b = h("button", "m2g-iconbtn sp-ib"); b.type = "button"; b.innerHTML = ICON[name]; b.setAttribute("aria-label", label); return b; };
    prevBtn = ib("prev", "Previous scene"); playBtn = ib("play", "Play"); nextBtn = ib("next", "Next scene");
    playBtn.setAttribute("aria-pressed", "false");
    prevBtn.addEventListener("click", () => stepScene(-1));
    nextBtn.addEventListener("click", () => stepScene(1));
    playBtn.addEventListener("click", userToggle);
    btns.append(prevBtn, playBtn, nextBtn);
    const scrub = h("div", "sp-scrub");
    rangeEl = h("input", "m2g-range sp-range");
    rangeEl.type = "range"; rangeEl.min = "0"; rangeEl.max = "1000"; rangeEl.step = "1"; rangeEl.value = "0";
    rangeEl.setAttribute("aria-label", "Position in the explainer");
    rangeEl.addEventListener("input", () => {
      dragging = true;
      if (clk && clk.running) { pause(); M.userPaused("selfpatch", true); }
      T = tOfW((+rangeEl.value / 1000) * WT); render();
    });
    rangeEl.addEventListener("change", () => { dragging = false; render(); });
    const ticks = h("div", "sp-ticks");
    CH_T.slice(1).forEach((tc) => { const i = h("i"); i.style.left = ((wOfT(tc) / WT) * 100).toFixed(2) + "%"; ticks.appendChild(i); });
    scrub.append(rangeEl, ticks);
    const spd = h("div", "m2g-seg sp-speed");
    spd.setAttribute("role", "group"); spd.setAttribute("aria-label", "Playback speed");
    [0.5, 1, 2].forEach((v) => {
      const b = h("button", null, `${v}×`);
      b.type = "button"; b.setAttribute("aria-pressed", v === speed ? "true" : "false");
      b.setAttribute("aria-label", `Speed ${v}×`);
      b.addEventListener("click", () => { speed = v; speedBtns.forEach((x) => x.setAttribute("aria-pressed", x === b ? "true" : "false")); });
      speedBtns.push(b); spd.appendChild(b);
    });
    ctl.append(btns, scrub, spd);

    // (no footnote under the controls: the figcaption and the "About the metric" note below the figure carry it;
    //  the map's provenance sits with the Ch4 stats, the schematic tag on the tower)
    dom.top = top;
    root.append(top, stageEl, capEl, ctl);
    host.appendChild(root);

    cv.addEventListener("pointermove", onMove);
    cv.addEventListener("pointerleave", onLeave);
    cv.addEventListener("pointerdown", onDown);
    stageEl.addEventListener("keydown", onKey);
  }

  function paintSwatches() {
    (dom.sw || []).forEach((s) => { s.style.background = M.heat(s.__v); });
  }

  function handoff() {
    host.dispatchEvent(new CustomEvent("m2g:handoff", { bubbles: true, detail: { run: D.id, epoch: D.demo.epoch } }));
    const tgt = document.getElementById("viz-permeation");
    if (tgt) tgt.scrollIntoView({ behavior: M.reducedMotion ? "auto" : "smooth", block: "start" });
  }

  function relayout() {
    if (!root) return;
    const w = root.clientWidth;
    if (!w) return;
    lastW = w;
    computeGeo();
    fitPanel();
    render();
  }

  function mount(el, data) {
    D = data && data.selfpatch;
    if (!D) throw new Error("M2G_DATA.selfpatch is missing");
    host = el;
    N = D.n_layers; E = D.demo.epoch;
    X = derive();
    EXPS = buildExps();
    SC = buildScenes();
    buildWarp();
    injectStyle();
    build();
    readTokens();
    paintSwatches();
    mounted = true;
    T = M.reducedMotion ? 1 : 0;
    relayout();
    if ("ResizeObserver" in window) {
      new ResizeObserver(() => { if (root.clientWidth !== lastW) relayout(); }).observe(root);
    } else window.addEventListener("resize", relayout);
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => { readTokens(); relayout(); });
    const retheme = () => { readTokens(); paintSwatches(); render(); };
    M.onTheme(retheme);
    new MutationObserver(retheme).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  }

  M.register("selfpatch", { mount, seek, play, pause });
})();
