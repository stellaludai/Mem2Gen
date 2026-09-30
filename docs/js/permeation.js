/* Knowledge permeation explorer (#viz-permeation) and mosaic (#viz-mosaic).
 *
 * Interactive ports of the README animations assets/permeation*.gif and assets/permeation_mosaic*.gif
 * (analysis/permeation/make_permeation_gif.py, make_permeation_mosaic_gif.py): same runs, texts, statistics and timing
 * rules, plus scrubbing, hover read-outs, crisp rendering at any size and theme switching.
 *
 * Map cell (s, t): the head entity's hidden state after layer s is written into layer t of the same two-hop prompt;
 * value = reciprocal rank of the answer's FIRST token (1 = ranked 1st). Row = source layer, column = target layer, the
 * diagonal is the unpatched model. Only scanned epochs have maps: between two scans the map cross-fades in the last
 * third of the interval (it snaps into the generalization scan), and every read-out names the last scanned epoch.
 * The check / cross and the diagonal casing follow the training log (greedy two-hop answer correct, no patch).
 */
(function () {
  "use strict";
  const M = window.M2G;
  if (!M) return;

  const EPOCHS = 30;
  const EPS = 1e-6;
  const floorE = (x) => Math.floor(x + EPS);
  const snap = (x) => Math.round(x * 1e6) / 1e6;
  const ML = 44;              // left margin of maps and charts (y tick labels + axis title)

  /* ------------------------------------------------------------------ styles (component-scoped fallbacks) */
  const CSS = `
.pm-root{position:relative;font-family:var(--font-sans,Inter,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif);color:var(--ink);font-size:14px;line-height:1.45;text-align:left}
.pm-root *,.pm-root *::before,.pm-root *::after{box-sizing:border-box}
.pm-root canvas{display:block;touch-action:pan-y;-webkit-tap-highlight-color:transparent}
.pm-head{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:flex-end;gap:14px 32px;margin:0 0 22px}
.pm-htext{flex:1 1 360px;min-width:0}
.pm-sub{font-size:13px;line-height:1.45;color:var(--ink-2);padding-bottom:6px}
.pm-sub .pm-ico{vertical-align:-2px}
.pm-hside{display:flex;align-items:flex-end;gap:14px 28px;flex-wrap:wrap}
.pm-clock{font-size:22px;line-height:1;font-weight:600;font-variant-numeric:tabular-nums;white-space:nowrap;color:var(--ink);padding-bottom:6px}
.pm-clock small{font-size:13px;font-weight:450;color:var(--muted);margin-left:7px}
.pm-legend{font-size:12px;line-height:1.2;color:var(--ink-2)}
.pm-legt{padding:0 10px 5px;text-align:right}
.pm-panels{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);column-gap:40px;row-gap:36px;position:relative;outline:none;border-radius:6px}
.pm-panels:focus-visible,.pm-groups:focus-visible{box-shadow:0 0 0 2px var(--focus)}
.pm-panels:not(.pm-stack)::before{content:"";position:absolute;left:50%;top:0;bottom:0;border-left:1px solid var(--grid)}
.pm-panels.pm-stack{grid-template-columns:minmax(0,1fr)}
.pm-panel{min-width:0}
.pm-pt{display:flex;align-items:center;gap:7px;font-size:16px;font-weight:600;color:var(--ink);margin-bottom:6px}
.pm-pt .pm-ico{width:16px;height:16px}
.pm-meta{display:grid;grid-template-columns:max-content minmax(0,1fr);gap:2px 12px;margin:0 0 14px;font-size:12.5px;line-height:1.4}
.pm-meta dt{color:var(--muted)}
.pm-meta dd{margin:0;color:var(--ink-2)}
.pm-body{display:flex;align-items:flex-start;gap:20px}
.pm-body.pm-below{flex-direction:column;gap:14px}
.pm-mapwrap{position:relative;flex:none}
.pm-map{cursor:crosshair}
.pm-ring{position:absolute;pointer-events:none;border-radius:2px}
.pm-flash{animation:pm-flash 2.6s ease-out}
@keyframes pm-flash{0%,55%{box-shadow:0 0 0 3px var(--accent)}100%{box-shadow:0 0 0 3px transparent}}
.pm-reads{flex:1 1 0;min-width:0;display:flex;flex-direction:column;gap:16px;padding-top:1px}
.pm-below .pm-reads{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:12px 18px;width:100%;padding-left:${ML}px}
.pm-rl{display:flex;align-items:center;gap:7px;font-size:12.5px;line-height:1.35;color:var(--ink-2)}
.pm-rv{display:flex;align-items:center;gap:6px;margin-top:3px;font-size:22px;line-height:1.15;font-weight:600;font-variant-numeric:tabular-nums;color:var(--ink);white-space:nowrap}
.pm-rv.pm-big{font-size:34px;letter-spacing:-.02em;line-height:1.05;margin-top:5px}
.pm-rs{margin-top:3px;font-size:12px;color:var(--muted);font-variant-numeric:tabular-nums}
.pm-note{font-size:13px;line-height:1.4;font-weight:600;color:var(--ink);min-height:3.6em}
.pm-below .pm-note{min-height:0;align-self:end}
.pm-key{flex:none;display:inline-block;width:14px;height:2px;border-radius:1px}
.pm-key-gen{height:3px}
.pm-ico{flex:none;width:1em;height:1em;display:inline-block}
.pm-good{color:var(--good)}.pm-bad{color:var(--bad)}
.pm-chart{margin-top:14px;cursor:pointer}
.pm-controls{margin-top:20px}
.pm-readout{display:none}
.pm-narrow .pm-readout{display:inline}
.pm-narrow .pm-controls{position:sticky;bottom:0;z-index:5;background:var(--pm-surface,var(--stage));box-shadow:0 -1px 0 var(--grid);padding:8px 0;margin-bottom:-8px;flex-wrap:nowrap}
.pm-groups{display:grid;grid-template-columns:auto auto;justify-content:space-between;column-gap:40px;row-gap:26px;position:relative;outline:none;border-radius:6px}
.pm-groups:not(.pm-stack)::before{content:"";position:absolute;left:50%;top:0;bottom:0;border-left:1px solid var(--grid)}
.pm-groups.pm-stack{grid-template-columns:auto;justify-content:center}
.pm-gt{font-size:15px;font-weight:600;color:var(--ink);margin-bottom:10px}
.pm-gt small{font-size:12.5px;font-weight:450;color:var(--muted);margin-left:8px}
.pm-tiles{display:grid}
.pm-tile{position:relative}
.pm-tile canvas{cursor:crosshair}
.pm-badge{position:absolute;top:4px;right:4px;width:20px;height:20px;border-radius:50%;background:var(--pm-surface,var(--stage));color:var(--good);display:grid;place-items:center;opacity:0;transform:scale(.5);transition:opacity .18s ease-out,transform .18s ease-out;pointer-events:none}
.pm-badge.pm-on{opacity:1;transform:none}
.pm-badge svg{width:13px;height:13px}
.pm-small .pm-badge{width:16px;height:16px;top:3px;right:3px}.pm-small .pm-badge svg{width:11px;height:11px}
@media (prefers-reduced-motion:reduce){.pm-badge{transition:none}.pm-flash{animation:none}}
:where(.pm-btn){flex:none;width:32px;height:32px;border-radius:50%;border:0;padding:0;cursor:pointer;display:inline-grid;place-items:center;background:var(--surface-2);color:var(--ink)}
:where(.pm-btn) svg{width:16px;height:16px}
:where(.pm-controls){display:flex;align-items:center;gap:12px;min-height:36px;font-size:13px;color:var(--ink-2)}
:where(.pm-range){flex:1 1 160px;min-width:0;height:24px;margin:0;accent-color:var(--ink-2)}
:where(.pm-tip){position:absolute;z-index:30;pointer-events:none;max-width:290px;background:var(--tip-bg,var(--surface));color:var(--ink-2);border:1px solid var(--border);border-radius:8px;padding:9px 11px;font-size:13px;line-height:1.35;box-shadow:0 6px 20px rgba(0,0,0,.14)}
.pm-tip[hidden]{display:none}
:where(.pm-tip) .tip-h{font-weight:600;color:var(--ink);margin-bottom:5px}
:where(.pm-tip) .tip-row{display:flex;align-items:center;gap:8px;margin:3px 0}
:where(.pm-tip) .tip-key{flex:none;width:12px;height:2px;border-radius:1px;background:var(--k,var(--ink-2))}
:where(.pm-tip) .tip-v{font-weight:600;color:var(--ink)}
:where(.pm-tip) .tip-f{margin-top:6px;font-size:12px;color:var(--muted)}
.pm-tip .pm-ico{width:13px;height:13px}
`;
  function injectStyle() {
    if (document.getElementById("pm-style")) return;
    const s = document.createElement("style");
    s.id = "pm-style";
    s.textContent = CSS;
    document.head.insertBefore(s, document.head.firstChild);   // style.css (later) wins over the :where() fallbacks
  }

  /* ------------------------------------------------------------------ small helpers */
  function el(tag, cls, text, attrs) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    if (attrs) for (const k in attrs) e.setAttribute(k, attrs[k]);
    return e;
  }
  function setText(node, s) { if (node.textContent !== s) node.textContent = s; }
  const svg = (inner, cls) => `<svg viewBox="0 0 16 16" class="${cls || ""}" aria-hidden="true" focusable="false">${inner}</svg>`;
  const ICON = {
    play: svg('<path d="M5 3.2v9.6L12.8 8z" fill="currentColor"/>'),
    pause: svg('<path d="M4.3 3h2.6v10H4.3zM9.1 3h2.6v10H9.1z" fill="currentColor"/>'),
    replay: svg('<path d="M3.2 8.6A4.9 4.9 0 1 0 4.8 4.4" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/><path d="M3.6 1.9v3.4H7" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>'),
    ok: svg('<path d="M2.8 8.7l3.3 3.2 7.1-7.6" fill="none" stroke="currentColor" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round"/>', "pm-ico"),
    bad: svg('<path d="M4 4l8 8M12 4l-8 8" fill="none" stroke="currentColor" stroke-width="2.3" stroke-linecap="round"/>', "pm-ico"),
  };
  function glyph(ok) {
    const s = el("span", ok ? "pm-good" : "pm-bad");
    s.style.display = "inline-flex";
    s.innerHTML = ok ? ICON.ok : ICON.bad;
    return s;
  }
  const fmtPct = (p) => (p >= 10 || p === 0 ? `${Math.round(p)}%` : `${p.toFixed(1)}%`);
  const fmtK = (k) => (k > 0 ? `+${k}` : k < 0 ? `\u2212${-k}` : "0");
  function rankText(v) { return v < 0.19 ? "ranked below 5th" : `ranked ${M.ordinal(Math.round(1 / v))}`; }

  /* ------------------------------------------------------------------ colours (refreshed on theme change) */
  const C = { lut: [], css: [] };
  function surfaceOf(node) {
    for (let e = node; e && e.nodeType === 1; e = e.parentElement) {
      const m = getComputedStyle(e).backgroundColor.match(/rgba?\(([^)]+)\)/);
      if (m) {
        const p = m[1].split(/[\s,/]+/).filter(Boolean).map(Number);
        if (p.length < 4 || p[3] > 0.5) return `rgb(${p[0]},${p[1]},${p[2]})`;
      }
    }
    return M.css("--stage") || "#fff";
  }
  function refreshColors(root) {
    for (const n of ["ink", "ink-2", "muted", "grid", "band", "mem", "gen", "good", "bad"]) C[n] = M.css("--" + n);
    C.dark = M.isDark();
    C.wash = C.dark ? "rgba(255,255,255,.16)" : "rgba(11,11,11,.09)";
    C.lut = [];
    C.css = [];
    for (let k = 0; k < 256; k++) { const c = M.heatRGB(k / 255); C.lut.push(c); C.css.push(`rgb(${c[0]},${c[1]},${c[2]})`); }
    C.font = M.css("--font-sans") || "system-ui, sans-serif";
    root.style.setProperty("--pm-surface", surfaceOf(root));
  }
  const font = (px, w) => `${w || 400} ${px}px ${C.font}`;

  /* ------------------------------------------------------------------ runs and scans */
  function prepRun(r) {
    const n = r.n_layers;
    const grids = r.grids.map((b) => M.decodeGrid(b, n));
    const cov = grids.map((g) => {   // % of OFF-diagonal layer pairs that rank the answer's first token 1st
      let c = 0;
      for (let s = 0; s < n; s++) for (let t = 0; t < n; t++) if (s !== t && g[s * n + t] >= M.FULL) c++;
      return (100 * c) / (n * n - n);
    });
    return Object.assign({}, r, { n, grids, cov, buf: new Float32Array(n * n) });
  }
  /* Displayed map at fractional epoch x: {vals|null, i = index of the last scan <= x (-1 before the first), w = fade}. */
  function frameAt(run, x) {
    const ep = run.epochs, last = ep.length - 1;
    if (x < ep[0] - EPS) return { vals: null, i: -1, w: 0 };
    if (x >= ep[last] - EPS) return { vals: run.grids[last], i: last, w: 0 };
    let i = 0;
    while (ep[i + 1] <= x + EPS) i++;
    const a = ep[i], b = ep[i + 1];
    const w = b === run.t_gen ? 0 : M.lateFade((x - a) / (b - a));
    if (w <= 0) return { vals: run.grids[i], i, w: 0 };
    const A = run.grids[i], B = run.grids[i + 1], out = run.buf;
    for (let k = 0; k < out.length; k++) out[k] = A[k] + (B[k] - A[k]) * w;
    return { vals: out, i, w };
  }
  const answers = (run, e) => run.gen[Math.min(EPOCHS, Math.max(0, e))] >= 1;

  /* ------------------------------------------------------------------ timeline: [duration s, x0, x1] segments */
  function timeline(segs) {
    let T = 0;
    const S = segs.map(([d, a, b]) => { const s = { t0: T, d, a, b }; T += d; return s; });
    return {
      total: T,
      xmin: S[0].a,
      xmax: S[S.length - 1].b,
      x(t) {
        for (const s of S) if (t < s.t0 + s.d) return snap(s.a + ((s.b - s.a) * (t - s.t0)) / s.d);
        return S[S.length - 1].b;
      },
      t(x) {      // first time at which the clock reaches x
        for (const s of S) {
          if (s.b >= x - EPS) return s.b === s.a ? s.t0 : s.t0 + s.d * M.clamp((x - s.a) / (s.b - s.a), 0, 1);
        }
        return T;
      },
    };
  }

  /* ------------------------------------------------------------------ player + controls (shared) */
  function makePlayer(id, tl, draw) {
    const P = { time: 0, x: tl.xmin, playing: false, onState: null };
    const clock = M.clock((dt) => {
      P.time += dt;
      if (P.time >= tl.total) P.time -= tl.total;      // hold at the end, then loop
      P.x = tl.x(P.time);
      draw(P.x);
    });
    P.atEnd = () => P.x >= tl.xmax - EPS;
    P.play = () => {
      if (P.atEnd() && P.time >= tl.t(tl.xmax) - EPS) P.time = 0;
      P.x = tl.x(P.time);
      P.playing = true;
      clock.start();
      draw(P.x);
      if (P.onState) P.onState();
    };
    P.pause = () => {
      clock.stop();
      P.playing = false;
      draw(P.x);
      if (P.onState) P.onState();
    };
    P.seekT = (t) => { P.time = M.clamp(t, 0, 1) * tl.total; P.x = tl.x(P.time); draw(P.x); };
    P.setX = (x) => { P.x = M.clamp(Math.round(x), tl.xmin, tl.xmax); P.time = tl.t(P.x); draw(P.x); };
    P.user = (fn) => { M.userPaused(id, true); P.pause(); fn(); };      // a user action takes the clock
    return P;
  }
  function makeControls(P, opts) {
    const bar = el("div", "m2g-controls pm-controls");
    const btn = el("button", "m2g-iconbtn pm-btn", null, { type: "button", "aria-label": "Play", "aria-pressed": "false" });
    const range = el("input", "m2g-range pm-range", null, {
      type: "range", min: String(opts.min), max: String(opts.max), step: "1", "aria-label": opts.label,
    });
    const out = el("span", "m2g-readout pm-readout", null, { "aria-hidden": "true" });
    bar.append(btn, range, out);
    btn.addEventListener("click", () => {
      if (P.playing) { M.userPaused(opts.id, true); P.pause(); } else { M.userPaused(opts.id, false); P.play(); }
    });
    range.addEventListener("input", () => { const v = +range.value; P.user(() => P.setX(v)); });
    let last = "";
    return {
      el: bar,
      update(e) {
        const mode = P.playing ? "pause" : P.atEnd() ? "replay" : "play";
        if (mode !== last) {
          last = mode;
          btn.innerHTML = ICON[mode];
          btn.setAttribute("aria-label", mode === "pause" ? "Pause" : mode === "replay" ? "Replay from the start" : "Play");
          btn.setAttribute("aria-pressed", P.playing ? "true" : "false");
        }
        if (range.value !== String(e)) range.value = String(e);
        range.style.setProperty("--fill", `${(100 * (e - opts.min)) / (opts.max - opts.min)}%`);
        range.setAttribute("aria-valuetext", opts.valueText(e));
        setText(out, opts.readout(e));
      },
    };
  }

  /* ------------------------------------------------------------------ tooltip */
  function makeTip(root) {
    const tip = el("div", "m2g-tip pm-tip", null, { "aria-hidden": "true" });
    tip.hidden = true;
    root.appendChild(tip);
    return {
      owner: null,
      show(nodes, cx, cy) {         // cx, cy: client coordinates of the pointer
        tip.replaceChildren(...nodes);
        tip.hidden = false;
        const r = root.getBoundingClientRect(), W = root.clientWidth;
        const px = cx - r.left, py = cy - r.top, w = tip.offsetWidth, h = tip.offsetHeight;
        let left = px + 16, top = py - h - 14;
        if (left + w > W) left = px - w - 16;
        if (left < 0) left = M.clamp(px - w / 2, 0, Math.max(0, W - w));
        if (top < -r.top + 8 && py + 22 + h < root.clientHeight) top = py + 22;
        tip.style.left = `${Math.round(left)}px`;
        tip.style.top = `${Math.round(top)}px`;
      },
      hide() { tip.hidden = true; this.owner = null; },
    };
  }
  function tipRow(key, value, label) {
    const r = el("div", "tip-row");
    if (key) { const k = el("span", "tip-key"); k.style.setProperty("--k", key); k.style.height = "2px"; r.append(k); }
    if (value instanceof Node) r.append(value); else r.append(el("span", "tip-v", value));
    if (label) r.append(el("span", "tip-l", label));
    return r;
  }
  function cellTip(run, fr, s, t, head) {
    const nodes = [];
    if (head) nodes.push(el("div", "tip-f", head));
    if (fr.i < 0) { nodes.push(el("div", "tip-h", "not scanned yet")); return nodes; }
    const v = run.grids[fr.i][s * run.n + t];
    nodes.push(el("div", "tip-h", s === t ? `layer ${s}, no patch (s = t)` : `layer ${s} \u2192 layer ${t}`));
    nodes.push(el("div", "tip-l", `answer's first token ${rankText(v)}`));
    nodes.push(el("div", "tip-f", `map of epoch ${run.epochs[fr.i]}`));
    if (head) nodes.push(nodes.shift());        // run info last, as a footnote
    return nodes;
  }

  /* ------------------------------------------------------------------ canvas drawing */
  function edges(o, S, n) {
    const dpr = Math.min(window.devicePixelRatio || 1, 3), e = [];
    for (let i = 0; i <= n; i++) e.push(Math.round((o + (S * i) / n) * dpr) / dpr);
    return e;
  }
  /* The large map: cells with hairline gaps (transparent, so the stage shows through), diagonal, hover, axes. */
  function drawBigMap(ctx, run, fr, ans, L, hover) {
    const n = run.n, S = L.S, ox = L.ox, oy = L.oy;
    const ex = edges(ox, S, n), ey = edges(oy, S, n);
    const dpr = Math.min(window.devicePixelRatio || 1, 3);
    const cell = S / n, g = Math.max(1, Math.round(cell * 0.1 * dpr)) / dpr;
    ctx.clearRect(0, 0, L.w, L.h);
    if (!fr.vals) {
      ctx.fillStyle = C.grid;
      for (let i = 0; i <= n; i++) {
        const o = i === 0 ? 0 : i === n ? -g : -g / 2;
        ctx.fillRect(ex[i] + o, ey[0], g, S);
        ctx.fillRect(ex[0], ey[i] + o, S, g);
      }
      const lines = L.emptyNote;
      ctx.font = font(12.5);
      const tw = Math.max(...lines.map((s) => ctx.measureText(s).width)) + 18, th = lines.length * 18 + 12;
      ctx.clearRect(ox + S / 2 - tw / 2, oy + S / 2 - th / 2, tw, th);
      ctx.fillStyle = C["ink-2"];
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      lines.forEach((s, k) => ctx.fillText(s, ox + S / 2, oy + S / 2 + (k - (lines.length - 1) / 2) * 18));
    } else {
      const v = fr.vals;
      for (let s = 0; s < n; s++) {
        for (let t = 0; t < n; t++) {
          ctx.fillStyle = C.css[Math.round(v[s * n + t] * 255)];
          ctx.fillRect(ex[t], ey[s], ex[t + 1] - ex[t] - g, ey[s + 1] - ey[s] - g);
        }
      }
      if (hover) {
        const [hs, ht] = hover;
        ctx.fillStyle = C.wash;
        ctx.fillRect(ex[0], ey[hs], S - g, ey[hs + 1] - ey[hs] - g);
        ctx.fillRect(ex[ht], ey[0], ex[ht + 1] - ex[ht] - g, S - g);
      }
      const box = (i) => [ex[i] - g / 2, ey[i] - g / 2, ex[i + 1] - ex[i], ey[i + 1] - ey[i]];
      ctx.strokeStyle = C.ink;
      if (ans) {            // the model answers with no patch: ink casing on every diagonal cell (as in the GIF)
        ctx.globalCompositeOperation = "destination-out";
        ctx.lineWidth = Math.max(2.6, cell * 0.4);
        for (let i = 0; i < n; i++) ctx.strokeRect(...box(i));
        ctx.globalCompositeOperation = "source-over";
        ctx.lineWidth = Math.max(1.4, cell * 0.22);
        for (let i = 0; i < n; i++) ctx.strokeRect(...box(i));
      } else {              // subtle staircase along both sides of the diagonal band
        ctx.strokeStyle = C["ink-2"];
        ctx.globalAlpha = 0.75;
        ctx.lineWidth = g;
        for (let i = 0; i < n; i++) ctx.strokeRect(...box(i));
        ctx.globalAlpha = 1;
      }
      if (hover) {
        const [hs, ht] = hover;
        ctx.strokeStyle = C.ink;
        ctx.lineWidth = 1.5;
        ctx.strokeRect(ex[ht] - g / 2, ey[hs] - g / 2, ex[ht + 1] - ex[ht], ey[hs + 1] - ey[hs]);
      }
    }
    // axes
    ctx.fillStyle = C["ink-2"];
    ctx.font = font(11);
    const ticks = [0, 8, 16, 24, n - 1];
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (const t of ticks) ctx.fillText(String(t), (ex[t] + ex[t + 1] - g) / 2, oy + S + 5);
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const s of ticks) ctx.fillText(String(s), ox - 7, (ey[s] + ey[s + 1] - g) / 2);
    ctx.font = font(12);
    ctx.textAlign = "center";
    ctx.textBaseline = "alphabetic";
    ctx.fillText("target layer (state written)", ox + S / 2, oy + S + 35);
    ctx.save();
    ctx.translate(12, oy + S / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textBaseline = "middle";
    ctx.fillText("source layer (state read)", 0, 0);
    ctx.restore();
  }
  /* A mosaic tile: n x n pixels scaled up nearest-neighbour, thin diagonal (cased while the model answers), frame. */
  const scratch = {};
  function drawTile(ctx, run, fr, ans, T, hover) {
    const n = run.n;
    let sc = scratch[n];
    if (!sc) {
      const cv = document.createElement("canvas");
      cv.width = cv.height = n;
      const c2 = cv.getContext("2d");
      sc = scratch[n] = { cv, c2, img: c2.createImageData(n, n) };
    }
    const d = sc.img.data, v = fr.vals;
    for (let i = 0; i < n * n; i++) {
      const c = C.lut[Math.round(v[i] * 255)];
      d[4 * i] = c[0]; d[4 * i + 1] = c[1]; d[4 * i + 2] = c[2]; d[4 * i + 3] = 255;
    }
    sc.c2.putImageData(sc.img, 0, 0);
    ctx.clearRect(0, 0, T, T);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(sc.cv, 0, 0, T, T);
    const u = T / n;
    if (hover) {
      const [hs, ht] = hover;
      ctx.fillStyle = C.wash;
      ctx.fillRect(0, hs * u, T, u);
      ctx.fillRect(ht * u, 0, u, T);
    }
    ctx.lineCap = "butt";
    if (ans) {
      ctx.globalCompositeOperation = "destination-out";
      ctx.lineWidth = Math.max(3.2, T * 0.03);
      ctx.beginPath(); ctx.moveTo(0, 0); ctx.lineTo(T, T); ctx.stroke();
      ctx.globalCompositeOperation = "source-over";
      ctx.strokeStyle = C.ink;
      ctx.lineWidth = Math.max(1.5, T * 0.013);
    } else {
      ctx.strokeStyle = C["ink-2"];
      ctx.globalAlpha = 0.5;
      ctx.lineWidth = 0.8;
    }
    ctx.beginPath(); ctx.moveTo(0, 0); ctx.lineTo(T, T); ctx.stroke();
    ctx.globalAlpha = 1;
    if (hover) {
      ctx.strokeStyle = C.ink;
      ctx.lineWidth = 1.25;
      ctx.strokeRect(hover[1] * u, hover[0] * u, u, u);
    }
    ctx.strokeStyle = C.grid;
    ctx.lineWidth = 1;
    ctx.strokeRect(0.5, 0.5, T - 1, T - 1);
  }
  /* Rank legend, drawn from M2G.heat so it always matches the maps. */
  function makeLegend() {
    const wrap = el("div", "pm-legend");
    wrap.append(el("div", "pm-legt", "rank of the answer's first token"));
    const cv = el("canvas", null, null, { role: "img", "aria-label": "colour scale: rank of the answer's first token, from below 5th to 1st" });
    wrap.append(cv);
    return {
      el: wrap,
      draw(w) {
        const h = 29, bar = 10, pad = 10, bw = w - 2 * pad;
        const ctx = M.fitCanvas(cv, w, h);
        ctx.clearRect(0, 0, w, h);
        for (let i = 0; i < bw; i++) { ctx.fillStyle = M.heat(i / (bw - 1)); ctx.fillRect(pad + i, 0, 1.2, bar); }
        ctx.strokeStyle = C.grid;
        ctx.lineWidth = 1;
        ctx.strokeRect(pad + 0.5, 0.5, bw - 1, bar - 1);
        ctx.fillStyle = C["ink-2"];
        ctx.font = font(11.5);
        ctx.textAlign = "center";
        ctx.textBaseline = "alphabetic";
        for (const [v, lab] of M.RANK_TICKS) {
          const x = Math.round(pad + v * (bw - 1));
          ctx.fillRect(x, bar, 1, 4);
          ctx.fillText(lab, x, bar + 17);
        }
      },
    };
  }

  /* ================================================================== #viz-permeation */
  function buildExplorer(host, D) {
    const runs = D.hero.map(prepRun);
    const root = el("div", "pm-root pm-explorer");
    refreshColors(host);
    const first = runs[0].epochs[0];
    // header: one key line (the figcaption is the title), clock, legend
    const head = el("div", "pm-head");
    const htext = el("div", "pm-htext");
    htext.append(el("div", "pm-sub", "Diagonal = no patch; outlined once the model answers on its own."));
    const side = el("div", "pm-hside");
    const clock = el("div", "pm-clock", null, { "aria-hidden": "true" });
    const clockN = el("span");
    clock.append("epoch ", clockN, el("small", null, `/ ${EPOCHS}`));
    const legend = makeLegend();
    side.append(clock, legend.el);
    head.append(htext, side);
    const grid = el("div", "pm-panels", null, {
      tabindex: "0", role: "group",
      "aria-label": "Self-patching maps of two fine-tuning runs. Left and right arrow keys step one epoch.",
    });
    const tip = makeTip(root);

    const panels = runs.map((r) => {
      const ok = r.t_gen != null;
      const p = { run: r, hover: null, hoverE: null, mapKey: "", okState: null };
      p.sec = el("section", "pm-panel", null, { "aria-label": `${ok ? "Run that generalizes" : "Run that never generalizes"}: ${r.head}` });
      const title = el("div", "pm-pt");
      title.append(glyph(ok), ok ? "Generalizes" : "Never generalizes");
      const meta = el("dl", "pm-meta");
      meta.append(el("dt", null, "facts"), el("dd", null, `${r.head} \u2192 ${r.bridge} \u2192 ${r.answer}`),
        el("dt", null, "question"), el("dd", null, r.question));
      p.body = el("div", "pm-body");
      const mw = el("div", "pm-mapwrap");
      p.map = el("canvas", "pm-map", null, { role: "img" });
      p.ring = el("div", "pm-ring");
      mw.append(p.map, p.ring);
      const reads = el("div", "pm-reads");
      const readout = (key, label, big) => {
        const box = el("div", "pm-read");
        const l = el("div", "pm-rl");
        if (key) { const k = el("i", `pm-key pm-key-${key}`); k.style.background = `var(--${key})`; l.append(k); }
        l.append(label);
        const v = el("div", "pm-rv" + (big ? " pm-big" : ""));
        box.append(l, v);
        const s = big ? el("div", "pm-rs") : null;
        if (s) box.append(s);
        reads.append(box);
        return { v, s };
      };
      p.rMem = readout("mem", "facts recalled");
      p.rGen = readout("gen", "two-hop answer, no patch");
      p.gGlyph = el("span");
      p.gGlyph.style.display = "inline-flex";
      p.gText = el("span");
      p.rGen.v.append(p.gGlyph, p.gText);
      p.rCov = readout(null, "layer pairs that rank the answer's first token 1st", true);
      p.note = el("div", "pm-note", null, { "aria-live": "off" });
      reads.append(p.note);
      p.body.append(mw, reads);
      p.chart = el("canvas", "pm-chart", null, { role: "img", "aria-label": `Training log of run ${r.id}: facts recalled and two-hop answer per epoch` });
      p.sec.append(title, meta, p.body, p.chart);
      grid.append(p.sec);
      // statistics quoted by the notes, computed from the data
      const late = r.cov.filter((_, i) => r.epochs[i] >= 18);
      p.stall = late.length ? Math.round(late.reduce((a, b) => a + b, 0) / late.length) : null;
      return p;
    });

    const tlSegs = [[0.6, 0, 0]];
    const tGen = runs[0].t_gen;
    for (let e = 1; e <= EPOCHS; e++) {   // GIF schedule: quick start, slow permeation, pause at generalization, quick tail
      tlSegs.push([e < first ? 0.2 : e === first ? 0.5 : e <= tGen ? 0.8 : 0.25, e - 1, e]);
      if (e === tGen) tlSegs.push([2.5, e, e]);
    }
    tlSegs.push([2.0, EPOCHS, EPOCHS]);
    const tl = timeline(tlSegs);
    const P = makePlayer("permeation", tl, (x) => render(x));
    const ctl = makeControls(P, {
      id: "permeation", min: 0, max: EPOCHS, label: "Fine-tuning epoch",
      valueText: (e) => `epoch ${e} of ${EPOCHS}`, readout: (e) => `epoch ${e} / ${EPOCHS}`,
    });
    P.onState = () => ctl.update(floorE(P.x));
    root.append(head, grid, ctl.el);
    host.querySelectorAll(".m2g-fallback").forEach((f) => f.remove());
    host.append(root);

    /* ---- layout ---- */
    let W = 0;
    function layout(force) {
      const w = root.clientWidth;
      if (!w || (w === W && !force)) return;
      W = w;
      const stack = w < 760;
      root.classList.toggle("pm-narrow", w < 600);
      grid.classList.toggle("pm-stack", stack);
      const PW = stack ? w : Math.floor((w - 40) / 2);
      const beside = PW >= 470;
      const S = beside ? Math.min(330, PW - ML - 2 - 20 - 184) : Math.min(420, PW - ML - 2);
      legend.draw(w < 480 ? 200 : 240);
      for (const p of panels) {
        const r = p.run;
        p.body.classList.toggle("pm-below", !beside);
        p.ml = {
          ox: ML, oy: 6, S, w: ML + S + 2, h: S + 48,
          emptyNote: r.epochs[0] === r.t_mem - 1
            ? [`first scan at epoch ${r.epochs[0]},`, "one epoch before", "the facts are memorized"]
            : [`first scan at epoch ${r.epochs[0]}`],
        };
        Object.assign(p.ring.style, { left: `${ML}px`, top: "6px", width: `${S}px`, height: `${S}px` });
        p.cl = chartLayout(p, PW);
        p.mapKey = "";
      }
      render(P.x);
    }
    function chartLayout(p, w) {
      const r = p.run, pl = ML, pr = w - 10, X = (e) => pl + (e / EPOCHS) * (pr - pl);
      const end = r.t_gen != null ? r.t_gen : EPOCHS;
      const ctx = p.chart.getContext("2d");
      ctx.font = font(12);
      const mk = (a, b, lines) => ({ e0: a, e1: b, a: X(a), b: X(b), cx: X((a + b) / 2), lines });
      const labs = [mk(0, r.t_mem, ["memorizing"]),
        mk(r.t_mem, end, [r.t_gen != null ? "memorized, not yet used" : "memorized, never used"])];
      if (r.t_gen != null) labs.push(mk(r.t_gen, EPOCHS, ["generalized"]));
      const place = () => {      // final positions (band fully revealed); drawChart slides each label towards it
        let prevR = -Infinity, ok = true;
        for (const l of labs) {
          const tw = (l.tw = Math.max(...l.lines.map((s) => ctx.measureText(s).width)));
          l.lo = pl - 30 + tw / 2;
          l.x = M.clamp(l.cx, l.lo, w - tw / 2);
          if (l.x - tw / 2 < prevR + 8) ok = false;
          prevR = l.x + tw / 2;
        }
        return ok;
      };
      let two = false;
      if (!place()) { two = true; labs[1].lines = labs[1].lines[0].split(", ").map((s, i, a) => (i < a.length - 1 ? s + "," : s)); place(); }
      const pt = two ? 38 : 24, ph = w < 420 ? 76 : 86;
      return { w, h: pt + ph + 42, pl, pr, pt, ph, X, labs };
    }

    /* ---- drawing ---- */
    function drawChart(p, x) {
      const L = p.cl, r = p.run, ctx = M.fitCanvas(p.chart, L.w, L.h);
      const X = L.X, Y = (v) => L.pt + ((1.1 - v) / 1.2) * L.ph;
      const e = floorE(x), end = r.t_gen != null ? r.t_gen : EPOCHS;
      ctx.clearRect(0, 0, L.w, L.h);
      if (x > r.t_mem) { ctx.fillStyle = C.band; ctx.fillRect(X(r.t_mem), L.pt, X(Math.min(x, end)) - X(r.t_mem), L.ph); }
      ctx.fillStyle = C.grid;
      for (const v of [0, 0.5, 1]) ctx.fillRect(L.pl, Math.round(Y(v)), L.pr - L.pl, 1);
      ctx.fillStyle = C["ink-2"];
      ctx.font = font(12);
      ctx.textAlign = "center";
      ctx.textBaseline = "alphabetic";
      /* Phase labels follow the reveal: each is centred on the part of its phase left of the playhead (never past its
         final spot or into the previous label) and fades in as that part grows, so no label runs ahead of the data. */
      let prevR = -Infinity;
      for (const l of L.labs) {
        if (x <= l.e0 + EPS) continue;
        const done = x >= l.e1 - EPS, rx = done ? l.b : X(x);
        const need = Math.min(l.tw, l.b - l.a), rw = rx - l.a;
        const alpha = done ? 1 : M.clamp((rw - 0.5 * need) / (0.5 * need), 0, 1);
        if (alpha <= 0) continue;
        const lx = done ? l.x : Math.min(l.x, Math.max((l.a + rx) / 2, prevR + 14 + l.tw / 2, l.lo));
        prevR = lx + l.tw / 2;
        ctx.globalAlpha = alpha;
        l.lines.forEach((s, k) => ctx.fillText(s, lx, L.pt - 7 - (l.lines.length - 1 - k) * 14));
      }
      ctx.globalAlpha = 1;
      ctx.fillStyle = C.muted;
      ctx.fillRect(Math.round(X(x)), L.pt - 3, 1, L.ph + 6);
      const step = (vals, lw, col) => {
        ctx.beginPath();
        ctx.moveTo(X(0), Y(vals[0]));
        for (let i = 1; i <= e; i++) { ctx.lineTo(X(i), Y(vals[i - 1])); ctx.lineTo(X(i), Y(vals[i])); }
        ctx.lineTo(X(x), Y(vals[e]));
        ctx.lineWidth = lw;
        ctx.strokeStyle = col;
        ctx.lineJoin = "round";
        ctx.lineCap = "round";
        ctx.stroke();
      };
      step(r.gen, 3, C.gen);
      step(r.mem, 1.75, C.mem);
      const dot = (cx, cy, rad, col) => {
        ctx.globalCompositeOperation = "destination-out";
        ctx.beginPath(); ctx.arc(cx, cy, rad + 1.75, 0, 7); ctx.fill();
        ctx.globalCompositeOperation = "source-over";
        ctx.fillStyle = col;
        ctx.beginPath(); ctx.arc(cx, cy, rad, 0, 7); ctx.fill();
      };
      if (p.hoverE != null) {
        const hx = Math.round(X(p.hoverE)) + 0.5;
        ctx.strokeStyle = C["ink-2"];
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 3]);
        ctx.beginPath(); ctx.moveTo(hx, L.pt - 3); ctx.lineTo(hx, L.pt + L.ph + 3); ctx.stroke();
        ctx.setLineDash([]);
        dot(X(p.hoverE), Y(r.gen[p.hoverE]), 4, C.gen);
        dot(X(p.hoverE), Y(r.mem[p.hoverE]), 3, C.mem);
      }
      dot(X(x), Y(r.gen[e]), 4.5, C.gen);
      dot(X(x), Y(r.mem[e]), 3.25, C.mem);
      ctx.fillStyle = C["ink-2"];
      ctx.font = font(11);
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      for (let t = 0; t <= EPOCHS; t += 5) ctx.fillText(String(t), X(t), L.pt + L.ph + 6);
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      ctx.fillText("0", L.pl - 8, Y(0));
      ctx.fillText("1", L.pl - 8, Y(1));
      ctx.font = font(12);
      ctx.textAlign = "center";
      ctx.textBaseline = "alphabetic";
      ctx.fillText("fine-tuning epoch", (L.pl + L.pr) / 2, L.pt + L.ph + 36);
      ctx.save();
      ctx.translate(11, L.pt + L.ph / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textBaseline = "middle";
      ctx.fillText("accuracy", 0, 0);
      ctx.restore();
    }
    function renderPanel(p, x, e) {
      const r = p.run, fr = frameAt(r, x), ok = answers(r, e);
      const key = `${fr.i}|${fr.w.toFixed(4)}|${ok}|${p.hover}|${p.ml.S}|${C.dark}`;
      if (key !== p.mapKey) {
        p.mapKey = key;
        drawBigMap(M.fitCanvas(p.map, p.ml.w, p.ml.h), r, fr, ok, p.ml, p.hover);
      }
      p.fr = fr;
      setText(p.rMem.v, `${Math.round(2 * r.mem[e])} / 2`);
      if (p.okState !== ok) {
        p.okState = ok;
        p.gGlyph.className = ok ? "pm-good" : "pm-bad";
        p.gGlyph.innerHTML = ok ? ICON.ok : ICON.bad;
        setText(p.gText, ok ? "correct" : "wrong");
      }
      if (fr.i < 0) { setText(p.rCov.v, "\u2013"); setText(p.rCov.s, "not scanned yet"); } else {
        setText(p.rCov.v, fmtPct(r.cov[fr.i]));
        setText(p.rCov.s, `map of epoch ${r.epochs[fr.i]}`);
      }
      let note = "";
      if (r.t_gen != null && ok) note = `epoch ${r.t_gen}: diagonal reached, answers with no patch`;
      else if (r.t_gen == null && x >= 18 - EPS && p.stall != null) note = `stalled at ~${p.stall}%; never reaches the diagonal`;
      setText(p.note, note);
      p.map.setAttribute("aria-label", fr.i < 0 ? "Self-patching map: not scanned yet"
        : `Self-patching map of epoch ${r.epochs[fr.i]}: ${fmtPct(r.cov[fr.i])} of layer pairs rank the answer's first token 1st; two-hop answer ${ok ? "correct" : "wrong"} with no patch`);
      drawChart(p, x);
    }
    function render(x) {
      x = snap(x);
      const e = floorE(x);
      setText(clockN, String(e));
      for (const p of panels) renderPanel(p, x, e);
      ctl.update(e);
      if (tip.owner) tip.owner();
    }

    /* ---- interaction ---- */
    const jump = (e) => P.user(() => P.setX(e));
    for (const p of panels) {
      const r = p.run;
      const mapHover = (ev) => {
        const b = p.map.getBoundingClientRect();
        const t = Math.floor(((ev.clientX - b.left - p.ml.ox) / p.ml.S) * r.n);
        const s = Math.floor(((ev.clientY - b.top - p.ml.oy) / p.ml.S) * r.n);
        const inside = s >= 0 && t >= 0 && s < r.n && t < r.n;
        p.hover = inside ? [s, t] : null;
        if (inside) {
          const show = () => tip.show(cellTip(r, p.fr, s, t), ev.clientX, ev.clientY);
          tip.owner = show;
          show();
        } else if (tip.owner) tip.hide();
        p.mapKey = "";
        renderPanel(p, snap(P.x), floorE(P.x));
      };
      p.map.addEventListener("pointermove", mapHover);
      p.map.addEventListener("pointerdown", mapHover);
      p.map.addEventListener("pointerleave", (ev) => {
        if (ev.pointerType === "touch") return;
        p.hover = null; tip.hide(); p.mapKey = ""; renderPanel(p, snap(P.x), floorE(P.x));
      });
      const chartEpoch = (ev) => {
        const b = p.chart.getBoundingClientRect(), px = ev.clientX - b.left;
        if (px < p.cl.pl - 12 || px > p.cl.pr + 12) return null;
        return M.clamp(Math.round(((px - p.cl.pl) / (p.cl.pr - p.cl.pl)) * EPOCHS), 0, EPOCHS);
      };
      const chartHover = (ev) => {
        const ep = chartEpoch(ev);
        p.hoverE = ep;
        if (ep == null) { if (tip.owner) tip.hide(); } else {
          const show = () => {
            const ok = answers(r, ep);
            const g = el("span", "tip-v"); g.append(glyph(ok), ` ${ok ? "correct" : "wrong"}`);
            g.style.display = "inline-flex"; g.style.alignItems = "center"; g.style.gap = "2px";
            tip.show([el("div", "tip-h", `epoch ${ep}`), tipRow(C.mem, `${Math.round(2 * r.mem[ep])}/2`, "facts recalled"),
              tipRow(C.gen, g, "two-hop answer"), el("div", "tip-f", "click to show this epoch")], ev.clientX, ev.clientY);
          };
          tip.owner = show;
          show();
        }
        drawChart(p, snap(P.x));
      };
      p.chart.addEventListener("pointermove", chartHover);
      p.chart.addEventListener("pointerleave", (ev) => {
        if (ev.pointerType === "touch") return;
        p.hoverE = null; tip.hide(); drawChart(p, snap(P.x));
      });
      p.chart.addEventListener("click", (ev) => { const ep = chartEpoch(ev); if (ep != null) jump(ep); });
    }
    document.addEventListener("pointerdown", (ev) => {
      if (!root.contains(ev.target) || !(ev.target instanceof HTMLCanvasElement)) {
        if (tip.owner) tip.hide();
        let dirty = false;
        for (const p of panels) if (p.hover || p.hoverE != null) { p.hover = null; p.hoverE = null; p.mapKey = ""; dirty = true; }
        if (dirty) render(P.x);
      }
    });
    root.addEventListener("keydown", (ev) => {
      if (ev.target.tagName === "INPUT") return;
      const d = ev.key === "ArrowRight" ? 1 : ev.key === "ArrowLeft" ? -1 : 0;
      if (d) { ev.preventDefault(); jump(floorE(P.x) + d); }
      else if (ev.key === "Home" || ev.key === "End") { ev.preventDefault(); jump(ev.key === "Home" ? 0 : EPOCHS); }
    });
    /* Hand-off from the self-patching explainer: land on the map the reader just saw (hero[0] at epoch 14 = its demo grid). */
    document.addEventListener("m2g:handoff", (ev) => {
      const d = ev.detail || {};
      if (typeof d.epoch !== "number") return;
      jump(d.epoch);
      const p = panels.find((q) => q.run.id === d.run);
      if (p) { p.ring.classList.remove("pm-flash"); void p.ring.offsetWidth; p.ring.classList.add("pm-flash"); }
    });
    const retheme = () => { refreshColors(host); W = 0; layout(true); };
    M.onTheme(() => { retheme(); setTimeout(retheme, 400); });   // second pass after any background transition
    if ("ResizeObserver" in window) new ResizeObserver(() => layout(false)).observe(root);
    else window.addEventListener("resize", () => layout(false));
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => { refreshColors(host); layout(true); });

    P.x = M.reducedMotion ? 16 : 0;
    P.time = tl.t(P.x);
    layout(true);
    return P;
  }

  /* ================================================================== #viz-mosaic */
  function buildMosaic(host, D) {
    const succ = D.mosaic.success.map(prepRun), fail = D.mosaic.failure.map(prepRun);
    const all = succ.concat(fail);
    const k1 = EPOCHS - Math.max(...all.map((r) => r.t_mem));             // no run goes past its last trained epoch
    const slowUntil = Math.max(...succ.map((r) => r.t_gen - r.t_mem)) + 1;  // slow while runs start answering
    const root = el("div", "pm-root pm-mosaic");
    refreshColors(host);
    const head = el("div", "pm-head");
    const htext = el("div", "pm-htext");
    const sub = el("div", "pm-sub");     // one key line; the figcaption is the title
    const okg = glyph(true);
    okg.style.verticalAlign = "-2px";
    sub.append(okg, " = answers the two-hop question with no patch (diagonal outlined).");
    htext.append(sub);
    const side = el("div", "pm-hside");
    const clock = el("div", "pm-clock", null, { "aria-hidden": "true" });
    const clockN = el("span");
    clock.append(clockN, el("small", null, "epochs since memorization"));
    const legend = makeLegend();
    side.append(clock, legend.el);
    head.append(htext, side);
    const groups = el("div", "pm-groups");
    const tip = makeTip(root);
    const tiles = [];
    const mkGroup = (runs, name) => {
      const g = el("section", "pm-group", null, { "aria-label": `${runs.length} runs that ${name.toLowerCase()}` });
      const t = el("div", "pm-gt", name);
      t.append(el("small", null, `${runs.length} runs`));
      const box = el("div", "pm-tiles");
      for (const r of runs) {
        const info = r.t_gen != null ? `memorized at epoch ${r.t_mem}; answers from epoch ${r.t_gen}` : `memorized at epoch ${r.t_mem}; never answers`;
        const w = el("div", "pm-tile", null, { role: "img", "aria-label": `Run that ${r.t_gen != null ? "generalizes" : "never generalizes"}: ${info}` });
        const cv = el("canvas");
        const badge = el("span", "pm-badge", null, { "aria-hidden": "true" });
        badge.innerHTML = ICON.ok;
        w.append(cv, badge);
        box.append(w);
        tiles.push({ run: r, cv, badge, info, hover: null, key: "", on: null });
      }
      g.append(t, box);
      return g;
    };
    groups.append(mkGroup(succ, "Generalize"), mkGroup(fail, "Never generalize"));

    const segs = [[1.0, -1, -1]];
    for (let k = 0; k <= k1; k++) segs.push([k <= slowUntil ? 0.7 : 0.3, k - 1, k]);
    segs.push([2.5, k1, k1]);
    const tl = timeline(segs);
    const P = makePlayer("mosaic", tl, (k) => render(k));
    const ctl = makeControls(P, {
      id: "mosaic", min: -1, max: k1, label: "Epochs since memorization",
      valueText: (k) => `${fmtK(k)} epochs since memorization`, readout: (k) => `${fmtK(k)} epochs`,
    });
    P.onState = () => ctl.update(floorE(P.x));
    root.append(head, groups, ctl.el);
    host.querySelectorAll(".m2g-fallback").forEach((f) => f.remove());
    host.append(root);

    let W = 0, T = 0;
    function layout(force) {
      const w = root.clientWidth;
      if (!w || (w === W && !force)) return;
      W = w;
      const stack = w < 700, gap = w < 500 ? 6 : 10;
      root.classList.toggle("pm-narrow", w < 600);
      groups.classList.toggle("pm-stack", stack);
      const G = stack ? w : Math.floor((w - 48) / 2);
      T = Math.min(stack ? 150 : 170, Math.floor((G - 3 * gap) / 4));
      root.classList.toggle("pm-small", T < 100);
      root.querySelectorAll(".pm-tiles").forEach((b) => {
        b.style.gridTemplateColumns = `repeat(4, ${T}px)`;
        b.style.gap = `${gap}px`;
      });
      legend.draw(w < 480 ? 200 : 240);
      for (const t of tiles) t.key = "";
      render(P.x);
    }
    function render(k) {
      k = snap(k);
      const kk = floorE(k);
      setText(clockN, fmtK(kk));
      for (const t of tiles) {
        const r = t.run, x = r.t_mem + k, fr = frameAt(r, x), ok = answers(r, r.t_mem + kk);
        t.fr = fr;
        const key = `${fr.i}|${fr.w.toFixed(4)}|${ok}|${t.hover}|${T}|${C.dark}`;
        if (key !== t.key) { t.key = key; drawTile(M.fitCanvas(t.cv, T, T), r, fr, ok, T, t.hover); }
        if (t.on !== ok) { t.on = ok; t.badge.classList.toggle("pm-on", ok); }
      }
      ctl.update(kk);
      if (tip.owner) tip.owner();
    }
    for (const t of tiles) {
      const r = t.run;
      const hover = (ev) => {
        const b = t.cv.getBoundingClientRect();
        const c = Math.floor(((ev.clientX - b.left) / T) * r.n), s = Math.floor(((ev.clientY - b.top) / T) * r.n);
        t.hover = s >= 0 && c >= 0 && s < r.n && c < r.n ? [s, c] : null;
        if (t.hover) {
          const show = () => tip.show(cellTip(r, t.fr, s, c, t.info), ev.clientX, ev.clientY);
          tip.owner = show;
          show();
        }
        render(P.x);
      };
      t.cv.addEventListener("pointermove", hover);
      t.cv.addEventListener("pointerdown", hover);
      t.cv.addEventListener("pointerleave", (ev) => {
        if (ev.pointerType === "touch") return;
        t.hover = null; tip.hide(); render(P.x);
      });
    }
    document.addEventListener("pointerdown", (ev) => {
      if (root.contains(ev.target) && ev.target instanceof HTMLCanvasElement) return;
      if (tip.owner) tip.hide();
      if (tiles.some((t) => t.hover)) { tiles.forEach((t) => { t.hover = null; }); render(P.x); }
    });
    root.addEventListener("keydown", (ev) => {
      if (ev.target.tagName === "INPUT") return;
      const d = ev.key === "ArrowRight" ? 1 : ev.key === "ArrowLeft" ? -1 : 0;
      if (d) { ev.preventDefault(); P.user(() => P.setX(floorE(P.x) + d)); }
    });
    groups.setAttribute("tabindex", "0");
    groups.setAttribute("aria-label", "Sixteen self-patching maps. Left and right arrow keys step one epoch.");
    const retheme = () => { refreshColors(host); layout(true); };
    M.onTheme(() => { retheme(); setTimeout(retheme, 400); });
    if ("ResizeObserver" in window) new ResizeObserver(() => layout(false)).observe(root);
    else window.addEventListener("resize", () => layout(false));
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => { refreshColors(host); layout(true); });

    P.x = M.reducedMotion ? k1 : -1;
    P.time = tl.t(P.x);
    layout(true);
    return P;
  }

  /* ------------------------------------------------------------------ registration */
  injectStyle();
  const component = (build) => {
    let P = null;
    return {
      mount(el, data) { P = build(el, data.permeation); },
      seek(t) { if (P) P.seekT(t); },
      play() { if (P && !P.playing) P.play(); },
      pause() { if (P && P.playing) P.pause(); },
    };
  };
  M.register("permeation", component(buildExplorer));
  M.register("mosaic", component(buildMosaic));
})();
