/* Results charts for the Mem2Gen page, all drawn from window.M2G_DATA:
 *   'gap'    -> #viz-gap     Figure 2, the Knowing–Using Gap per model (results.gap)
 *   'oracle' -> #viz-oracle  Figure 7, chaining without vs with oracle self-patching (results.oracle)
 *   Table 1  -> #table-controls  (results.controls)
 *   Table 2  -> #table-lrsd      (lrsd.summary + lrsd.models)
 * Charts are hand-built SVG; colours come from tokens.css through the classes in style.css, so a theme switch needs no repaint.
 * Each chart plays once (dots travel to their values), seek(t) renders progress t in [0, 1] deterministically.
 */
(function () {
  "use strict";

  const SVGNS = "http://www.w3.org/2000/svg";

  /* ---------------------------------------------------------------- helpers */
  function s(tag, attrs, parent, text) {
    const e = document.createElementNS(SVGNS, tag);
    if (attrs) for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }
  function h(tag, cls, parent, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }
  const clamp01 = (x) => Math.max(0, Math.min(1, x));
  const easeOut = (t) => 1 - Math.pow(1 - t, 3);
  const lerp = (a, b, t) => a + (b - a) * t;
  const f1 = (x) => x.toFixed(1);
  const f2 = (x) => x.toFixed(2);
  const f3 = (x) => x.toFixed(3);
  const reducedMotion = () => (window.M2G ? window.M2G.reducedMotion
    : window.matchMedia("(prefers-reduced-motion: reduce)").matches);
  const rowsAsObjects = (tbl) => tbl.rows.map((r) => {
    const o = {};
    tbl.cols.forEach((c, i) => { o[c] = r[i]; });
    return o;
  });
  const range = (xs, fmt) => {
    const lo = Math.min(...xs), hi = Math.max(...xs);
    return fmt(lo) === fmt(hi) ? fmt(lo) : fmt(lo) + "–" + fmt(hi);
  };
  const tick01 = (v) => (v === 0 ? "0" : v === 1 ? "1" : String(v));

  /* progress clock: p runs 0 -> 1 once over durMs; never loops */
  function makeProgress(durMs, onFrame) {
    let p = 0, raf = 0, last = 0;
    const stop = () => { if (raf) cancelAnimationFrame(raf); raf = 0; };
    const loop = (ts) => {
      if (!last) last = ts;
      p = Math.min(1, p + (ts - last) / durMs);
      last = ts;
      onFrame(p);
      raf = p < 1 ? requestAnimationFrame(loop) : 0;
    };
    return {
      get p() { return p; },
      set(v) { stop(); p = clamp01(v); onFrame(p); },
      play() { if (raf || p >= 1) return; last = 0; raf = requestAnimationFrame(loop); },
      pause: stop,
    };
  }
  /* per-row local progress with a stagger */
  function local(p, i, n, rowMs, staggerMs) {
    const total = rowMs + (n - 1) * staggerMs;
    return clamp01((p * total - i * staggerMs) / rowMs);
  }

  function legend(parent, items) {
    const L = h("div", "m2g-legend", parent);
    items.forEach(([kind, color, text]) => {
      const it = h("span", "lg", L);
      const sw = h("span", "lg-" + kind, it);
      sw.style.setProperty("--k", color);
      sw.setAttribute("aria-hidden", "true");
      h("span", null, it, text);
    });
    return L;
  }

  function makeTip(host) {
    const tip = h("div", "m2g-tip", host);
    tip.hidden = true;
    tip.setAttribute("aria-hidden", "true");   // rows carry the same text as aria-label
    return {
      show(build, x, y) {
        tip.textContent = "";
        build(tip);
        tip.hidden = false;
        const W = host.clientWidth, tw = tip.offsetWidth, th = tip.offsetHeight;
        let left = x + 18;
        if (left + tw > W) left = x - 18 - tw;
        if (left < 0) left = Math.max(0, Math.min(W - tw, x - tw / 2));
        const top = Math.max(-8, y - th / 2);
        tip.style.left = Math.round(left) + "px";
        tip.style.top = Math.round(top) + "px";
      },
      hide() { tip.hidden = true; },
    };
  }
  function tipHead(tip, main, sub) {
    const d = h("div", "tip-h", tip, main);
    if (sub) { d.appendChild(document.createTextNode(" ")); h("small", null, d, sub); }
  }
  function tipRow(tip, color, value, label) {
    const r = h("div", "tip-row", tip);
    const k = h("span", "tip-key" + (color ? "" : " none"), r);
    if (color) k.style.setProperty("--k", color);
    h("span", "tip-v", r, value);
    h("span", "tip-l", r, label);
  }

  /* table from a header spec and rows of cells: cell = string | {t, cls, html} */
  function table(parent, head, body, opts) {
    const wrap = h("div", "table-wrap", parent);
    const t = h("table", "tbl", wrap);
    if (opts && opts.caption) h("caption", "visually-hidden", t, opts.caption);
    const thead = h("thead", null, t);
    head.forEach((hr) => {
      const tr = h("tr", null, thead);
      hr.forEach((c) => {
        const th = h("th", c.cls || null, tr);
        th.setAttribute("scope", "col");
        if (c.colspan) th.colSpan = c.colspan;
        if (c.html) th.innerHTML = c.html; else th.textContent = c.t;   // headers are page constants
      });
    });
    const tb = h("tbody", null, t);
    body.forEach((row) => {
      const tr = h("tr", row.cls || null, tb);
      row.cells.forEach((c, j) => {
        const td = h(j === 0 && !row.cls ? "th" : "td", null, tr);
        if (j === 0 && !row.cls) td.setAttribute("scope", "row");
        if (typeof c === "string") td.textContent = c;
        else {
          if (c.cls) td.className = c.cls;
          if (c.colspan) td.colSpan = c.colspan;
          if (c.gain) { h("span", "v", td, c.t); h("span", "gain", td, c.gain); }
          else td.textContent = c.t;
          if (c.sub) h("span", "td-sub", td, c.sub);
        }
      });
    });
    return t;
  }

  /* Rows are <g tabindex> groups; one tab stop, arrows move between rows. */
  function roving(groups, onEnter, onLeave) {
    groups.forEach((g, i) => {
      g.setAttribute("tabindex", i === 0 ? "0" : "-1");
      g.addEventListener("keydown", (e) => {
        let j = null;
        if (e.key === "ArrowDown" || e.key === "ArrowRight") j = Math.min(groups.length - 1, i + 1);
        else if (e.key === "ArrowUp" || e.key === "ArrowLeft") j = Math.max(0, i - 1);
        else if (e.key === "Home") j = 0;
        else if (e.key === "End") j = groups.length - 1;
        else if (e.key === "Escape") { onLeave(i); return; }
        if (j === null) return;
        e.preventDefault();
        groups.forEach((x) => x.setAttribute("tabindex", "-1"));
        groups[j].setAttribute("tabindex", "0");
        groups[j].focus();
      });
      g.addEventListener("focus", () => onEnter(i, null));
      g.addEventListener("blur", () => onLeave(i));
      g.addEventListener("pointermove", (e) => onEnter(i, e));
      g.addEventListener("pointerleave", () => onLeave(i));
    });
  }

  /* shared chart skeleton: header (legend), chart host with svg + tooltip, data table, note */
  function frame(el, legendItems, ariaSummary) {
    el.textContent = "";
    const head = h("div", "rs-head", el);
    legend(head, legendItems);
    const chart = h("div", "rs-chart", el);
    const svg = s("svg", { role: "group", "aria-label": ariaSummary }, chart);
    const tip = makeTip(chart);
    return { head, chart, svg, tip };
  }
  function dataTwin(el, build) {
    const d = h("details", "m2g-data", el);
    h("summary", null, d, "Show data table");
    build(d);
    return d;
  }
  function observeWidth(target, cb) {
    let w = -1;
    const run = () => {
      const nw = Math.round(target.clientWidth);
      if (nw !== w && nw > 0) { w = nw; cb(nw); }
    };
    if ("ResizeObserver" in window) new ResizeObserver(run).observe(target);
    else window.addEventListener("resize", run);
    run();
  }
  /* Show the final frame when nothing will ever call play() (reduced motion, no IntersectionObserver). */
  const startsAtEnd = () => reducedMotion() || !("IntersectionObserver" in window);

  /* ================================================================ Figure 2: gap */
  (function () {
    let st = null;
    const ROW_MS = 900, STAGGER = 45;

    function mount(el, data) {
      const R = data.results && data.results.gap;
      if (!R) throw new Error("M2G_DATA.results.gap is missing");
      const rows = rowsAsObjects(R);
      const methods = [...new Set(rows.map((r) => r.method))];
      const nice = (m) => (m === "Full FT" ? "Full fine-tuning" : m);
      const by = (m) => rows.filter((r) => r.method === m);
      const summary = "Chart of the Knowing–Using Gap in per-fact runs. " + methods.map((m) => {
        const g = by(m);
        return nice(m) + ": memorized after " + range(g.map((r) => r.t_mem), f1) + " epochs, first answered the two-hop question after "
          + range(g.map((r) => r.t_gen), f1) + " epochs; final single-hop accuracy " + range(g.map((r) => r.a_mem), f2)
          + ", two-hop " + range(g.map((r) => r.a_gen), f2) + ".";
      }).join(" ");

      const fr = frame(el, [
        ["dot", "var(--mem)", "Memorization (single-hop recall)"],
        ["dot", "var(--gen)", "Generalization (two-hop answer)"],
      ], summary);

      dataTwin(el, (d) => table(d, [[
        { html: "Model <span class=\"th-sub\">fine-tuning</span>" },
        { html: "T<sub>mem</sub>" }, { html: "T<sub>gen</sub>" }, { t: "ΔT" },
        { html: "A<sub>mem</sub>" }, { html: "A<sub>gen</sub>" }, { t: "ΔA" },
      ]], rows.map((r) => ({ cells: [
        { t: r.model, sub: nice(r.method) }, { t: f1(r.t_mem) }, { t: f1(r.t_gen) }, { t: "+" + f1(r.dT) },
        { t: f2(r.a_mem) }, { t: f2(r.a_gen) }, { t: f2(r.dA) },
      ] })), { caption: "Knowing–Using Gap per model (T in epochs, A = accuracy at the final epoch)" }));
      // The setup (R.note: per-fact runs, 30 epochs, STaRK-Prime) is stated in the Figure 2 caption in index.html.

      st = { el, rows, methods, nice, fr, marks: [], anim: null };
      st.anim = makeProgress(ROW_MS + (rows.length - 1) * STAGGER, update);
      if (startsAtEnd()) st.anim.set(1);
      observeWidth(fr.chart, layout);
    }

    function layout(W) {
      const { rows, methods, nice, fr } = st;
      const svg = fr.svg;
      svg.textContent = "";
      fr.tip.hide();
      const narrow = W < 600;
      const rowH = narrow ? 25 : 26, groupH = 24, groupGap = 14;
      const labelW = narrow ? 104 : 132, dW = 46, titleH = 30, axisH = 26, panelGap = narrow ? 34 : 40, pad = 7;

      // vertical positions inside a panel body
      const rowY = [], groupY = [];
      let y = 0;
      methods.forEach((m, gi) => {
        if (gi) y += groupGap;
        groupY.push({ m, y, sep: gi > 0 });
        y += groupH;
        rows.forEach((r, i) => { if (r.method === m) { rowY[i] = y + rowH / 2; y += rowH; } });
      });
      const bodyH = y, panelH = titleH + bodyH + axisH;

      const panels = [
        { title: "Epoch first reached", unit: " · T, in epochs", dHead: "ΔT", dom: [0, 20], ticks: [0, 5, 10, 15, 20], tf: String,
          mem: (r) => r.t_mem, gen: (r) => r.t_gen, d: (r) => "+" + f1(r.dT) },
        { title: "Final-epoch accuracy", unit: " · A", dHead: "ΔA", dom: [0, 1], ticks: [0, 0.25, 0.5, 0.75, 1], tf: tick01,
          mem: (r) => r.a_mem, gen: (r) => r.a_gen, d: (r) => f2(r.dA) },
      ];
      let H;
      if (!narrow) {
        const plotW = (W - labelW - panelGap - 2 * dW) / 2;
        panels[0].box = { x: labelW, y: 0, w: plotW };
        panels[1].box = { x: labelW + plotW + dW + panelGap, y: 0, w: plotW };
        H = panelH;
      } else {
        const plotW = W - labelW - dW;
        panels[0].box = { x: labelW, y: 0, w: plotW };
        panels[1].box = { x: labelW, y: panelH + panelGap, w: plotW };
        H = 2 * panelH + panelGap;
      }
      svg.setAttribute("width", W);
      svg.setAttribute("height", H);
      svg.setAttribute("viewBox", `0 0 ${W} ${H}`);

      const gAxes = s("g", { "aria-hidden": "true" }, svg);
      panels.forEach((P, pi) => {
        const b = P.box, top = b.y + titleH;
        P.top = top;
        P.sx = (v) => b.x + pad + (v - P.dom[0]) / (P.dom[1] - P.dom[0]) * (b.w - 2 * pad);
        const t = s("text", { x: narrow ? 0 : b.x + pad, y: b.y + 14, class: "rs-ptitle" }, gAxes, P.title);
        s("tspan", { class: "u" }, t, P.unit);
        s("text", { x: b.x + b.w + dW, y: b.y + 14, class: "rs-dhead", "text-anchor": "end" }, gAxes, P.dHead);
        P.ticks.forEach((v) => {
          const x = Math.round(P.sx(v)) + 0.5;
          s("line", { x1: x, x2: x, y1: top, y2: top + bodyH, class: "rs-grid" }, gAxes);
          s("text", { x, y: top + bodyH + 17, class: "rs-tick", "text-anchor": "middle" }, gAxes, P.tf(v));
        });
        s("line", { x1: b.x, x2: b.x + b.w, y1: Math.round(top + bodyH) + 0.5, y2: Math.round(top + bodyH) + 0.5, class: "rs-axis" }, gAxes);
        if (pi === 0 || narrow) {
          groupY.forEach((g) => {
            if (g.sep) {
              const yy = Math.round(top + g.y - groupGap / 2) + 0.5;
              s("line", { x1: 0, x2: narrow ? W : W, y1: yy, y2: yy, class: "rs-sep" }, gAxes);
            }
            s("text", { x: 0, y: top + g.y + groupH - 8, class: "rs-group" }, gAxes, nice(g.m));
          });
        }
      });

      // rows
      st.marks = [];
      const groups = [];
      rows.forEach((r, i) => {
        const g = s("g", { class: "rs-row", role: "img",
          "aria-label": `${r.model}, ${nice(r.method)}: memorized after ${f1(r.t_mem)} epochs, first answers the two-hop question after ${f1(r.t_gen)} epochs, a lag of ${f1(r.dT)} epochs. Final accuracy ${f2(r.a_mem)} single-hop, ${f2(r.a_gen)} two-hop, a gap of ${f2(r.dA)}.` }, svg);
        groups.push(g);
        const m = { conns: [], gens: [], dls: [], panels };
        (narrow ? panels : [panels[0]]).forEach((P) => {
          const yy = P.top + rowY[i];
          s("rect", { x: -8, y: yy - rowH / 2, width: W + 16, height: rowH, rx: 5, class: "rs-band" }, g);
          s("text", { x: 0, y: yy + 4.5, class: "rs-label" }, g, r.model);
        });
        panels.forEach((P) => {
          const yy = P.top + rowY[i];
          const xm = P.sx(P.mem(r));
          m.conns.push(s("line", { x1: xm, x2: xm, y1: yy, y2: yy, class: "rs-conn" }, g));
          m.gens.push(s("circle", { cx: xm, cy: yy, r: 4.5, class: "rs-dot rs-dot--gen" }, g));
          s("circle", { cx: xm, cy: yy, r: 4.5, class: "rs-dot rs-dot--mem" }, g);
          m.dls.push(s("text", { x: P.box.x + P.box.w + dW, y: yy + 4, class: "rs-dlabel", "text-anchor": "end" }, g, P.d(r)));
        });
        m.y = panels[0].top + rowY[i];
        st.marks.push(m);
      });

      roving(groups, (i, e) => {
        groups.forEach((x, j) => x.classList.toggle("is-on", j === i));
        const r = rows[i];
        const rect = fr.chart.getBoundingClientRect();
        const x = e ? e.clientX - rect.left : (narrow ? W * 0.45 : panels[1].box.x - panelGap / 2);
        const yTip = e ? e.clientY - rect.top : st.marks[i].y;
        fr.tip.show((tip) => {
          tipHead(tip, r.model, "· " + nice(r.method));
          tipRow(tip, "var(--mem)", f1(r.t_mem), "epochs to memorize");
          tipRow(tip, "var(--gen)", f1(r.t_gen), "epochs to generalize");
          tipRow(tip, null, "+" + f1(r.dT), "epoch lag, ΔT");
          h("div", "tip-sep", tip);
          tipRow(tip, "var(--mem)", f2(r.a_mem), "single-hop accuracy");
          tipRow(tip, "var(--gen)", f2(r.a_gen), "two-hop accuracy");
          tipRow(tip, null, f2(r.dA), "accuracy gap, ΔA");
        }, x, yTip);
      }, (i) => { groups[i].classList.remove("is-on"); fr.tip.hide(); });

      update(st.anim.p);
    }

    function update(p) {
      if (!st || !st.marks.length) return;
      const n = st.rows.length;
      st.rows.forEach((r, i) => {
        const q = local(p, i, n, ROW_MS, STAGGER), e = easeOut(q);
        const m = st.marks[i];
        m.panels.forEach((P, k) => {
          const x = lerp(P.sx(P.mem(r)), P.sx(P.gen(r)), e);
          m.gens[k].setAttribute("cx", x);
          m.conns[k].setAttribute("x2", x);
          m.dls[k].style.opacity = clamp01((q - 0.83) / 0.17);
        });
      });
    }

    const comp = {
      mount,
      seek(t) { if (st) st.anim.set(t); },
      play() { if (st) st.anim.play(); },
      pause() { if (st) st.anim.pause(); },
    };
    if (window.M2G) window.M2G.register("gap", comp);
  })();

  /* ================================================================ Figure 7: oracle */
  (function () {
    let st = null;
    const ROW_MS = 800, STAGGER = 50;
    const DOMAINS = [
      { key: "prime", name: "STaRK-Prime" },
      { key: "mag", name: "STaRK-MAG" },
    ];

    function mount(el, data) {
      const R = data.results && data.results.oracle;
      if (!R) throw new Error("M2G_DATA.results.oracle is missing");
      const rows = rowsAsObjects(R);
      rows.forEach((r) => { r.family = (r.model.match(/^[A-Za-z]+/) || [""])[0]; });
      const ratios = [];
      rows.forEach((r) => DOMAINS.forEach((d) => ratios.push(r[d.key + "_pat"] / r[d.key + "_wo"])));
      const summary = `Chart of two-hop accuracy without patching and with oracle self-patching for ${rows.length} models on STaRK-Prime and STaRK-MAG. `
        + `Oracle patching raises accuracy ${f1(Math.min(...ratios))} to ${f1(Math.max(...ratios))} times in every cell; `
        + `memorization is ${range(rows.flatMap((r) => DOMAINS.map((d) => r[d.key + "_mem"])), f2)}. The oracle is a diagnostic upper bound, not a method.`;

      const fr = frame(el, [
        ["ring", "var(--gen)", "Two-hop, without patching"],
        ["dot", "var(--gen)", "Two-hop, oracle self-patching"],
        ["tick", "var(--mem)", "Memorization (single-hop)"],
      ], summary);
      const pill = h("span", "rs-pill", fr.head);
      pill.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><g fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7.5v.01"/></g></svg>';
      pill.appendChild(document.createTextNode("Diagnostic upper bound, not a method"));

      dataTwin(el, (d) => table(d, [
        [{ t: "" }, { t: "STaRK-Prime", colspan: 4, cls: "grp-h" }, { t: "STaRK-MAG", colspan: 4, cls: "grp-h" }],
        [{ t: "Model" }, { t: "Memorization" }, { t: "Without patching" }, { t: "Oracle patching" }, { t: "Gain" },
          { t: "Memorization" }, { t: "Without patching" }, { t: "Oracle patching" }, { t: "Gain" }],
      ], rows.map((r) => ({ cells: [{ t: r.model }].concat(...DOMAINS.map((d) => [
        { t: f3(r[d.key + "_mem"]) }, { t: f3(r[d.key + "_wo"]) }, { t: f3(r[d.key + "_pat"]) },
        { t: "×" + f1(r[d.key + "_pat"] / r[d.key + "_wo"]) },
      ])) })), { caption: "Two-hop accuracy without and with oracle self-patching" }));
      // The setup (R.note: 1,000 facts, 50 epochs, best pair per question) is stated in the Figure 7 caption in index.html.

      st = { el, rows, fr, marks: [], anim: null };
      st.anim = makeProgress(ROW_MS + (rows.length - 1) * STAGGER, update);
      if (startsAtEnd()) st.anim.set(1);
      observeWidth(fr.chart, layout);
    }

    function layout(W) {
      const { rows, fr } = st;
      const svg = fr.svg;
      svg.textContent = "";
      fr.tip.hide();
      const narrow = W < 600;
      const rowH = narrow ? 27 : 30, famGap = 12;
      const labelW = narrow ? 104 : 116, titleH = 30, axisH = 26, panelGap = narrow ? 30 : 36, pad = 7, padR = 8;

      const rowY = [];
      let y = 6, fam = null, sepY = [];
      rows.forEach((r, i) => {
        if (fam !== null && r.family !== fam) { sepY.push(y + famGap / 2); y += famGap; }
        fam = r.family;
        rowY[i] = y + rowH / 2;
        y += rowH;
      });
      const bodyH = y + 4, panelH = titleH + bodyH + axisH;

      const panels = DOMAINS.map((d) => ({ ...d }));
      let H;
      if (!narrow) {
        const plotW = (W - labelW - panelGap - padR) / 2;
        panels[0].box = { x: labelW, y: 0, w: plotW };
        panels[1].box = { x: labelW + plotW + panelGap, y: 0, w: plotW };
        H = panelH;
      } else {
        const plotW = W - labelW - padR;
        panels[0].box = { x: labelW, y: 0, w: plotW };
        panels[1].box = { x: labelW, y: panelH + panelGap, w: plotW };
        H = 2 * panelH + panelGap;
      }
      svg.setAttribute("width", W);
      svg.setAttribute("height", H);
      svg.setAttribute("viewBox", `0 0 ${W} ${H}`);

      const gAxes = s("g", { "aria-hidden": "true" }, svg);
      panels.forEach((P, pi) => {
        const b = P.box, top = b.y + titleH;
        P.top = top;
        P.sx = (v) => b.x + pad + v * (b.w - 2 * pad);
        const t = s("text", { x: narrow ? 0 : b.x + pad, y: b.y + 14, class: "rs-ptitle" }, gAxes, P.name);
        s("tspan", { class: "u" }, t, " · two-hop accuracy");
        [0, 0.25, 0.5, 0.75, 1].forEach((v) => {
          const x = Math.round(P.sx(v)) + 0.5;
          s("line", { x1: x, x2: x, y1: top, y2: top + bodyH, class: "rs-grid" }, gAxes);
          s("text", { x, y: top + bodyH + 17, class: "rs-tick", "text-anchor": "middle" }, gAxes, tick01(v));
        });
        s("line", { x1: b.x, x2: b.x + b.w, y1: Math.round(top + bodyH) + 0.5, y2: Math.round(top + bodyH) + 0.5, class: "rs-axis" }, gAxes);
        if (pi === 0 || narrow) {
          sepY.forEach((yy) => {
            const Y = Math.round(top + yy) + 0.5;
            s("line", { x1: 0, x2: W, y1: Y, y2: Y, class: "rs-sep" }, gAxes);
          });
        }
      });

      st.marks = [];
      const groups = [];
      rows.forEach((r, i) => {
        const aria = r.model + ": " + DOMAINS.map((d) => `${d.name} two-hop accuracy ${f3(r[d.key + "_wo"])} without patching, ${f3(r[d.key + "_pat"])} with oracle patching (${f1(r[d.key + "_pat"] / r[d.key + "_wo"])} times), memorization ${f3(r[d.key + "_mem"])}`).join("; ") + ".";
        const g = s("g", { class: "rs-row", role: "img", "aria-label": aria }, svg);
        groups.push(g);
        const m = { conns: [], dots: [], dls: [], panels };
        (narrow ? panels : [panels[0]]).forEach((P) => {
          const yy = P.top + rowY[i];
          s("rect", { x: -8, y: yy - rowH / 2, width: W + 16, height: rowH, rx: 5, class: "rs-band" }, g);
          s("text", { x: 0, y: yy + 4.5, class: "rs-label" }, g, r.model);
        });
        panels.forEach((P) => {
          const yy = P.top + rowY[i];
          const xw = P.sx(r[P.key + "_wo"]);
          s("rect", { x: P.sx(r[P.key + "_mem"]) - 1, y: yy - 6, width: 2, height: 12, rx: 1, class: "rs-memtick" }, g);
          m.conns.push(s("line", { x1: xw, x2: xw, y1: yy, y2: yy, class: "rs-conn rs-conn--gen" }, g));
          s("circle", { cx: xw, cy: yy, r: 4.5, class: "rs-ring" }, g);
          m.dots.push(s("circle", { cx: xw, cy: yy, r: 5, class: "rs-dot rs-dot--gen" }, g));
          m.dls.push(s("text", { x: 0, y: yy + 4, class: "rs-dlabel" }, g, "×" + f1(r[P.key + "_pat"] / r[P.key + "_wo"])));
        });
        m.y = panels[0].top + rowY[i];
        st.marks.push(m);
      });

      roving(groups, (i, e) => {
        groups.forEach((x, j) => x.classList.toggle("is-on", j === i));
        const r = rows[i];
        const rect = fr.chart.getBoundingClientRect();
        const x = e ? e.clientX - rect.left : (narrow ? W * 0.5 : panels[1].box.x - panelGap / 2);
        const yTip = e ? e.clientY - rect.top : st.marks[i].y;
        fr.tip.show((tip) => {
          tipHead(tip, r.model);
          DOMAINS.forEach((d, k) => {
            if (k) h("div", "tip-sep", tip);
            h("div", "tip-l", tip, d.name);
            tipRow(tip, "var(--gen)", `${f3(r[d.key + "_wo"])} → ${f3(r[d.key + "_pat"])}`, "two-hop, ×" + f1(r[d.key + "_pat"] / r[d.key + "_wo"]));
            tipRow(tip, "var(--mem)", f3(r[d.key + "_mem"]), "memorization");
          });
          h("div", "tip-f", tip, "Best layer pair per question, answer known: a diagnostic upper bound.");
        }, x, yTip);
      }, (i) => { groups[i].classList.remove("is-on"); fr.tip.hide(); });

      update(st.anim.p);
    }

    function update(p) {
      if (!st || !st.marks.length) return;
      const n = st.rows.length;
      st.rows.forEach((r, i) => {
        const q = local(p, i, n, ROW_MS, STAGGER), e = easeOut(q);
        const m = st.marks[i];
        m.panels.forEach((P, k) => {
          const x = lerp(P.sx(r[P.key + "_wo"]), P.sx(r[P.key + "_pat"]), e);
          m.dots[k].setAttribute("cx", x);
          m.conns[k].setAttribute("x2", x);
          m.dls[k].setAttribute("x", x + 10);
          m.dls[k].style.opacity = clamp01((q - 0.8) / 0.2);
        });
      });
    }

    const comp = {
      mount,
      seek(t) { if (st) st.anim.set(t); },
      play() { if (st) st.anim.play(); },
      pause() { if (st) st.anim.pause(); },
    };
    if (window.M2G) window.M2G.register("oracle", comp);
  })();

  /* ================================================================ Tables 1 and 2 */
  function buildTables() {
    const D = window.M2G_DATA || {};

    const tc = document.getElementById("table-controls");
    if (tc && D.results && D.results.controls) {
      try {
        const rows = rowsAsObjects(D.results.controls);
        const keys = ["wo", "cot", "irrelevant", "self"];
        tc.textContent = "";
        table(tc, [[
          { t: "Model" }, { t: "Without patching" }, { t: "CoT prompting" }, { t: "Irrelevant-fact patch" },
          { t: "Self-patch (oracle)", cls: "hl" },
        ]], rows.map((r) => {
          const best = Math.max(...keys.map((k) => r[k]));
          return { cells: [{ t: r.model }].concat(keys.map((k) => ({
            t: f3(r[k]), cls: [k === "self" ? "hl" : "", r[k] === best ? "best" : ""].join(" ").trim(),
          }))) };
        }), { caption: "Controls on STaRK-Prime" });
        h("p", "tbl-note", tc, "Irrelevant-fact patch = patching in the representation of an unrelated fact.");
      } catch (e) { console.error("M2G: table 1 failed", e); }
    }

    const tl = document.getElementById("table-lrsd");
    if (tl && D.lrsd && D.lrsd.summary && D.lrsd.models) {
      try {
        const arms = ["B", "L01", "L1"];
        const m0 = D.lrsd.models[0], seeds0 = (m0 && m0.arms && m0.arms.B) || [];
        const lastEp = seeds0.length ? seeds0[0].chain.length - 1 : 50, nSeeds = seeds0.length || 3;
        let maxSd = 0;
        const body = D.lrsd.models.map((m) => {
          const S = D.lrsd.summary[m.key];
          if (!S) return null;
          const [src, tgt] = String(m.pair).split(/[-→>]+/).map((x) => x.trim());
          const base = S.B.late;
          const best = Math.max(...arms.map((a) => S[a].late));
          arms.forEach((a) => { maxSd = Math.max(maxSd, S[a].late_sd); });
          const cell = (a) => {
            const v = S[a].late;
            const c = { t: f3(v), cls: v === best ? "best" : "" };
            if (a !== "B") {
              const g = Math.round((v / base - 1) * 100);
              c.gain = (g >= 0 ? "+" : "−") + Math.abs(g) + "%";
            }
            if (a === "L1") c.cls += " hl";
            return c;
          };
          return { cells: [
            { t: m.name, sub: `layers ${src} → ${tgt} of ${m.n_layers}` },
            cell("B"), cell("L01"), cell("L1"),
            { t: range(arms.map((a) => S[a].mem), f3), cls: "mem" },
          ] };
        }).filter(Boolean);
        tl.textContent = "";
        table(tl, [[
          { html: "Model <span class=\"th-sub\">layers <i>l</i><sub>src</sub> → <i>l</i><sub>tgt</sub></span>" },
          // non-breaking space / hyphen: on phones these wrap only between "LRSD" and "λ = 0.1", "Plain" and "fine-tuning"
          { t: "Plain fine\u2011tuning" }, { t: "LRSD λ\u00A0=\u00A00.1" }, { t: "LRSD λ\u00A0=\u00A01", cls: "hl" }, { t: "Memo\u00ADrization" },
        ]], body, { caption: `LRSD two-hop exact match, mean over epochs 30–${lastEp} and ${nSeeds} seeds` });
        h("p", "tbl-note", tl, `Seed standard deviations are at most ${f3(maxSd)}. Percentages are relative to plain fine-tuning of the same model.`);
      } catch (e) { console.error("M2G: table 2 failed", e); }
    }
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", buildTables);
  else buildTables();
})();
