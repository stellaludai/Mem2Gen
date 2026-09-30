/* Shared helpers for the Mem2Gen project page.
 *
 * Components register themselves:  M2G.register('lrsd', { mount(el, data), seek(t), play(), pause() })
 * and are mounted into <div id="viz-lrsd"> on DOMContentLoaded (data = window.M2G_DATA). A component plays while at least
 * 30% of it is on screen and pauses when it leaves (never autoplays under prefers-reduced-motion).
 * QA hook: window.__seek('lrsd', 0.6) pauses the component and renders normalized storyboard time t in [0, 1].
 */
(function () {
  "use strict";

  const registry = new Map();     // id -> component
  const mounted = new Map();      // id -> {el, comp}
  const themeCbs = [];

  const M2G = {
    registry,
    reducedMotion: window.matchMedia("(prefers-reduced-motion: reduce)").matches,

    register(id, comp) {
      registry.set(id, comp);
      if (document.readyState !== "loading") mountOne(id);
    },

    /* ---------- colours ---------- */
    css(name, el) {
      return getComputedStyle(el || document.documentElement).getPropertyValue(name).trim();
    },
    isDark() {
      const t = document.documentElement.getAttribute("data-theme");
      if (t === "dark") return true;
      if (t === "light") return false;
      return window.matchMedia("(prefers-color-scheme: dark)").matches;
    },
    onTheme(cb) { themeCbs.push(cb); },
    setTheme(name) {            // 'light' | 'dark' | null (follow the system)
      if (name) document.documentElement.setAttribute("data-theme", name);
      else document.documentElement.removeAttribute("data-theme");
      try { name ? localStorage.setItem("m2g-theme", name) : localStorage.removeItem("m2g-theme"); } catch (e) { /* storage off */ }
      fireTheme();
    },
    /* Heat map colour of a reciprocal-rank value v in [0, 1], following the active theme:
       light = the paper's map #6095CE -> #F3F3F3 -> #DB766D, dark = the amber "heat" ramp. */
    heatRGB(v) {
      const lut = M2G.isDark() ? LUT_DARK : LUT_LIGHT;
      const k = Math.max(0, Math.min(255, Math.round(v * 255)));
      return lut[k];
    },
    heat(v) {
      const c = M2G.heatRGB(v);
      return `rgb(${c[0]},${c[1]},${c[2]})`;
    },
    RANK_TICKS: [[0.2, "5th"], [1 / 3, "3rd"], [0.5, "2nd"], [1, "1st"]],
    FULL: 0.99,   // decoded value >= 0.99 <=> the answer's first token is ranked 1st

    /* ---------- data ---------- */
    decodeGrid(b64, n) {
      const bin = atob(b64);
      const out = new Float32Array(n * n);
      for (let i = 0; i < out.length; i++) out[i] = bin.charCodeAt(i) / 255;
      return out;
    },
    ordinal(k) {
      const s = ["th", "st", "nd", "rd"], v = k % 100;
      return k + (s[(v - 20) % 10] || s[v] || s[0]);
    },
    /* rank of the answer's first token from a reciprocal-rank value (null if below 5th) */
    rankOf(v) { return v >= 0.19 ? Math.round(1 / v) : null; },

    /* ---------- maths / motion ---------- */
    clamp: (x, a, b) => Math.max(a, Math.min(b, x)),
    lerp: (a, b, t) => a + (b - a) * t,
    ease: (t) => t * t * (3 - 2 * t),                                   // smoothstep
    easeInOut: (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2),
    lateFade(w) { return M2G.ease(M2G.clamp((w - 2 / 3) * 3, 0, 1)); },   // cross-fade only in the last third

    /* Canvas sized to its CSS box at devicePixelRatio; returns the 2d context scaled to CSS pixels. */
    fitCanvas(canvas, w, h) {
      const dpr = Math.min(window.devicePixelRatio || 1, 3);
      canvas.style.width = w + "px";
      canvas.style.height = h + "px";
      if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
        canvas.width = Math.round(w * dpr);
        canvas.height = Math.round(h * dpr);
      }
      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      return ctx;
    },

    /* A small rAF clock: calls tick(dtSeconds) while running. */
    clock(tick) {
      let raf = 0, last = 0, running = false;
      const loop = (ts) => {
        if (!running) return;
        const dt = last ? Math.min(0.1, (ts - last) / 1000) : 0;
        last = ts;
        tick(dt);
        raf = requestAnimationFrame(loop);
      };
      return {
        start() { if (!running) { running = true; last = 0; raf = requestAnimationFrame(loop); } },
        stop() { running = false; cancelAnimationFrame(raf); },
        get running() { return running; },
      };
    },
  };

  /* ---------- colour look-up tables ---------- */
  function hexRGB(h) { return [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16)); }
  function buildLight() {
    const a = hexRGB("#6095CE"), m = hexRGB("#F3F3F3"), b = hexRGB("#DB766D");
    const lut = [];
    for (let k = 0; k < 256; k++) {
      const x = k / 255;
      const [p, q, w] = x <= 0.5 ? [a, m, x / 0.5] : [m, b, (x - 0.5) / 0.5];
      lut.push(p.map((c, i) => Math.round(c + (q[i] - c) * w)));
    }
    return lut;
  }
  function oklchRGB(L, C, hDeg) {
    const h = (hDeg * Math.PI) / 180, A = C * Math.cos(h), B = C * Math.sin(h);
    const l_ = L + 0.3963377774 * A + 0.2158037573 * B;
    const m_ = L - 0.1055613458 * A - 0.0638541728 * B;
    const s_ = L - 0.0894841775 * A - 1.291485548 * B;
    const l = l_ ** 3, m = m_ ** 3, s = s_ ** 3;
    const lin = [4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
      -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
      -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s];
    return lin.map((c) => {
      c = Math.min(1, Math.max(0, c));
      const g = c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
      return Math.round(g * 255);
    });
  }
  function buildDark() {
    const lut = [];
    for (let k = 0; k < 256; k++) {
      const x = k / 255;
      lut.push(oklchRGB(0.27 + 0.63 * Math.pow(x, 1.8), 0.015 + 0.165 * Math.pow(x, 1.1), 35 + 40 * x));
    }
    return lut;
  }
  const LUT_LIGHT = buildLight();
  const LUT_DARK = buildDark();

  /* ---------- theme ---------- */
  function fireTheme() { themeCbs.forEach((cb) => { try { cb(M2G.isDark()); } catch (e) { console.error(e); } }); }
  try {
    const saved = localStorage.getItem("m2g-theme");
    if (saved === "light" || saved === "dark") document.documentElement.setAttribute("data-theme", saved);
  } catch (e) { /* storage off */ }
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", fireTheme);

  /* ---------- mounting / visibility ---------- */
  let io = null;
  function observer() {
    if (io || !("IntersectionObserver" in window)) return io;
    io = new IntersectionObserver((entries) => {
      entries.forEach((en) => {
        const id = en.target.getAttribute("data-m2g");
        const m = mounted.get(id);
        if (!m || m.frozen) return;
        if (en.isIntersecting && en.intersectionRatio >= 0.3) {
          if (!M2G.reducedMotion && !m.userPaused && m.comp.play) m.comp.play();
        } else if (m.comp.pause) {
          m.comp.pause();
        }
      });
    }, { threshold: [0, 0.3, 0.6] });
    return io;
  }
  function mountOne(id) {
    if (mounted.has(id)) return;
    const el = document.getElementById("viz-" + id);
    const comp = registry.get(id);
    if (!el || !comp) return;
    try {
      comp.mount(el, window.M2G_DATA || {});
      el.setAttribute("data-m2g", id);
      mounted.set(id, { el, comp, userPaused: false, frozen: false });
      const o = observer();
      if (o) o.observe(el);
    } catch (e) {
      console.error("M2G: mounting " + id + " failed", e);
      el.insertAdjacentHTML("beforeend", '<p class="viz-error">This figure could not be drawn in your browser.</p>');
    }
  }
  /* Components call this when the user presses pause / play, so scrolling does not override the choice. */
  M2G.userPaused = function (id, paused) { const m = mounted.get(id); if (m) m.userPaused = paused; };

  document.addEventListener("DOMContentLoaded", () => {
    registry.forEach((_, id) => mountOne(id));
    document.querySelectorAll("[data-theme-toggle]").forEach((b) => {
      b.addEventListener("click", () => M2G.setTheme(M2G.isDark() ? "light" : "dark"));
    });
  });

  /* QA: render component `id` at normalized time t (0..1), paused and frozen (scrolling will not restart it). */
  window.__seek = function (id, t) {
    const m = mounted.get(id);
    if (!m) return "no component " + id;
    m.frozen = true;
    if (m.comp.pause) m.comp.pause();
    m.comp.seek(t);
    return "ok";
  };

  window.M2G = M2G;
})();
