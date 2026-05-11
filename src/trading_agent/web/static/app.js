/* trading-agent dashboard — vanilla JS, no build step.
 *
 * Architecture:
 *   - One `fetchAll()` pulls every endpoint in parallel.
 *   - Per-section `render*()` functions take only what they need.
 *   - Auto-refresh every 10s; chart re-fetches only on ticker change.
 */

const REFRESH_MS = 10_000;
const CHART_DAYS = 90;

const state = {
  severity_filter: 0,
  current_ticker: null,
  chart: null,
  candleSeries: null,
  volumeSeries: null,
};

// ─────────────────────────── helpers

const $ = (id) => document.getElementById(id);

function fmt(n, digits = 2) {
  if (n === null || n === undefined || isNaN(n)) return "—";
  return Number(n).toFixed(digits);
}
function fmtPct(n, digits = 2) {
  if (n === null || n === undefined || isNaN(n)) return "—";
  const v = Number(n);
  return (v >= 0 ? "+" : "") + v.toFixed(digits) + "%";
}
function fmtMoney(n) {
  if (n === null || n === undefined || isNaN(n)) return "—";
  const v = Number(n);
  const sign = v >= 0 ? "+" : "−";
  return `${sign}$${Math.abs(v).toLocaleString("en-US", { minimumFractionDigits: 0, maximumFractionDigits: 0 })}`;
}
function pnlClass(n) {
  if (n === null || n === undefined || isNaN(n)) return "";
  return n > 0 ? "pnl-pos" : n < 0 ? "pnl-neg" : "";
}
function ago(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  const s = Math.floor((Date.now() - d.getTime()) / 1000);
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}
function timeHM(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  const hh = String(d.getUTCHours()).padStart(2, "0");
  const mm = String(d.getUTCMinutes()).padStart(2, "0");
  return `${hh}:${mm}`;
}
function escapeHtml(s) {
  return (s || "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );
}
async function fetchJSON(url) {
  try {
    const r = await fetch(url, { cache: "no-store" });
    if (!r.ok) throw new Error(r.statusText);
    return await r.json();
  } catch (e) {
    console.warn("fetch failed", url, e);
    return null;
  }
}

// ─────────────────────────── status banner

function renderStatus(s) {
  if (!s) return;
  const soak = $("status-soak");
  const regime = $("status-regime");
  const halt = $("status-halt");
  if (soak) {
    soak.querySelector("strong").textContent = s.soak_phase || "—";
    soak.classList.remove("alive", "warn", "crit");
    if (s.soak_phase === "autonomous") soak.classList.add("alive");
    if (s.soak_phase === "read_only") soak.classList.add("warn");
  }
  if (regime) {
    const conf = s.regime_confidence ? ` · ${Number(s.regime_confidence).toFixed(2)}` : "";
    regime.querySelector("strong").textContent = (s.regime_label || "—") + conf;
    regime.classList.remove("alive", "warn", "crit");
    if (s.regime_label === "CRISIS") regime.classList.add("crit");
    else if (s.regime_label === "BULL_TREND" || s.regime_label === "RANGE_LOW_VOL") regime.classList.add("alive");
    else if (s.regime_label === "VOLATILE_TRANSITION") regime.classList.add("warn");
  }
  if (halt) {
    halt.querySelector("strong").textContent = s.halted ? "TRIGGERED" : "armed";
    halt.classList.remove("alive", "warn", "crit");
    if (s.halted) halt.classList.add("crit");
    else halt.classList.add("alive");
  }
}

// ─────────────────────────── pipeline

function renderPipeline(p) {
  const grid = $("pipeline-grid");
  if (!p || !p.pipeline || p.pipeline.length === 0) {
    grid.innerHTML = `<div class="empty">no pipeline data</div>`;
    return;
  }
  const totalEvents = p.pipeline.reduce((a, b) => a + (b.events_24h || 0), 0);
  $("pipeline-meta").textContent = `${totalEvents} events / 24h`;

  grid.innerHTML = p.pipeline.map((row) => {
    let dotClass = "cold";
    if (row.errors_24h > 0) dotClass = "err";
    else if (row.last_ts) {
      const age = (Date.now() - new Date(row.last_ts).getTime()) / 1000;
      if (age < 3600) dotClass = "";          // green (default)
      else if (age < 86400) dotClass = "warm"; // amber
    }
    return `
      <div class="pipeline-row">
        <span class="dot ${dotClass}"></span>
        <span class="pipe-name">${row.trigger}</span>
        <span class="pipe-meta">${row.last_relative} · ${row.events_24h} events</span>
        <span class="pipe-errors">${row.errors_24h > 0 ? row.errors_24h + " err" : ""}</span>
      </div>
    `;
  }).join("");
}

// ─────────────────────────── positions

function renderPositions(p) {
  const list = $("positions-list");
  if (!p || !p.positions) { list.innerHTML = `<div class="empty">no data</div>`; return; }
  $("positions-meta").textContent =
    `${p.count} contracts · equity $${Number(p.account?.equity || 0).toLocaleString("en-US", {maximumFractionDigits:0})} · total ${fmtMoney(p.total_unrealized_pnl)}`;

  if (p.count === 0) {
    list.innerHTML = `<div class="empty">no open positions</div>`;
    return;
  }
  list.innerHTML = p.positions.map((pos) => {
    const ivStr = pos.iv !== null ? `IV ${(pos.iv * 100).toFixed(1)}%` : "";
    const deltaStr = pos.delta !== null ? `Δ ${pos.delta.toFixed(2)}` : "";
    const thesis = pos.thesis && pos.thesis.thesis_text
      ? `<div class="pos-thesis">
           <em>${pos.thesis.direction || ""}</em>${escapeHtml(pos.thesis.thesis_text)}
           ${pos.thesis.invalidation ? `<br><em>invalidation</em>${escapeHtml(pos.thesis.invalidation)}` : ""}
         </div>`
      : "";
    return `
      <div class="position-row">
        <div class="pos-label">
          ${escapeHtml(pos.label)}
          <span class="symbol">${escapeHtml(pos.symbol)} · ${pos.age || ""}</span>
        </div>
        <div class="cell-right">${pos.qty}</div>
        <div class="cell-right">${fmt(pos.entry, 2)} → ${fmt(pos.mark, 2)}</div>
        <div class="cell-right ${pnlClass(pos.unrealized_pnl)}">${fmtMoney(pos.unrealized_pnl)}</div>
        <div class="cell-right ${pnlClass(pos.unrealized_pnl)}">${fmtPct(pos.pnl_pct)}</div>
        ${thesis}
      </div>
    `;
  }).join("");
}

// ─────────────────────────── risk

function renderRisk(r) {
  const grid = $("risk-grid");
  if (!r || !r.snapshot) { grid.innerHTML = `<div class="empty">no snapshot yet</div>`; return; }
  const s = r.snapshot;
  $("risk-meta").textContent = s.age ? `as of ${s.age}` : "—";

  const g = s.aggregate_greeks || {};
  const h = s.heat || {};
  const cells = [
    ["heat",       fmtPct((h.heat_pct || 0) * 100, 2)],
    ["max / name", fmtPct((h.max_per_underlying_pct || 0) * 100, 2)],
    ["net delta",  fmt(g.net_delta, 2)],
    ["net gamma",  fmt(g.net_gamma, 3)],
    ["net vega",   fmt(g.net_vega, 2)],
    ["net theta",  fmt(g.net_theta, 2)],
    ["positions",  String(s.n_positions || 0)],
    ["data q",     `lvl ${s.data_quality?.degradation_level ?? 0}`],
  ];
  grid.innerHTML = cells.map(([k, v]) => `
    <div class="risk-row">
      <label>${k}</label>
      <strong>${v}</strong>
    </div>
  `).join("");
}

// ─────────────────────────── regime

const REGIME_ORDER = ["BULL_TREND", "RANGE_LOW_VOL", "VOLATILE_TRANSITION", "BEAR_TREND", "CRISIS"];

function renderRegime(r) {
  const block = $("regime-block");
  if (!r || !r.current) { block.innerHTML = `<div class="empty">no regime yet</div>`; return; }
  const c = r.current;
  $("regime-meta").textContent = c.as_of ? ago(c.as_of) : "—";

  const probs = c.probabilities || {};
  const total = Object.values(probs).reduce((a, b) => a + Number(b || 0), 0) || 1;
  const bars = REGIME_ORDER.map((label) => {
    const p = Number(probs[label] || 0) / total;
    const dim = label !== c.label ? " dim" : "";
    return `
      <div class="regime-bar${dim}">
        <span>${label}</span>
        <span class="bar-fill"><span class="bar-inner" style="width:${(p * 100).toFixed(1)}%"></span></span>
        <span class="pct">${(p * 100).toFixed(0)}%</span>
      </div>
    `;
  }).join("");

  const gate = c.gate || {};
  block.innerHTML = `
    <div class="label-big">${c.label || "UNKNOWN"}</div>
    <div class="conf">confidence ${(c.confidence || 0).toFixed(2)} · level ${c.degradation_level || 0}</div>
    <div class="regime-bars">${bars}</div>
    <div class="regime-gates">
      <div>allow new entries · <strong>${gate.allow_new_entries === false ? "no" : "yes"}</strong></div>
      <div>size multiplier · <strong>${(gate.size_multiplier ?? 1).toFixed(2)}×</strong></div>
      <div>require risk LLM · <strong>${gate.require_llm_risk_review ? "yes" : "no"}</strong></div>
      <div>crisis flags · <strong>${(c.crisis_flags || []).length || 0}</strong></div>
    </div>
  `;
}

// ─────────────────────────── agent activity

function renderEvents(ev) {
  const stream = $("agent-stream");
  if (!ev || !ev.events || ev.events.length === 0) {
    stream.innerHTML = `<div class="empty">no events</div>`;
    return;
  }
  stream.innerHTML = ev.events.map((e) => {
    const sevCls = e.severity >= 2 ? "sev-2" : e.severity >= 1 ? "sev-1" : "";
    const payload = e.payload || {};
    let what = e.event_type;
    if (payload.ticker)         what += ` · ${payload.ticker}`;
    else if (payload.symbol)    what += ` · ${payload.symbol}`;
    else if (payload.label)     what += ` · ${payload.label}`;
    else if (payload.decision)  what += ` · ${payload.decision}`;
    return `
      <div class="agent-row ${sevCls}">
        <span class="when">${timeHM(e.ts)}</span>
        <span class="who">${escapeHtml(e.agent || "")}</span>
        <span class="what">${escapeHtml(what)}</span>
      </div>
    `;
  }).join("");
}

// ─────────────────────────── trades

function renderTrades(t) {
  const list = $("trades-list");
  if (!t || !t.trades) { list.innerHTML = `<div class="empty">no closed trades</div>`; return; }
  if (t.trades.length === 0) { list.innerHTML = `<div class="empty">no closed trades</div>`; return; }
  const s = t.stats || {};
  $("trades-meta").textContent = `${s.n} closed · ${(s.win_rate * 100).toFixed(0)}% win · ${fmtMoney(s.pnl_total)} total`;

  list.innerHTML = t.trades.map((tr) => `
    <div class="trade-row">
      <span><span class="outcome ${tr.outcome}">${tr.outcome}</span></span>
      <div class="pos-label">
        ${escapeHtml(tr.label || tr.symbol)}
        <span class="symbol">${escapeHtml(tr.symbol)}</span>
      </div>
      <div class="cell-right">${tr.qty}</div>
      <div class="cell-right">${fmt(tr.entry)} → ${fmt(tr.exit)}</div>
      <div class="cell-right ${pnlClass(tr.pnl)}">${fmtMoney(tr.pnl)}</div>
      <div class="cell-right cell-extra" style="color:var(--ink-faint);font-size:11px">${tr.close_reason || ""}</div>
    </div>
  `).join("");
}

// ─────────────────────────── learning

function renderLearning(L) {
  const block = $("learning-block");
  if (!L) { block.innerHTML = `<div class="empty">no data</div>`; return; }
  const meta = `${L.canaries?.length || 0} canary · ${L.recent_events?.length || 0} recent events`;
  $("learning-meta").textContent = meta;

  const sections = [];

  // Active version
  if (L.active) {
    sections.push(`
      <div class="learning-section">
        <h3>active param version</h3>
        ${renderVersionCard(L.active)}
      </div>
    `);
  }

  // Canaries
  if (L.canaries && L.canaries.length > 0) {
    sections.push(`
      <div class="learning-section">
        <h3>canaries</h3>
        ${L.canaries.map(renderVersionCard).join("")}
      </div>
    `);
  }

  // Recent learning events
  if (L.recent_events && L.recent_events.length > 0) {
    sections.push(`
      <div class="learning-section">
        <h3>recent learning events</h3>
        <div class="agent-stream" style="max-height:200px">
          ${L.recent_events.map((e) => `
            <div class="agent-row">
              <span class="when">${e.age}</span>
              <span class="who">${escapeHtml(e.event_type)}</span>
              <span class="what">v${e.param_version_id || "—"}</span>
            </div>
          `).join("")}
        </div>
      </div>
    `);
  }

  block.innerHTML = sections.join("") || `<div class="empty">no learning state yet</div>`;
}

function renderVersionCard(v) {
  return `
    <div class="param-version">
      <span class="ver-id">v${v.version_id}</span>
      <span class="ver-status ${v.status}">${v.status}</span>
      <span style="margin-left:12px">${v.scope}${v.scope_key ? `/${v.scope_key}` : ""}</span>
      ${v.traffic_pct !== null && v.status === "CANARY" ? `<span style="margin-left:12px">${v.traffic_pct.toFixed(0)}% traffic</span>` : ""}
      ${v.params && Object.keys(v.params).length > 0 ? `<pre>${escapeHtml(JSON.stringify(v.params, null, 2))}</pre>` : ""}
    </div>
  `;
}

// ─────────────────────────── chart

function ensureChart() {
  if (state.chart) return;
  const el = $("chart-container");
  state.chart = LightweightCharts.createChart(el, {
    layout: {
      background: { color: "transparent" },
      textColor: "#525252",
      fontFamily: "'SF Mono', 'JetBrains Mono', ui-monospace, monospace",
      fontSize: 11,
    },
    grid: {
      vertLines: { color: "rgba(15,15,15,0.04)" },
      horzLines: { color: "rgba(15,15,15,0.04)" },
    },
    rightPriceScale: { borderVisible: false },
    timeScale: { borderVisible: false, timeVisible: false, secondsVisible: false },
    crosshair: {
      mode: 1,
      vertLine: { width: 1, color: "rgba(15,15,15,0.2)", style: 3 },
      horzLine: { width: 1, color: "rgba(15,15,15,0.2)", style: 3 },
    },
  });
  state.candleSeries = state.chart.addCandlestickSeries({
    upColor: "#0F766E",
    downColor: "#B91C1C",
    borderUpColor: "#0F766E",
    borderDownColor: "#B91C1C",
    wickUpColor: "rgba(15,118,110,0.6)",
    wickDownColor: "rgba(185,28,28,0.6)",
  });
  state.volumeSeries = state.chart.addHistogramSeries({
    color: "rgba(15,15,15,0.08)",
    priceFormat: { type: "volume" },
    priceScaleId: "",
  });
  state.volumeSeries.priceScale().applyOptions({
    scaleMargins: { top: 0.85, bottom: 0 },
  });
  window.addEventListener("resize", () => {
    if (state.chart) state.chart.applyOptions({ width: el.clientWidth });
  });
}

async function loadChart(ticker) {
  ensureChart();
  state.current_ticker = ticker;
  const data = await fetchJSON(`/api/quote/${encodeURIComponent(ticker)}?days=${CHART_DAYS}`);
  if (!data || !data.candles || data.candles.length === 0) {
    $("ticker-quote").textContent = "no data";
    return;
  }
  state.candleSeries.setData(data.candles);
  state.volumeSeries.setData(data.candles.map((c) => ({
    time: c.time,
    value: c.volume,
    color: c.close >= c.open ? "rgba(15,118,110,0.15)" : "rgba(185,28,28,0.15)",
  })));
  state.chart.timeScale().fitContent();
  if (data.latest) {
    const cls = data.latest.change_pct >= 0 ? "pos" : "neg";
    const arrow = data.latest.change_pct >= 0 ? "▲" : "▼";
    $("ticker-quote").innerHTML =
      `<strong>${data.latest.last.toFixed(2)}</strong>` +
      `<span class="${cls}" style="margin-left:8px">${arrow} ${Math.abs(data.latest.change_pct * 100).toFixed(2)}%</span>`;
  }
}

async function populateTickerSelect() {
  const sel = $("ticker-select");
  const data = await fetchJSON("/api/watchlist");
  if (!data || !data.tickers) return;
  sel.innerHTML = data.tickers.map((t) => `<option value="${t}">${t}</option>`).join("");
  sel.value = state.current_ticker || data.tickers[0];
  sel.addEventListener("change", (e) => loadChart(e.target.value));
  if (!state.current_ticker) loadChart(sel.value);
}

// ─────────────────────────── orchestration

async function refreshAll() {
  const [status, pipeline, positions, regime, risk, events, trades, learning] = await Promise.all([
    fetchJSON("/api/status"),
    fetchJSON("/api/pipeline"),
    fetchJSON("/api/positions"),
    fetchJSON("/api/regime"),
    fetchJSON("/api/risk"),
    fetchJSON(`/api/events?limit=80&severity_min=${state.severity_filter}`),
    fetchJSON("/api/trades?limit=20"),
    fetchJSON("/api/learning"),
  ]);

  renderStatus(status);
  renderPipeline(pipeline);
  renderPositions(positions);
  renderRegime(regime);
  renderRisk(risk);
  renderEvents(events);
  renderTrades(trades);
  renderLearning(learning);

  $("last-refresh").textContent = new Date().toLocaleTimeString();
}

function setupFilterChips() {
  document.querySelectorAll(".chip[data-sev]").forEach((chip) => {
    chip.addEventListener("click", () => {
      document.querySelectorAll(".chip[data-sev]").forEach((c) => c.classList.remove("active"));
      chip.classList.add("active");
      state.severity_filter = Number(chip.dataset.sev);
      refreshAll();
    });
  });
}

(async function init() {
  setupFilterChips();
  await populateTickerSelect();
  await refreshAll();
  setInterval(refreshAll, REFRESH_MS);
})();
