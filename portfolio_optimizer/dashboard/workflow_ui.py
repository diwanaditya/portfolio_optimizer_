"""
Workflow Dashboard -- the full UI/UX rebuild.

Single self-contained HTML file (same "no server, no build step" pattern
as generator.py) implementing all five requested pieces:

  1. One-click workflow cards (Create Portfolio / Optimize / Backtest /
     Paper Trade / Generate Report) as the landing view, tabbing into
     the relevant section rather than dumping every module on the user
     at once. Icons are custom line-drawn SVG shapes (a small
     instrument-panel icon set), not emoji or unicode glyphs.
  2. Professional charts: interactive efficient frontier (scatter, with
     individual assets and the current portfolio plotted against the
     frontier), allocation treemap (custom squarified-treemap SVG, no
     extra CDN dependency), rolling Sharpe, rolling drawdown, factor
     exposure, and risk contribution.
  3. Portfolio Health Card: Expected Return / Volatility / Sharpe / Max
     Drawdown / VaR / CVaR, color-coded with this repo's established
     white/black/green/red contract (green=favorable, red=unfavorable,
     black=neutral -- not just "positive=green").
  4. Explainability: real weight-change narratives from
     `WeightChangeExplainer`, not templated filler.
  5. Interactive report: tabbed sections for allocation, attribution
     (factor + risk contribution), and weight-change history, all
     clickable from one page.

DESIGN SYSTEM (grounded in the brief: an institutional quant desk's
internal instrument, not a consumer app): a fixed 8px spacing scale,
a grotesk sans (Inter) for structure/labels and a monospace face
(JetBrains Mono) for every numeric value -- tabular-figure numerics are
a real trading-terminal convention (numbers align in a column the way
they do on a Bloomberg screen), used here as the page's one deliberate
signature rather than decoration. Custom line-icon shapes for the
workflow cards, consistent card radius/shadow, and calibrated (not
blanket positive=green) health-metric coloring.
"""
from __future__ import annotations
import json


COLOR_WHITE = "#ffffff"
COLOR_INK = "#14161a"          # near-black text -- softer than pure #000
COLOR_GREEN = "#0a8a3f"
COLOR_RED = "#c0392b"
COLOR_MUTED = "#6b7280"        # secondary/label text
COLOR_BORDER = "#e6e6e4"
COLOR_BORDER_STRONG = "#d8d8d5"
COLOR_ACCENT = "#25406b"       # deep slate-navy -- neutral chart accent, never used for P&L
COLOR_SURFACE_HOVER = "#fafaf9"

SPACE = {1: "4px", 2: "8px", 3: "12px", 4: "16px", 5: "24px", 6: "32px", 7: "48px"}

FONT_SANS = "'Inter', -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
FONT_MONO = "'JetBrains Mono', 'SF Mono', Consolas, monospace"


# --------------------------------------------------------------------- #
# Custom line-icon shapes (24x24 viewBox, stroke-based, matched weight) --
# an instrument-panel-style icon set instead of emoji/unicode glyphs.
# --------------------------------------------------------------------- #
_ICON_CREATE = """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><circle cx="12" cy="12" r="8.2"/><path d="M12 8.4v7.2M8.4 12h7.2"/></svg>"""
_ICON_OPTIMIZE = """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><path d="M4 17V7M4 7l3 3M4 7l-3 3" transform="translate(1 0)"/><circle cx="6" cy="14" r="2"/><circle cx="12" cy="8" r="2"/><circle cx="18" cy="16" r="2"/><path d="M6 12v-5M12 6V4M18 14v-2" stroke-opacity="0"/><path d="M4 20h16"/><path d="M6 20v-4M12 20V10M18 20v-2"/></svg>"""
_ICON_BACKTEST = """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M4 12a8 8 0 1 1 2.6 5.9"/><path d="M4 12V7M4 12h5"/></svg>"""
_ICON_PAPERTRADE = """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="8.2"/><path d="M10 8.5l6 3.5-6 3.5z" fill="currentColor" stroke="none"/></svg>"""
_ICON_REPORT = """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><rect x="4.5" y="3.5" width="15" height="17" rx="1.5"/><path d="M8 14v3M12 11v6M16 8.5V17"/></svg>"""

_WORKFLOW_STEPS = [
    ("optimize", _ICON_CREATE, "Create Portfolio"),
    ("optimize", _ICON_OPTIMIZE, "Optimize"),
    ("backtest", _ICON_BACKTEST, "Backtest"),
    ("papertrade", _ICON_PAPERTRADE, "Paper Trade"),
    ("report", _ICON_REPORT, "Generate Report"),
]


def _metric_color(name: str, value: float) -> str:
    """Health-card color logic: which direction is 'good' differs per
    metric (higher Sharpe is good, higher drawdown-magnitude is bad), so
    this is NOT a blind positive=green rule -- it's calibrated per metric.
    """
    if value is None or (isinstance(value, float) and value != value):  # NaN check
        return COLOR_INK
    if name in ("expected_return", "sharpe_ratio"):
        return COLOR_GREEN if value > 0 else (COLOR_RED if value < 0 else COLOR_INK)
    if name in ("max_drawdown",):
        return COLOR_RED if value < -0.001 else COLOR_INK
    if name in ("volatility", "var_95", "cvar_95"):
        return COLOR_INK  # magnitude-only metrics -- neither "good" nor "bad" in isolation
    return COLOR_INK


def render_workflow_dashboard(data: dict) -> str:
    health = data["health"]
    frontier = data["efficient_frontier"]
    treemap = data["treemap"]
    rolling = data["rolling"]
    factor_exposure = data["factor_exposure"]
    risk_contribution = data["risk_contribution"]
    weight_changes = data.get("weight_changes", [])
    portfolio_name = data.get("portfolio_name", "Portfolio")

    workflow_cards_html = ""
    for i, (tab, icon, label) in enumerate(_WORKFLOW_STEPS):
        active_class = " active" if i == 0 else ""
        workflow_cards_html += f"""
        <div class="workflow-card{active_class}" data-tab="{tab}">
          <div class="workflow-icon">{icon}</div>
          <div class="workflow-label">{label}</div>
        </div>"""

    health_cards_html = ""
    health_specs = [
        ("expected_return", "Expected Return", "{:.2%}"),
        ("volatility", "Volatility", "{:.2%}"),
        ("sharpe_ratio", "Sharpe Ratio", "{:.2f}"),
        ("max_drawdown", "Max Drawdown", "{:.2%}"),
        ("var_95", "VaR (95%)", "{:.2%}"),
        ("cvar_95", "CVaR (95%)", "{:.2%}"),
    ]
    for key, label, fmt in health_specs:
        val = health.get(key, float("nan"))
        color = _metric_color(key, val)
        display = fmt.format(val) if val == val else "—"
        health_cards_html += f"""
        <div class="health-card">
          <div class="health-label">{label}</div>
          <div class="health-value" style="color:{color}">{display}</div>
        </div>"""

    explanations_html = ""
    for wc in weight_changes:
        change_color = COLOR_GREEN if wc["weight_change"] > 1e-6 else (COLOR_RED if wc["weight_change"] < -1e-6 else COLOR_INK)
        explanations_html += f"""
        <div class="explain-row">
          <div class="explain-asset" style="color:{change_color}">{wc['asset']}</div>
          <div class="explain-text">{wc['narrative']}</div>
        </div>"""
    if not explanations_html:
        explanations_html = '<div class="empty">No prior allocation supplied for comparison.</div>'

    data_json = json.dumps({
        "frontier": frontier, "treemap": treemap, "rolling": rolling,
        "factor_exposure": factor_exposure, "risk_contribution": risk_contribution,
    })

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{portfolio_name} — Workflow Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
  :root {{
    --sp-1: {SPACE[1]}; --sp-2: {SPACE[2]}; --sp-3: {SPACE[3]}; --sp-4: {SPACE[4]};
    --sp-5: {SPACE[5]}; --sp-6: {SPACE[6]}; --sp-7: {SPACE[7]};
    --ink: {COLOR_INK}; --muted: {COLOR_MUTED}; --border: {COLOR_BORDER};
    --border-strong: {COLOR_BORDER_STRONG}; --accent: {COLOR_ACCENT};
    --green: {COLOR_GREEN}; --red: {COLOR_RED}; --surface-hover: {COLOR_SURFACE_HOVER};
    --radius: 12px; --radius-sm: 8px;
  }}
  * {{ box-sizing: border-box; }}
  html {{ -webkit-font-smoothing: antialiased; }}
  body {{
    font-family: {FONT_SANS}; background: {COLOR_WHITE}; color: var(--ink);
    margin: 0; padding: 0; line-height: 1.5; font-size: 14px;
  }}
  code {{ font-family: {FONT_MONO}; background: var(--surface-hover); padding: 1px 5px; border-radius: 4px; font-size: 0.92em; }}

  .topbar {{ padding: var(--sp-6) var(--sp-6) 0; max-width: 1220px; margin: 0 auto; }}
  h1 {{ font-size: 22px; font-weight: 700; margin: 0 0 var(--sp-1) 0; letter-spacing: -0.01em; }}
  .subtitle {{ color: var(--muted); font-size: 13px; margin-bottom: var(--sp-5); }}

  /* Workflow cards */
  .workflow-cards {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: var(--sp-3); margin-bottom: var(--sp-2); }}
  .workflow-card {{
    border: 1px solid var(--border); border-radius: var(--radius); padding: var(--sp-4) var(--sp-3);
    cursor: pointer; text-align: center; background: {COLOR_WHITE};
    transition: border-color 0.15s ease, box-shadow 0.15s ease, transform 0.1s ease;
    display: flex; flex-direction: column; align-items: center; gap: var(--sp-2);
  }}
  .workflow-card:hover {{ border-color: var(--border-strong); box-shadow: 0 2px 10px rgba(20,22,26,0.06); transform: translateY(-1px); }}
  .workflow-card.active {{ border: 1.5px solid var(--ink); box-shadow: 0 2px 12px rgba(20,22,26,0.08); }}
  .workflow-icon {{ width: 26px; height: 26px; color: var(--muted); }}
  .workflow-card.active .workflow-icon {{ color: var(--ink); }}
  .workflow-icon svg {{ width: 100%; height: 100%; }}
  .workflow-label {{ font-size: 12.5px; font-weight: 600; letter-spacing: -0.005em; }}

  .container {{ max-width: 1220px; margin: 0 auto; padding: var(--sp-5) var(--sp-6) var(--sp-7); }}
  .tab-content {{ display: none; }}
  .tab-content.active {{ display: block; animation: fadein 0.18s ease; }}
  @keyframes fadein {{ from {{ opacity: 0; transform: translateY(2px); }} to {{ opacity: 1; transform: translateY(0); }} }}

  /* Section label -- small-caps eyebrow + thin rule, used consistently */
  .section-label {{
    font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.07em;
    color: var(--muted); margin: 0 0 var(--sp-4) 0; display: flex; align-items: center; gap: var(--sp-3);
  }}
  .section-label::after {{ content: ""; flex: 1; height: 1px; background: var(--border); }}

  /* Health card row */
  .health-grid {{ display: grid; grid-template-columns: repeat(6, 1fr); gap: var(--sp-3); margin-bottom: var(--sp-6); }}
  .health-card {{
    border: 1px solid var(--border); border-radius: var(--radius-sm); padding: var(--sp-4) var(--sp-3);
    background: {COLOR_WHITE};
  }}
  .health-label {{ font-size: 10.5px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: var(--sp-2); font-weight: 600; }}
  .health-value {{ font-family: {FONT_MONO}; font-size: 20px; font-weight: 600; font-variant-numeric: tabular-nums; letter-spacing: -0.01em; }}

  .card {{
    border: 1px solid var(--border); border-radius: var(--radius); padding: var(--sp-5);
    margin-bottom: var(--sp-5); background: {COLOR_WHITE};
  }}
  .card h3 {{
    font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted);
    margin: 0 0 var(--sp-4) 0; font-weight: 600;
  }}
  .chart-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: var(--sp-5); }}
  canvas {{ max-height: 280px; }}

  .explain-row {{ display: flex; gap: var(--sp-4); padding: var(--sp-3) 0; border-bottom: 1px solid var(--border); align-items: baseline; }}
  .explain-row:last-child {{ border-bottom: none; }}
  .explain-asset {{ font-family: {FONT_MONO}; font-weight: 600; font-size: 12.5px; min-width: 96px; letter-spacing: -0.01em; }}
  .explain-text {{ font-size: 13px; color: var(--ink); line-height: 1.6; }}
  .empty {{ color: var(--muted); padding: var(--sp-6) 0; text-align: center; font-size: 13px; }}

  #treemap {{ width: 100%; height: 280px; }}
  .treemap-cell-label {{ font-family: {FONT_SANS}; font-size: 11.5px; font-weight: 600; fill: white; pointer-events: none; }}
  .treemap-cell-weight {{ font-family: {FONT_MONO}; font-size: 10.5px; fill: rgba(255,255,255,0.85); pointer-events: none; }}

  .placeholder-text {{ font-size: 13px; color: var(--muted); line-height: 1.7; max-width: 640px; }}

  @media (max-width: 860px) {{
    .workflow-cards {{ grid-template-columns: repeat(3, 1fr); }}
    .health-grid {{ grid-template-columns: repeat(3, 1fr); }}
    .chart-row {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>
<div class="topbar">
  <h1>{portfolio_name}</h1>
  <div class="subtitle">Workflow dashboard — select a step to jump to that view</div>
  <div class="workflow-cards">{workflow_cards_html}</div>
</div>

<div class="container">

  <div class="section-label">Portfolio Health</div>
  <div class="health-grid">{health_cards_html}</div>

  <!-- OPTIMIZE / CREATE PORTFOLIO TAB -->
  <div class="tab-content active" id="tab-optimize">
    <div class="card">
      <h3>Efficient Frontier</h3>
      <canvas id="frontierChart"></canvas>
    </div>
    <div class="card">
      <h3>Allocation Treemap</h3>
      <svg id="treemap"></svg>
    </div>
  </div>

  <!-- BACKTEST TAB -->
  <div class="tab-content" id="tab-backtest">
    <div class="chart-row">
      <div class="card"><h3>Rolling Sharpe · 63-Day</h3><canvas id="rollingSharpeChart"></canvas></div>
      <div class="card"><h3>Rolling Drawdown</h3><canvas id="rollingDrawdownChart"></canvas></div>
    </div>
  </div>

  <!-- PAPER TRADE TAB -->
  <div class="tab-content" id="tab-papertrade">
    <div class="card">
      <h3>Paper Trading</h3>
      <p class="placeholder-text">
        Live paper-trading state (positions, orders, NAV history) is rendered by the
        separate <code>dashboard/generator.py</code> view, which reads directly from
        <code>infra.persistence.PortfolioStateStore</code> — see
        <code>trading_dashboard.html</code> for that live view. This tab is a
        navigation placeholder pointing to it rather than duplicating a second
        live-data reader in the same page.
      </p>
    </div>
  </div>

  <!-- REPORT TAB (attribution + risk contribution + explainability) -->
  <div class="tab-content" id="tab-report">
    <div class="chart-row">
      <div class="card"><h3>Factor Exposure</h3><canvas id="factorChart"></canvas></div>
      <div class="card"><h3>Contribution to Risk</h3><canvas id="riskContribChart"></canvas></div>
    </div>
    <div class="card">
      <h3>Why Did The Allocation Change</h3>
      {explanations_html}
    </div>
  </div>

</div>

<script>
  const DATA = {data_json};
  const FONT_SANS = "{FONT_SANS}".split(",")[0].replace(/'/g, "");
  Chart.defaults.font.family = "{FONT_SANS}";
  Chart.defaults.color = "{COLOR_MUTED}";

  // ---- Tab navigation ----
  document.querySelectorAll(".workflow-card").forEach(card => {{
    card.addEventListener("click", () => {{
      document.querySelectorAll(".workflow-card").forEach(c => c.classList.remove("active"));
      card.classList.add("active");
      const tab = card.dataset.tab;
      document.querySelectorAll(".tab-content").forEach(t => t.classList.remove("active"));
      document.getElementById("tab-" + tab).classList.add("active");
    }});
  }});

  // ---- Efficient Frontier (scatter: curve + assets + current portfolio) ----
  new Chart(document.getElementById("frontierChart"), {{
    type: "scatter",
    data: {{
      datasets: [
        {{
          label: "Efficient Frontier", showLine: true, fill: false,
          data: DATA.frontier.frontier.map(p => ({{x: p.risk, y: p.return}})),
          borderColor: "{COLOR_ACCENT}", pointRadius: 0, borderWidth: 2, tension: 0.15,
        }},
        {{
          label: "Individual Assets", data: DATA.frontier.assets.map(a => ({{x: a.risk, y: a.return, name: a.name}})),
          backgroundColor: "{COLOR_MUTED}", pointRadius: 4.5, pointHoverRadius: 6,
        }},
        {{
          label: "Max-Sharpe Portfolio",
          data: DATA.frontier.tangency ? [{{x: DATA.frontier.tangency.risk, y: DATA.frontier.tangency.return}}] : [],
          backgroundColor: "{COLOR_GREEN}", pointRadius: 7, pointHoverRadius: 9, pointStyle: "star",
        }},
      ]
    }},
    options: {{
      responsive: true,
      layout: {{ padding: {{ top: 8, right: 8 }} }},
      plugins: {{
        legend: {{ position: "bottom", labels: {{ font: {{ size: 11.5 }}, usePointStyle: true, boxWidth: 7, padding: 16 }} }},
        tooltip: {{
          backgroundColor: "{COLOR_INK}", padding: 10, cornerRadius: 6, displayColors: false,
          titleFont: {{ size: 12 }}, bodyFont: {{ family: "{FONT_MONO}", size: 12 }},
          callbacks: {{
            label: (ctx) => {{
              const p = ctx.raw;
              const name = p.name ? p.name + "  " : "";
              return `${{name}}risk ${{(p.x*100).toFixed(1)}}%   return ${{(p.y*100).toFixed(1)}}%`;
            }}
          }}
        }}
      }},
      scales: {{
        x: {{
          title: {{ display: true, text: "Volatility (annualized)", font: {{ size: 11.5 }}, color: "{COLOR_MUTED}" }},
          ticks: {{ callback: v => (v*100).toFixed(0) + "%", font: {{ family: "{FONT_MONO}", size: 11 }} }},
          grid: {{ color: "{COLOR_BORDER}" }}
        }},
        y: {{
          title: {{ display: true, text: "Expected Return (annualized)", font: {{ size: 11.5 }}, color: "{COLOR_MUTED}" }},
          ticks: {{ callback: v => (v*100).toFixed(0) + "%", font: {{ family: "{FONT_MONO}", size: 11 }} }},
          grid: {{ color: "{COLOR_BORDER}" }}
        }},
      }}
    }}
  }});

  // ---- Rolling Sharpe / Drawdown ----
  function lineChart(canvasId, points, color, yFmt) {{
    new Chart(document.getElementById(canvasId), {{
      type: "line",
      data: {{
        labels: points.map(p => p.t),
        datasets: [{{ data: points.map(p => p.v), borderColor: color, borderWidth: 1.8,
                      pointRadius: 0, pointHoverRadius: 4, fill: true,
                      backgroundColor: color + "14", tension: 0.2 }}]
      }},
      options: {{
        responsive: true,
        layout: {{ padding: {{ top: 8, right: 8 }} }},
        plugins: {{
          legend: {{ display: false }},
          tooltip: {{
            backgroundColor: "{COLOR_INK}", padding: 10, cornerRadius: 6, displayColors: false,
            bodyFont: {{ family: "{FONT_MONO}", size: 12 }},
          }}
        }},
        scales: {{
          x: {{ ticks: {{ maxTicksLimit: 7, font: {{ size: 10.5 }} }}, grid: {{ display: false }} }},
          y: {{ ticks: {{ callback: yFmt, font: {{ family: "{FONT_MONO}", size: 11 }} }}, grid: {{ color: "{COLOR_BORDER}" }} }}
        }}
      }}
    }});
  }}
  lineChart("rollingSharpeChart", DATA.rolling.rolling_sharpe, "{COLOR_ACCENT}", v => v.toFixed(1));
  lineChart("rollingDrawdownChart", DATA.rolling.rolling_drawdown, "{COLOR_RED}", v => (v*100).toFixed(0) + "%");

  // ---- Factor Exposure (diverging bar, green/red by sign) ----
  new Chart(document.getElementById("factorChart"), {{
    type: "bar",
    data: {{
      labels: DATA.factor_exposure.factors.map(f => f.name),
      datasets: [{{
        data: DATA.factor_exposure.factors.map(f => f.exposure),
        backgroundColor: DATA.factor_exposure.factors.map(f => f.exposure >= 0 ? "{COLOR_GREEN}" : "{COLOR_RED}"),
        borderRadius: 4, borderSkipped: false, maxBarThickness: 56,
      }}]
    }},
    options: {{
      responsive: true,
      layout: {{ padding: {{ top: 8, right: 8 }} }},
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{ backgroundColor: "{COLOR_INK}", padding: 10, cornerRadius: 6, displayColors: false,
                    bodyFont: {{ family: "{FONT_MONO}", size: 12 }} }}
      }},
      scales: {{
        y: {{ grid: {{ color: "{COLOR_BORDER}" }}, ticks: {{ font: {{ family: "{FONT_MONO}", size: 11 }} }} }},
        x: {{ grid: {{ display: false }}, ticks: {{ font: {{ size: 11.5 }} }} }}
      }}
    }}
  }});

  // ---- Contribution to Risk ----
  new Chart(document.getElementById("riskContribChart"), {{
    type: "bar",
    data: {{
      labels: DATA.risk_contribution.map(r => r.name),
      datasets: [{{ data: DATA.risk_contribution.map(r => r.contribution_pct), backgroundColor: "{COLOR_ACCENT}",
                    borderRadius: 4, borderSkipped: false }}]
    }},
    options: {{
      responsive: true, indexAxis: "y",
      layout: {{ padding: {{ top: 8, right: 8 }} }},
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{ backgroundColor: "{COLOR_INK}", padding: 10, cornerRadius: 6, displayColors: false,
                    bodyFont: {{ family: "{FONT_MONO}", size: 12 }} }}
      }},
      scales: {{
        x: {{ ticks: {{ callback: v => (v*100).toFixed(0) + "%", font: {{ family: "{FONT_MONO}", size: 11 }} }}, grid: {{ color: "{COLOR_BORDER}" }} }},
        y: {{ grid: {{ display: false }}, ticks: {{ font: {{ size: 11.5 }} }} }}
      }}
    }}
  }});

  // ---- Allocation Treemap (custom squarified layout, no extra CDN dep) ----
  function squarify(items, x, y, w, h) {{
    if (items.length === 0) return [];
    if (items.length === 1) return [{{ ...items[0], x, y, w, h }}];
    const total = items.reduce((s, i) => s + i.weight, 0);
    let acc = 0, splitIdx = 0;
    for (let i = 0; i < items.length; i++) {{
      acc += items[i].weight;
      if (acc >= total / 2) {{ splitIdx = i + 1; break; }}
    }}
    const group1 = items.slice(0, Math.max(splitIdx, 1));
    const group2 = items.slice(Math.max(splitIdx, 1));
    const w1total = group1.reduce((s, i) => s + i.weight, 0);
    const frac = w1total / total;
    if (w >= h) {{
      const w1 = w * frac;
      return [...squarify(group1, x, y, w1, h), ...squarify(group2, x + w1, y, w - w1, h)];
    }} else {{
      const h1 = h * frac;
      return [...squarify(group1, x, y, w, h1), ...squarify(group2, x, y + h1, w, h - h1)];
    }}
  }}

  const svg = document.getElementById("treemap");
  const GAP = 3;
  const svgWidth = svg.parentElement.clientWidth - 4, svgHeight = 280;
  svg.setAttribute("viewBox", `0 0 ${{svgWidth}} ${{svgHeight}}`);
  const cells = squarify(DATA.treemap, 0, 0, svgWidth, svgHeight);
  cells.forEach(c => {{
    const color = c.expected_return > 0.001 ? "{COLOR_GREEN}" : (c.expected_return < -0.001 ? "{COLOR_RED}" : "{COLOR_MUTED}");
    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("x", c.x + GAP/2); rect.setAttribute("y", c.y + GAP/2);
    rect.setAttribute("width", Math.max(c.w - GAP, 0)); rect.setAttribute("height", Math.max(c.h - GAP, 0));
    rect.setAttribute("fill", color); rect.setAttribute("rx", 6);
    svg.appendChild(rect);
    if (c.w > 54 && c.h > 34) {{
      const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
      text.setAttribute("x", c.x + 12); text.setAttribute("y", c.y + 24);
      text.setAttribute("class", "treemap-cell-label");
      text.textContent = c.name;
      svg.appendChild(text);

      const weightText = document.createElementNS("http://www.w3.org/2000/svg", "text");
      weightText.setAttribute("x", c.x + 12); weightText.setAttribute("y", c.y + 40);
      weightText.setAttribute("class", "treemap-cell-weight");
      weightText.textContent = (c.weight*100).toFixed(1) + "%";
      svg.appendChild(weightText);
    }}
  }});
</script>
</body>
</html>"""
    return html


def generate_workflow_dashboard(data: dict, output_path: str) -> str:
    html = render_workflow_dashboard(data)
    with open(output_path, "w") as f:
        f.write(html)
    return output_path
