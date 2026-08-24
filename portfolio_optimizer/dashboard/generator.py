"""
Trading Dashboard Generator.

Produces a single self-contained HTML file (no server, no build step --
open it directly in a browser) from REAL persisted portfolio state (via
`infra.persistence.PortfolioStateStore`), not mock data. Color scheme is
fixed and deliberate:

    White background  -- the base canvas, no colored panels or gradients
    Black text          -- all neutral/informational text and numbers
    Green                -- any P&L, return, or change that is POSITIVE
    Red                  -- any P&L, return, or change that is NEGATIVE
    (Exactly zero / no change stays black, not green or red)

This is meant to be regenerated on a schedule (e.g. after each rebalance
cycle in `live/paper_trading_loop.py`) or run manually via:

    python -m portfolio_optimizer.dashboard.generator --portfolio-id adc_paper_v1 --db-path live_portfolio.db
"""
from __future__ import annotations
import os
import json
import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd


# Fixed color constants -- the entire design contract lives here so it's
# impossible for a color to drift from spec in different parts of the page.
COLOR_WHITE = "#ffffff"
COLOR_BLACK = "#111111"
COLOR_GREEN = "#0a8a3f"
COLOR_RED = "#c0392b"
COLOR_GRAY = "#666666"      # secondary/muted text only -- never used for P&L
COLOR_BORDER = "#e5e5e5"


def _pnl_color(value: float) -> str:
    if value > 1e-9:
        return COLOR_GREEN
    if value < -1e-9:
        return COLOR_RED
    return COLOR_BLACK


def _fmt_money(value: float) -> str:
    sign = "+" if value > 1e-9 else ("-" if value < -1e-9 else "")
    return f"{sign}${abs(value):,.2f}"


def _fmt_pct(value: float) -> str:
    sign = "+" if value > 1e-9 else ("-" if value < -1e-9 else "")
    return f"{sign}{abs(value):.2%}"


def build_dashboard_data(store, portfolio_id: str) -> dict:
    """Pull real data out of the SQLite-backed store and shape it into
    exactly what the dashboard template needs -- kept separate from HTML
    rendering so this data layer is independently testable.
    """
    snapshot = store.portfolio_snapshot(portfolio_id)
    nav_history = store.get_nav_history(portfolio_id)
    orders = store.get_orders(portfolio_id)

    if not nav_history:
        raise ValueError(f"No NAV history found for portfolio_id={portfolio_id!r}. "
                          f"Nothing to render -- has this portfolio recorded any state yet?")

    nav_series = pd.Series({h["timestamp"]: h["nav"] for h in nav_history}).sort_index()
    starting_nav = float(nav_series.iloc[0])
    current_nav = float(nav_series.iloc[-1])
    total_pnl = current_nav - starting_nav
    total_pnl_pct = (current_nav / starting_nav - 1) if starting_nav > 0 else 0.0

    if len(nav_series) >= 2:
        day_pnl = float(nav_series.iloc[-1] - nav_series.iloc[-2])
        day_pnl_pct = float(nav_series.iloc[-1] / nav_series.iloc[-2] - 1)
    else:
        day_pnl, day_pnl_pct = 0.0, 0.0

    running_max = nav_series.cummax()
    drawdown_series = (nav_series / running_max - 1)
    current_drawdown = float(drawdown_series.iloc[-1])
    max_drawdown = float(drawdown_series.min())

    positions_rows = []
    for symbol, pos in snapshot["positions"].items():
        qty = pos["quantity"]
        avg_cost = pos["avg_cost"]
        position_value = qty * avg_cost
        positions_rows.append({
            "symbol": symbol, "quantity": qty, "avg_cost": avg_cost,
            "position_value": position_value,
        })

    open_orders = [o for o in orders if o["status"] in ("new", "partially_filled", "pending_new")]
    filled_orders = [o for o in orders if o["status"] == "filled"]

    return {
        "portfolio_id": portfolio_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "current_nav": current_nav, "starting_nav": starting_nav,
        "total_pnl": total_pnl, "total_pnl_pct": total_pnl_pct,
        "day_pnl": day_pnl, "day_pnl_pct": day_pnl_pct,
        "current_drawdown": current_drawdown, "max_drawdown": max_drawdown,
        "nav_history": [{"t": t, "nav": v} for t, v in nav_series.items()],
        "positions": positions_rows,
        "open_orders": open_orders, "filled_orders_count": len(filled_orders),
        "total_orders": len(orders),
    }


def render_dashboard_html(data: dict) -> str:
    nav_points = data["nav_history"]
    nav_json = json.dumps(nav_points)

    total_pnl_color = _pnl_color(data["total_pnl"])
    day_pnl_color = _pnl_color(data["day_pnl"])
    dd_color = _pnl_color(data["current_drawdown"])

    positions_rows_html = ""
    for p in data["positions"]:
        positions_rows_html += f"""
        <tr>
          <td class="symbol">{p['symbol']}</td>
          <td class="num">{p['quantity']:,.4f}</td>
          <td class="num">${p['avg_cost']:,.2f}</td>
          <td class="num">${p['position_value']:,.2f}</td>
        </tr>"""
    if not positions_rows_html:
        positions_rows_html = '<tr><td colspan="4" class="empty">No open positions</td></tr>'

    orders_rows_html = ""
    for o in data["open_orders"]:
        orders_rows_html += f"""
        <tr>
          <td class="symbol">{o['symbol']}</td>
          <td>{o['side']}</td>
          <td class="num">{o['quantity']:,.2f}</td>
          <td>{o['status']}</td>
        </tr>"""
    if not orders_rows_html:
        orders_rows_html = '<tr><td colspan="4" class="empty">No open orders</td></tr>'

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>ADC Portfolio Dashboard -- {data['portfolio_id']}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    background: {COLOR_WHITE};
    color: {COLOR_BLACK};
    margin: 0;
    padding: 32px;
  }}
  .container {{ max-width: 1000px; margin: 0 auto; }}
  h1 {{ font-size: 20px; font-weight: 600; margin: 0 0 4px 0; }}
  .subtitle {{ color: {COLOR_GRAY}; font-size: 12.5px; margin-bottom: 28px; }}
  .metric-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 28px;
  }}
  .metric-card {{
    background: {COLOR_WHITE};
    border: 1px solid {COLOR_BORDER};
    border-radius: 8px;
    padding: 16px 18px;
  }}
  .metric-label {{ font-size: 11.5px; color: {COLOR_GRAY}; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 6px; }}
  .metric-value {{ font-size: 22px; font-weight: 700; }}
  .metric-sub {{ font-size: 12px; margin-top: 3px; }}
  .card {{
    background: {COLOR_WHITE};
    border: 1px solid {COLOR_BORDER};
    border-radius: 8px;
    padding: 20px 22px;
    margin-bottom: 20px;
  }}
  .card h3 {{ font-size: 13.5px; text-transform: uppercase; letter-spacing: 0.03em; color: {COLOR_GRAY}; margin: 0 0 14px 0; font-weight: 600; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{ text-align: left; font-size: 11px; color: {COLOR_GRAY}; text-transform: uppercase;
        padding: 6px 8px; border-bottom: 1px solid {COLOR_BORDER}; }}
  td {{ padding: 8px 8px; border-bottom: 1px solid {COLOR_BORDER}; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  td.symbol {{ font-weight: 600; }}
  td.empty {{ text-align: center; color: {COLOR_GRAY}; padding: 20px; }}
  canvas {{ max-height: 280px; }}
  .footer {{ font-size: 11px; color: {COLOR_GRAY}; margin-top: 24px; text-align: center; }}
</style>
</head>
<body>
<div class="container">
  <h1>ADC Portfolio Dashboard</h1>
  <div class="subtitle">Portfolio: {data['portfolio_id']} &middot; Generated {data['generated_at']}</div>

  <div class="metric-grid">
    <div class="metric-card">
      <div class="metric-label">Current NAV</div>
      <div class="metric-value">${data['current_nav']:,.2f}</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Total P&amp;L</div>
      <div class="metric-value" style="color:{total_pnl_color}">{_fmt_money(data['total_pnl'])}</div>
      <div class="metric-sub" style="color:{total_pnl_color}">{_fmt_pct(data['total_pnl_pct'])}</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Day P&amp;L</div>
      <div class="metric-value" style="color:{day_pnl_color}">{_fmt_money(data['day_pnl'])}</div>
      <div class="metric-sub" style="color:{day_pnl_color}">{_fmt_pct(data['day_pnl_pct'])}</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Current Drawdown</div>
      <div class="metric-value" style="color:{dd_color}">{_fmt_pct(data['current_drawdown'])}</div>
      <div class="metric-sub" style="color:{COLOR_GRAY}">max: {_fmt_pct(data['max_drawdown'])}</div>
    </div>
  </div>

  <div class="card">
    <h3>NAV History</h3>
    <canvas id="navChart"></canvas>
  </div>

  <div class="card">
    <h3>Positions ({len(data['positions'])})</h3>
    <table>
      <thead><tr><th>Symbol</th><th style="text-align:right">Quantity</th>
      <th style="text-align:right">Avg Cost</th><th style="text-align:right">Value</th></tr></thead>
      <tbody>{positions_rows_html}</tbody>
    </table>
  </div>

  <div class="card">
    <h3>Open Orders ({len(data['open_orders'])}) &middot; {data['filled_orders_count']} filled &middot; {data['total_orders']} total</h3>
    <table>
      <thead><tr><th>Symbol</th><th>Side</th><th style="text-align:right">Qty</th><th>Status</th></tr></thead>
      <tbody>{orders_rows_html}</tbody>
    </table>
  </div>

  <div class="footer">
    Rendered from persisted portfolio state (infra.persistence.PortfolioStateStore).
    This is a point-in-time snapshot, not a live-ticking view -- regenerate after each
    rebalance cycle to refresh. No forward-looking performance is implied or guaranteed.
  </div>
</div>

<script>
  const navData = {nav_json};
  const ctx = document.getElementById('navChart').getContext('2d');
  const labels = navData.map(p => p.t.slice(0, 10));
  const values = navData.map(p => p.nav);
  const startValue = values.length ? values[0] : 0;

  new Chart(ctx, {{
    type: 'line',
    data: {{
      labels: labels,
      datasets: [{{
        label: 'NAV',
        data: values,
        borderColor: '{COLOR_BLACK}',
        segment: {{
          borderColor: (c) => {{
            const y0 = c.p0.parsed.y, y1 = c.p1.parsed.y;
            if (y1 > startValue && y0 > startValue) return '{COLOR_GREEN}';
            if (y1 < startValue && y0 < startValue) return '{COLOR_RED}';
            return '{COLOR_BLACK}';
          }}
        }},
        pointRadius: 0,
        borderWidth: 2,
        fill: false,
        tension: 0.15,
      }}]
    }},
    options: {{
      responsive: true,
      plugins: {{ legend: {{ display: false }} }},
      scales: {{
        x: {{ grid: {{ display: false }}, ticks: {{ color: '{COLOR_GRAY}', maxTicksLimit: 10 }} }},
        y: {{ grid: {{ color: '{COLOR_BORDER}' }}, ticks: {{ color: '{COLOR_GRAY}' }} }}
      }}
    }}
  }});
</script>
</body>
</html>"""
    return html


def generate_dashboard(db_path: str, portfolio_id: str, output_path: str) -> str:
    from ..infra.persistence import PortfolioStateStore
    store = PortfolioStateStore(db_path)
    data = build_dashboard_data(store, portfolio_id)
    html = render_dashboard_html(data)
    with open(output_path, "w") as f:
        f.write(html)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate the ADC portfolio dashboard from persisted state")
    parser.add_argument("--db-path", default="live_portfolio.db")
    parser.add_argument("--portfolio-id", required=True)
    parser.add_argument("--output", default="dashboard.html")
    args = parser.parse_args()
    path = generate_dashboard(args.db_path, args.portfolio_id, args.output)
    print(f"Dashboard written to: {path}")


if __name__ == "__main__":
    main()
