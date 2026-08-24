"""
Explainability Dashboard.

Directly answers the question every allocator/investor eventually asks:
"why did the optimizer put 18% into gold?" This module decomposes that
decision into its interpretable ingredients rather than leaving the
optimizer as a black box:

  - **Posterior returns**: what expected return did the model actually use
    for this asset, and how does it compare to the raw historical mean?
  - **Confidence levels**: for Black-Litterman, how much did each view's
    stated confidence actually matter to the outcome (a view with low
    confidence barely moves the posterior, no matter how extreme).
  - **Covariance effects**: how much of the weight is explained by this
    asset's diversification value (its correlation profile) versus its
    raw expected return.
  - **View contributions**: for BL specifically, how much of the shift
    from the prior to the posterior return is attributable to *each*
    individual view (since views interact through the shared covariance
    structure, this uses a views-added-one-at-a-time marginal decomposition).

Produces both a programmatic result object (for further analysis / your
own UI) and a self-contained HTML report.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from ..optimizers.black_litterman import BlackLitterman, View
from ..optimizers.markowitz import MarkowitzOptimizer


@dataclass
class AssetExplanation:
    asset: str
    final_weight: float
    prior_return: float
    posterior_return: float
    historical_mean_return: float
    return_shift_from_views: float
    marginal_risk_contribution_pct: float
    diversification_score: float          # negative avg correlation to rest of portfolio -> higher = more diversifying
    view_contributions: dict              # {view_description: contribution to this asset's posterior return shift}


@dataclass
class ExplainabilityReport:
    asset_explanations: dict  # asset -> AssetExplanation
    summary_table: pd.DataFrame


class BlackLittermanExplainer:
    """Wraps a fitted BlackLitterman model (with its views already added)
    plus the resulting optimized weights, and decomposes *why* the
    optimizer landed on those weights.
    """

    def __init__(self, bl_model: BlackLitterman, historical_returns: pd.DataFrame,
                 final_weights: pd.Series):
        self.bl = bl_model
        self.returns = historical_returns
        self.weights = final_weights.reindex(bl_model.assets).fillna(0.0)
        self.prior = bl_model.implied_prior()
        self.post_mu, self.post_cov = bl_model.posterior()
        self.hist_mean = historical_returns.mean() * 252

    def _view_marginal_contributions(self) -> pd.DataFrame:
        """Add views one at a time (in the order they were specified) and
        record how much each incremental view shifts each asset's
        posterior mean — a Shapley-style marginal decomposition (order-
        dependent, but transparent and cheap; fine for the handful of
        views typical of a real BL setup).
        """
        assets = self.bl.assets
        contributions = pd.DataFrame(0.0, index=assets, columns=[f"view_{i}" for i in range(len(self.bl.views))])
        running = BlackLitterman(pd.DataFrame(self.bl.cov, index=assets, columns=assets),
                                  prior_returns=self.prior, risk_aversion=self.bl.delta, tau=self.bl.tau)
        prev_mu = self.prior.values.copy()
        for i, v in enumerate(self.bl.views):
            running.add_view(v)
            new_mu, _ = running.posterior()
            contributions[f"view_{i}"] = new_mu.values - prev_mu
            prev_mu = new_mu.values
        return contributions

    def _view_description(self, v: View) -> str:
        if len(v.assets) == 1:
            return f"{v.assets[0]} absolute view = {v.value:.2%} (conf={v.confidence:.0%})"
        return f"{v.assets[0]} vs {v.assets[1]} relative view = {v.value:.2%} (conf={v.confidence:.0%})"

    def explain(self) -> ExplainabilityReport:
        assets = self.bl.assets
        view_contribs = self._view_marginal_contributions()
        view_descriptions = [self._view_description(v) for v in self.bl.views]

        w = self.weights.values
        cov = self.post_cov.reindex(index=assets, columns=assets).values
        port_var = w @ cov @ w
        port_vol = np.sqrt(max(port_var, 1e-16))
        marginal = (cov @ w) / port_vol
        component_pct = (w * marginal) / port_vol

        corr = self.returns.corr()
        avg_corr_to_others = {}
        for a in assets:
            others = [x for x in assets if x != a]
            avg_corr_to_others[a] = corr.loc[a, others].mean() if others else 0.0

        explanations = {}
        for i, a in enumerate(assets):
            view_contrib_dict = {view_descriptions[j]: float(view_contribs.iloc[i, j])
                                  for j in range(len(self.bl.views))}
            explanations[a] = AssetExplanation(
                asset=a, final_weight=float(self.weights[a]),
                prior_return=float(self.prior[a]), posterior_return=float(self.post_mu[a]),
                historical_mean_return=float(self.hist_mean.get(a, np.nan)),
                return_shift_from_views=float(self.post_mu[a] - self.prior[a]),
                marginal_risk_contribution_pct=float(component_pct[i]),
                diversification_score=float(-avg_corr_to_others[a]),
                view_contributions=view_contrib_dict,
            )

        summary = pd.DataFrame({
            "weight": self.weights,
            "prior_return": self.prior,
            "posterior_return": self.post_mu,
            "historical_mean": self.hist_mean.reindex(assets),
            "view_shift": self.post_mu - self.prior,
            "risk_contribution_pct": pd.Series(component_pct, index=assets),
            "diversification_score": pd.Series(avg_corr_to_others, index=assets).apply(lambda x: -x),
        })
        return ExplainabilityReport(asset_explanations=explanations, summary_table=summary)

    def explain_asset_in_words(self, asset: str) -> str:
        """Natural-language explanation for a single asset — the direct
        answer to "why did Black-Litterman allocate X% to gold?"
        """
        report = self.explain()
        e = report.asset_explanations[asset]
        lines = [
            f"{asset} received a final weight of {e.final_weight:.1%}.",
            f"  - Equilibrium (prior) expected return: {e.prior_return:.2%}",
            f"  - Posterior (view-adjusted) expected return: {e.posterior_return:.2%} "
            f"({'+' if e.return_shift_from_views >= 0 else ''}{e.return_shift_from_views:.2%} from views)",
            f"  - Trailing historical mean return: {e.historical_mean_return:.2%}",
            f"  - Contributes {e.marginal_risk_contribution_pct:.1%} of total portfolio risk",
            f"  - Diversification score: {e.diversification_score:+.2f} "
            f"({'diversifying' if e.diversification_score > 0 else 'concentrating'} relative to the rest of the book)",
        ]
        if e.view_contributions:
            lines.append("  - View contributions to its posterior return shift:")
            for desc, contrib in e.view_contributions.items():
                if abs(contrib) > 1e-6:
                    lines.append(f"      * {desc}: {contrib:+.2%}")
        else:
            lines.append("  - No views directly referenced this asset; its posterior return "
                         "moved only through shared covariance with assets that were viewed.")
        return "\n".join(lines)

    def to_html(self, path: str) -> str:
        report = self.explain()
        rows = ""
        for asset, e in report.asset_explanations.items():
            view_rows = "".join(
                f"<li>{desc}: <b>{contrib:+.2%}</b></li>"
                for desc, contrib in e.view_contributions.items() if abs(contrib) > 1e-6
            ) or "<li><i>No direct view on this asset</i></li>"
            rows += f"""
            <div class="asset-card">
              <h3>{asset} — {e.final_weight:.1%} weight</h3>
              <table>
                <tr><td>Prior (equilibrium) return</td><td>{e.prior_return:.2%}</td></tr>
                <tr><td>Posterior return</td><td>{e.posterior_return:.2%}</td></tr>
                <tr><td>Shift from views</td><td>{e.return_shift_from_views:+.2%}</td></tr>
                <tr><td>Historical mean return</td><td>{e.historical_mean_return:.2%}</td></tr>
                <tr><td>Risk contribution</td><td>{e.marginal_risk_contribution_pct:.1%}</td></tr>
                <tr><td>Diversification score</td><td>{e.diversification_score:+.2f}</td></tr>
              </table>
              <b>View contributions:</b>
              <ul>{view_rows}</ul>
            </div>"""

        html = f"""
<!DOCTYPE html><html><head><meta charset="utf-8"><title>Explainability Report</title>
<style>
  body {{ font-family: -apple-system, 'Segoe UI', sans-serif; background:#f7f7f5; padding:32px; }}
  .asset-card {{ background:white; border-radius:10px; padding:18px 22px; margin-bottom:16px;
                 box-shadow:0 1px 3px rgba(0,0,0,0.08); max-width:720px; }}
  table {{ border-collapse:collapse; width:100%; margin:10px 0; }}
  td {{ padding:6px 8px; border-bottom:1px solid #eee; font-size:13.5px; }}
  td:first-child {{ color:#555; }} td:last-child {{ text-align:right; font-weight:600; }}
  h1 {{ font-size:22px; }} h3 {{ margin-bottom:4px; }}
  ul {{ font-size:13px; color:#333; }}
</style></head><body>
<h1>Portfolio Explainability Report</h1>
<p style="color:#666; font-size:13px;">Black-Litterman decision breakdown — posterior returns, view contributions, and risk/diversification effects per asset.</p>
{rows}
</body></html>"""
        with open(path, "w") as f:
            f.write(html)
        return path
