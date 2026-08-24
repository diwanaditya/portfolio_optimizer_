"""
General Weight-Change Explainer.

`BlackLittermanExplainer` (dashboard.py's sibling module) explains a BL
posterior in terms of views and priors. This module explains something
more general and more commonly needed: "why did the optimizer's output
change between two runs" -- comparing any two weight vectors (before vs
after a rebalance, this month vs last month, pre- vs post-view) and
attributing the change to the three things that actually move a
mean-variance-style optimizer's answer:

    1. Expected return changed for this asset
    2. This asset's volatility changed
    3. This asset's correlation to the REST of the portfolio changed
       (the diversification-value channel -- an asset whose correlation
       to everything else drops becomes more attractive on relative
       terms even with an unchanged expected return)

Produces the exact style of explanation requested: "Weight in MSFT
increased because expected return rose while correlation with the
portfolio declined" -- generated from the actual numbers, not a
template with blanks filled in from nowhere.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class WeightChangeExplanation:
    asset: str
    weight_before: float
    weight_after: float
    weight_change: float
    return_before: float
    return_after: float
    return_change: float
    volatility_before: float
    volatility_after: float
    volatility_change: float
    portfolio_correlation_before: float
    portfolio_correlation_after: float
    portfolio_correlation_change: float
    narrative: str


class WeightChangeExplainer:
    def __init__(self, returns_before: pd.DataFrame, returns_after: pd.DataFrame,
                 weights_before: pd.Series, weights_after: pd.Series,
                 periods_per_year: int = 252):
        self.returns_before = returns_before
        self.returns_after = returns_after
        self.weights_before = weights_before
        self.weights_after = weights_after
        self.ppy = periods_per_year
        self.assets = sorted(set(weights_before.index) | set(weights_after.index))

    def _portfolio_correlation(self, returns: pd.DataFrame, weights: pd.Series, asset: str) -> float:
        """Correlation of `asset`'s returns with the REST of the portfolio
        (weighted by everyone else's weight) -- the diversification-value
        signal. Excludes the asset itself so this doesn't just measure
        self-correlation.
        """
        others = [a for a in weights.index if a != asset and a in returns.columns]
        if not others or asset not in returns.columns:
            return np.nan
        other_weights = weights[others]
        if other_weights.abs().sum() < 1e-9:
            return np.nan
        other_weights = other_weights / other_weights.abs().sum()
        rest_of_portfolio_returns = returns[others] @ other_weights
        if rest_of_portfolio_returns.std() == 0 or returns[asset].std() == 0:
            return np.nan
        return float(returns[asset].corr(rest_of_portfolio_returns))

    def _build_narrative(self, asset: str, weight_change: float, return_change: float,
                          vol_change: float, corr_change: float,
                          weight_before: float, weight_after: float) -> str:
        direction = "increased" if weight_change > 1e-6 else ("decreased" if weight_change < -1e-6 else "stayed roughly flat")

        if abs(weight_change) < 1e-6:
            return f"Weight in {asset} stayed roughly flat ({weight_before:.1%} -> {weight_after:.1%})."

        reasons = []
        # A reason "supports" the direction of weight change if it points
        # the same way: rising return + falling correlation both push
        # weight UP; falling return + rising correlation both push weight DOWN.
        return_supports = (return_change > 0) == (weight_change > 0)
        corr_supports = (corr_change < 0) == (weight_change > 0)  # lower correlation -> more attractive
        vol_supports = (vol_change < 0) == (weight_change > 0)     # lower vol -> more attractive

        if not np.isnan(return_change) and abs(return_change) > 1e-4 and return_supports:
            reasons.append(f"expected return {'rose' if return_change > 0 else 'fell'} "
                            f"({return_change:+.2%})")
        if not np.isnan(corr_change) and abs(corr_change) > 0.02 and corr_supports:
            reasons.append(f"correlation with the rest of the portfolio "
                            f"{'declined' if corr_change < 0 else 'rose'} "
                            f"({corr_change:+.2f})")
        if not np.isnan(vol_change) and abs(vol_change) > 0.01 and vol_supports:
            reasons.append(f"volatility {'fell' if vol_change < 0 else 'rose'} ({vol_change:+.2%})")

        weight_text = f"Weight in {asset} {direction} ({weight_before:.1%} -> {weight_after:.1%})"
        if not reasons:
            return (f"{weight_text}. No single driver dominates -- this may reflect the "
                    f"optimizer redistributing weight elsewhere in the portfolio rather than "
                    f"a change specific to {asset}.")
        if len(reasons) == 1:
            return f"{weight_text} because {reasons[0]}."
        return f"{weight_text} because {reasons[0]} while {' and '.join(reasons[1:])}."

    def explain_asset(self, asset: str) -> WeightChangeExplanation:
        wb = float(self.weights_before.get(asset, 0.0))
        wa = float(self.weights_after.get(asset, 0.0))

        rb = float(self.returns_before[asset].mean() * self.ppy) if asset in self.returns_before.columns else np.nan
        ra = float(self.returns_after[asset].mean() * self.ppy) if asset in self.returns_after.columns else np.nan

        vb = float(self.returns_before[asset].std() * np.sqrt(self.ppy)) if asset in self.returns_before.columns else np.nan
        va = float(self.returns_after[asset].std() * np.sqrt(self.ppy)) if asset in self.returns_after.columns else np.nan

        cb = self._portfolio_correlation(self.returns_before, self.weights_before, asset)
        ca = self._portfolio_correlation(self.returns_after, self.weights_after, asset)

        return_change = ra - rb if not (np.isnan(ra) or np.isnan(rb)) else np.nan
        vol_change = va - vb if not (np.isnan(va) or np.isnan(vb)) else np.nan
        corr_change = ca - cb if not (np.isnan(ca) or np.isnan(cb)) else np.nan

        narrative = self._build_narrative(asset, wa - wb, return_change, vol_change, corr_change, wb, wa)

        return WeightChangeExplanation(
            asset=asset, weight_before=wb, weight_after=wa, weight_change=wa - wb,
            return_before=rb, return_after=ra, return_change=return_change,
            volatility_before=vb, volatility_after=va, volatility_change=vol_change,
            portfolio_correlation_before=cb, portfolio_correlation_after=ca,
            portfolio_correlation_change=corr_change, narrative=narrative,
        )

    def explain_all(self, top_n: int | None = None) -> list:
        """Returns explanations for every asset, sorted by magnitude of
        weight change (biggest movers first) -- the most useful default
        ordering for a report or dashboard panel.
        """
        explanations = [self.explain_asset(a) for a in self.assets]
        explanations.sort(key=lambda e: abs(e.weight_change), reverse=True)
        if top_n is not None:
            explanations = explanations[:top_n]
        return explanations

    def summary_table(self) -> pd.DataFrame:
        rows = [e.__dict__ for e in self.explain_all()]
        return pd.DataFrame(rows).set_index("asset")
