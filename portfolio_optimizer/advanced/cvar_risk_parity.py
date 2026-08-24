"""
NEW #2 — CVaR Risk Parity.

Classic risk parity (ERC) equalizes each asset's contribution to portfolio
*variance*. But variance treats a quiet grind-up and a violent crash
identically. This module instead equalizes each asset's contribution to
portfolio *CVaR* (expected shortfall) — i.e. every asset contributes the
same amount to the fund's actual tail-loss exposure, which is a materially
different (and arguably more honest) definition of "equal risk" for a
fund that lives or dies by its drawdowns.

CVaR risk contributions don't have a closed form the way variance
contributions do, so this is solved by:
  1. Computing scenario-based CVaR via the Rockafellar-Uryasev auxiliary
     variables (VaR threshold zeta, tail indicator).
  2. Using Euler's theorem for positively homogeneous risk measures: since
     CVaR(w) is homogeneous of degree 1 in w, the marginal contribution of
     asset i is w_i * dCVaR/dw_i, and these sum exactly to total CVaR.
     The gradient of CVaR w.r.t. w has the closed form
        dCVaR/dw = -E[R_t | portfolio loss exceeds VaR]
     i.e. the average of asset returns *conditional on* the portfolio being
     in its tail scenarios — computed directly from the scenario set.
  3. Numerically solving for weights that equalize these contributions.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from dataclasses import dataclass


@dataclass
class CVaRRiskParityResult:
    weights: pd.Series
    cvar_contributions: pd.Series
    total_cvar: float
    success: bool


class CVaRRiskParity:
    def __init__(self, returns: pd.DataFrame, alpha: float = 0.95,
                 risk_budget: pd.Series | None = None, weight_bounds: tuple = (0.001, 1.0)):
        self.returns = returns
        self.assets = list(returns.columns)
        self.R = returns.values
        self.T, self.N = self.R.shape
        self.alpha = alpha
        self.bounds = weight_bounds
        if risk_budget is None:
            self.budget = np.ones(self.N) / self.N
        else:
            b = risk_budget.reindex(self.assets).values
            self.budget = b / b.sum()

    def _portfolio_cvar_and_contributions(self, w):
        port_losses = -(self.R @ w)  # loss = negative return
        var_threshold = np.quantile(port_losses, self.alpha)
        tail_mask = port_losses >= var_threshold
        n_tail = max(tail_mask.sum(), 1)
        cvar = port_losses[tail_mask].mean()
        # marginal contribution: -E[R_i | tail]
        marginal = -self.R[tail_mask].mean(axis=0)
        contributions = w * marginal
        # rescale contributions to sum exactly to cvar (Euler decomposition)
        scale = cvar / contributions.sum() if contributions.sum() != 0 else 1.0
        contributions = contributions * scale
        return cvar, contributions

    def solve(self, max_iter: int = 300) -> CVaRRiskParityResult:
        def objective(w):
            w = w / w.sum()
            cvar, contrib = self._portfolio_cvar_and_contributions(w)
            target = cvar * self.budget
            return np.sum((contrib - target) ** 2)

        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [self.bounds] * self.N
        x0 = np.ones(self.N) / self.N
        result = minimize(objective, x0, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": max_iter, "ftol": 1e-14})
        w = np.clip(result.x, 0, None)
        w = w / w.sum()
        cvar, contrib = self._portfolio_cvar_and_contributions(w)
        return CVaRRiskParityResult(
            weights=pd.Series(w, index=self.assets, name="weight"),
            cvar_contributions=pd.Series(contrib, index=self.assets, name="cvar_contribution"),
            total_cvar=cvar, success=bool(result.success),
        )
