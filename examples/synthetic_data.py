"""
Synthetic multi-asset return generator for tests and demos.

Simulates a small universe with realistic structure: a common market factor,
sector correlation clusters, fat tails via a Student-t innovation, and a
volatility regime switch partway through — so the regime-switching,
factor-model, and stress-testing modules all have something real to find.
"""
import numpy as np
import pandas as pd


def generate_synthetic_universe(n_days: int = 1000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    assets = ["US_EQUITY", "INTL_EQUITY", "EM_EQUITY", "GOVT_BONDS",
              "CORP_BONDS", "GOLD", "COMMODITIES", "REIT"]
    n = len(assets)

    # base annualized vols and market betas per asset (rough real-world flavor)
    annual_vol = np.array([0.16, 0.18, 0.24, 0.06, 0.08, 0.15, 0.20, 0.19])
    market_beta = np.array([1.00, 0.95, 1.15, -0.10, 0.05, 0.02, 0.35, 0.85])
    annual_mu = np.array([0.09, 0.07, 0.08, 0.03, 0.045, 0.05, 0.04, 0.075])

    daily_vol = annual_vol / np.sqrt(252)
    daily_mu = annual_mu / 252

    dates = pd.bdate_range("2021-01-04", periods=n_days)

    # regime: calm for first 55%, stressed for a middle crisis block, calm after
    regime_vol_mult = np.ones(n_days)
    crisis_start, crisis_end = int(n_days * 0.55), int(n_days * 0.62)
    regime_vol_mult[crisis_start:crisis_end] = 2.0

    # Student-t innovations, standardized to unit variance before scaling so
    # df doesn't distort the target volatility (t-distribution has variance
    # df/(df-2), which we normalize out here).
    df_market, df_idio = 7, 8
    t_scale_market = np.sqrt(df_market / (df_market - 2))
    t_scale_idio = np.sqrt(df_idio / (df_idio - 2))

    market_factor = (rng.standard_t(df=df_market, size=n_days) / t_scale_market) * 0.010 * regime_vol_mult
    idio = ((rng.standard_t(df=df_idio, size=(n_days, n)) / t_scale_idio)
            * daily_vol[None, :] * regime_vol_mult[:, None] * 0.7)

    returns = daily_mu[None, :] + market_beta[None, :] * market_factor[:, None] + idio
    # modest additional correlated drag during the crisis block, like a real drawdown
    returns[crisis_start:crisis_end] -= 0.0008

    df = pd.DataFrame(returns, index=dates, columns=assets)
    return df


def generate_market_caps(assets: list[str], seed: int = 3) -> pd.Series:
    rng = np.random.default_rng(seed)
    raw = rng.uniform(50, 500, size=len(assets))
    return pd.Series(raw, index=assets, name="market_cap_bn")
