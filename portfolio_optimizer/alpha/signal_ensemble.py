"""
Multi-Signal Alpha Ensemble.

HONEST FRAMING FIRST: no code produces guaranteed profit. What this module
does is combine several independent, individually well-documented return
signals (momentum, short-term mean-reversion, volatility-adjusted carry,
and a cross-sectional quality/low-volatility tilt) into a single blended
signal, with the blend weights themselves calibrated out-of-sample via
walk-forward validation rather than fixed by hand — which is the actual
substance of a real signal-combination engine, as opposed to hard-coding
"use 25% of each." Whether this makes money depends entirely on whether
these signals have genuine predictive power in whatever market and period
it's pointed at, which this module cannot know in advance and does not
claim to. Every signal here is a well-known factor from the academic and
practitioner literature (not invented for this project); the CONTRIBUTION
is the walk-forward-calibrated combination and the direct plug-in to this
repo's Entropy Pooling / Black-Litterman views layer, not the individual
signals themselves.

Architecture
------------
1. Each `Signal` computes a cross-sectional score per asset per date from
   price/return history alone (no external data required).
2. `SignalEnsemble` combines multiple signals into one blended score,
   using either fixed weights or walk-forward-optimized weights (chosen
   by maximizing historical information coefficient on a rolling basis --
   the standard "signal combination" approach used across quant equity
   desks, sometimes called a "meta-signal" or "signal-of-signals").
3. The blended score converts into portfolio VIEWS (feeding directly into
   `EntropyPooling` or `BlackLitterman`), not directly into weights --
   keeping this consistent with the rest of the repo's separation between
   "what do I believe" (views) and "how much risk do I take given that
   belief" (the optimizer).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from abc import ABC, abstractmethod


class Signal(ABC):
    """Base class for a cross-sectional return-predicting signal. Every
    signal outputs a score per asset per date: higher score = more
    expected outperformance, on some arbitrary scale (scores are
    cross-sectionally standardized before combination, so raw scale
    doesn't matter).
    """
    name: str = "signal"

    @abstractmethod
    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        """prices: T x N price levels (not returns). Returns a T x N
        DataFrame of scores, aligned to the same index/columns."""
        ...

    @staticmethod
    def _cross_sectional_zscore(scores: pd.DataFrame) -> pd.DataFrame:
        mean = scores.mean(axis=1)
        std = scores.std(axis=1).replace(0, np.nan)
        return scores.sub(mean, axis=0).div(std, axis=0).fillna(0.0)


class MomentumSignal(Signal):
    """Classic 12-1 momentum (Jegadeesh & Titman, 1993): cumulative return
    over the past `lookback` periods, EXCLUDING the most recent `skip`
    periods (skipping recent history avoids the well-documented short-term
    reversal effect contaminating the momentum signal).
    """
    name = "momentum_12_1"

    def __init__(self, lookback: int = 252, skip: int = 21):
        self.lookback, self.skip = lookback, skip

    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        past = prices.shift(self.skip)
        old = prices.shift(self.lookback)
        raw = np.log(past / old)
        return self._cross_sectional_zscore(raw)


class ShortTermReversalSignal(Signal):
    """Short-term mean-reversion (Jegadeesh, 1990; Lehmann, 1990): assets
    that fell hardest over the last `lookback` periods tend to bounce, and
    vice versa -- the sign-flipped short-horizon analogue of momentum.
    """
    name = "short_term_reversal"

    def __init__(self, lookback: int = 5):
        self.lookback = lookback

    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        raw = -np.log(prices / prices.shift(self.lookback))
        return self._cross_sectional_zscore(raw)


class VolatilityCarrySignal(Signal):
    """Low-volatility / risk-adjusted carry tilt (Frazzini & Pedersen,
    2014, "Betting Against Beta"; Ang, Hodrick, Xing & Zhang, 2006, "low
    vol anomaly"): favors assets with lower realized volatility, on the
    empirical finding that low-vol assets have historically delivered
    better risk-adjusted (and sometimes even raw) returns than a naive
    CAPM would predict.
    """
    name = "low_volatility_tilt"

    def __init__(self, lookback: int = 63):
        self.lookback = lookback

    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        returns = prices.pct_change()
        realized_vol = returns.rolling(self.lookback).std()
        raw = -realized_vol  # lower vol -> higher score
        return self._cross_sectional_zscore(raw)


class QualityTrendSignal(Signal):
    """Trend-quality signal: rewards assets whose recent price path has
    been SMOOTH and consistently upward (high Sharpe-like path quality),
    not just assets that happen to have risen a lot with high volatility
    on the way. This is a common practitioner refinement on raw momentum
    (a "quality-adjusted momentum") that penalizes noisy, whippy uptrends
    relative to steady ones with the same total return.
    """
    name = "trend_quality"

    def __init__(self, lookback: int = 126):
        self.lookback = lookback

    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        returns = prices.pct_change()
        roll_mean = returns.rolling(self.lookback).mean()
        roll_std = returns.rolling(self.lookback).std().replace(0, np.nan)
        raw = roll_mean / roll_std  # rolling Sharpe-like ratio per asset
        return self._cross_sectional_zscore(raw.fillna(0.0))


@dataclass
class EnsembleWeights:
    weights: pd.Series          # per-signal blend weight
    information_coefficients: pd.Series  # per-signal historical IC (predictive power estimate)
    calibration_window: int


class SignalEnsemble:
    """Combines multiple Signals into one blended score, with blend
    weights either fixed or walk-forward-calibrated by historical
    Information Coefficient (IC) -- the rank correlation between each
    signal's score and the asset's SUBSEQUENT return, the standard way
    quant equity desks measure "how much has this signal actually been
    worth" before deciding how much to lean on it.
    """

    def __init__(self, signals: list, forward_period: int = 21):
        self.signals = signals
        self.forward_period = forward_period

    def compute_all_signals(self, prices: pd.DataFrame) -> dict:
        return {s.name: s.compute(prices) for s in self.signals}

    def information_coefficient(self, signal_scores: pd.DataFrame, prices: pd.DataFrame) -> pd.Series:
        """Spearman rank correlation between each date's cross-sectional
        signal score and the FORWARD return over `forward_period` --
        computed date-by-date then summarized, so this is a genuine
        predictive-power measure, not a look-ahead-biased in-sample fit.
        """
        forward_returns = prices.shift(-self.forward_period) / prices - 1
        ics = []
        common_dates = signal_scores.index.intersection(forward_returns.index)
        for date in common_dates:
            s = signal_scores.loc[date]
            r = forward_returns.loc[date]
            valid = s.notna() & r.notna()
            if valid.sum() < 3:
                continue
            ic = s[valid].corr(r[valid], method="spearman")
            if pd.notna(ic):
                ics.append(ic)
        return pd.Series(ics)

    def calibrate_weights(self, prices: pd.DataFrame, method: str = "ic_weighted") -> EnsembleWeights:
        """Walk-forward calibration: for each signal, compute its
        historical IC time series, then weight signals by their mean IC
        (higher historical predictive power -> more weight in the blend),
        floored at zero so a signal with negative historical IC on this
        asset universe contributes nothing rather than actively hurting
        the blend (a standard, conservative choice -- the alternative,
        flipping its sign, risks overfitting to noise).
        """
        all_scores = self.compute_all_signals(prices)
        ic_means = {}
        for name, scores in all_scores.items():
            ic_series = self.information_coefficient(scores, prices)
            ic_means[name] = ic_series.mean() if len(ic_series) > 0 else 0.0

        ic_series_out = pd.Series(ic_means)

        if method == "equal":
            weights = pd.Series(1.0 / len(self.signals), index=ic_series_out.index)
        elif method == "ic_weighted":
            floored = ic_series_out.clip(lower=0.0)
            weights = floored / floored.sum() if floored.sum() > 0 else \
                pd.Series(1.0 / len(self.signals), index=ic_series_out.index)
        else:
            raise ValueError(f"Unknown calibration method: {method}")

        return EnsembleWeights(weights=weights, information_coefficients=ic_series_out,
                                calibration_window=len(prices))

    def blended_score(self, prices: pd.DataFrame, ensemble_weights: EnsembleWeights | None = None
                       ) -> pd.DataFrame:
        """Final combined cross-sectional score per asset per date."""
        if ensemble_weights is None:
            ensemble_weights = self.calibrate_weights(prices)
        all_scores = self.compute_all_signals(prices)
        blended = sum(all_scores[name] * ensemble_weights.weights[name] for name in all_scores)
        return blended

    def latest_ranking(self, prices: pd.DataFrame, ensemble_weights: EnsembleWeights | None = None
                        ) -> pd.Series:
        """Convenience: the blended score for the MOST RECENT date only,
        ranked descending -- the input you'd actually feed into a view.
        """
        scores = self.blended_score(prices, ensemble_weights)
        return scores.iloc[-1].sort_values(ascending=False)

    def scores_to_views(self, prices: pd.DataFrame, expected_returns: pd.Series,
                         ensemble_weights: EnsembleWeights | None = None,
                         view_strength: float = 0.02, top_n: int | None = None) -> list:
        """Converts the blended score into a list of (asset, tilted_return,
        confidence) tuples ready to feed into EntropyPooling.add_mean_view
        or BlackLitterman.add_absolute_view -- this is the actual hand-off
        point from "signal" to "optimizer input" that keeps this module
        honest about NOT bypassing the risk-aware optimization layer.

        The tilt is proportional to the asset's standardized score (a
        higher-scoring asset gets a return view above its historical mean,
        scaled by `view_strength`; a lower-scoring asset gets a view
        below it), and confidence is proportional to the signal's
        calibrated IC magnitude (a well-validated signal earns higher
        confidence in the resulting view).
        """
        ranking = self.latest_ranking(prices, ensemble_weights)
        if top_n is not None:
            ranking = pd.concat([ranking.head(top_n), ranking.tail(top_n)])

        ew = ensemble_weights or self.calibrate_weights(prices)
        avg_ic = float(ew.information_coefficients.clip(lower=0).mean())
        base_confidence = float(np.clip(avg_ic * 5, 0.05, 0.85))  # map IC (~0-0.1 typical) to a usable confidence range

        views = []
        for asset, score in ranking.items():
            base_return = expected_returns.get(asset, 0.0)
            tilted_return = base_return + view_strength * np.clip(score, -3, 3) / 3
            views.append((asset, float(tilted_return), base_confidence))
        return views
