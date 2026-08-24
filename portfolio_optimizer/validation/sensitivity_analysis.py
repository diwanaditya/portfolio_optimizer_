"""
Sensitivity Analysis.

Every optimizer in this repo has hyperparameters (risk aversion, shrinkage
intensity, CVaR alpha, lookback window, rebalance frequency...) whose
values are typically chosen somewhat arbitrarily. This module
systematically sweeps those parameters and reports how sensitive the
resulting portfolio (weights, Sharpe, volatility) is to each — a
one-line "Sharpe = 1.4" is much less informative than "Sharpe ranges
from 1.1 to 1.6 as risk_aversion varies from 1 to 5, and the ranking
between our two candidate strategies is unaffected."
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Callable


@dataclass
class SensitivityResult:
    parameter_name: str
    parameter_values: list
    output_values: pd.DataFrame     # one row per parameter value, one column per output metric
    output_range: pd.Series         # max - min per metric, across the swept parameter values
    output_relative_sensitivity: pd.Series  # (max-min)/|mean| per metric -- normalized sensitivity


def sweep_parameter(param_values: list, run_fn: Callable[[float], dict],
                     parameter_name: str = "parameter") -> SensitivityResult:
    """Generic parameter sweep: `run_fn(value)` should return a dict of
    {metric_name: value} for a single choice of the swept parameter.
    """
    rows = []
    for v in param_values:
        try:
            metrics = run_fn(v)
        except Exception as e:
            metrics = {}
        row = {parameter_name: v, **metrics}
        rows.append(row)
    df = pd.DataFrame(rows).set_index(parameter_name)

    output_range = df.max() - df.min()
    mean_abs = df.abs().mean().replace(0, np.nan)
    relative_sensitivity = (output_range / mean_abs).fillna(0.0)

    return SensitivityResult(parameter_name=parameter_name, parameter_values=param_values,
                              output_values=df, output_range=output_range,
                              output_relative_sensitivity=relative_sensitivity)


def multi_parameter_sensitivity_report(sweeps: dict) -> pd.DataFrame:
    """Combine multiple SensitivityResult objects (one per swept
    parameter) into a single tornado-chart-style summary table, sorted by
    which parameter the outputs are most sensitive to.
    """
    rows = []
    for param_name, result in sweeps.items():
        for metric in result.output_relative_sensitivity.index:
            rows.append({
                "parameter": param_name, "metric": metric,
                "range": result.output_range[metric],
                "relative_sensitivity": result.output_relative_sensitivity[metric],
            })
    df = pd.DataFrame(rows).sort_values("relative_sensitivity", ascending=False)
    return df.reset_index(drop=True)


def weight_stability_across_sweep(param_values: list, weights_fn: Callable[[float], pd.Series],
                                    parameter_name: str = "parameter") -> pd.DataFrame:
    """Specifically track how much the PORTFOLIO WEIGHTS themselves (not
    just aggregate metrics) shift as a parameter varies -- e.g. does a
    small change in risk_aversion cause a small, smooth reallocation, or
    a discontinuous jump from one asset to a completely different one?
    Large jumps indicate the optimizer is sitting at an unstable corner
    solution, which is itself an important robustness finding.
    """
    rows = []
    for v in param_values:
        try:
            w = weights_fn(v)
            row = {parameter_name: v, **w.to_dict()}
        except Exception:
            row = {parameter_name: v}
        rows.append(row)
    return pd.DataFrame(rows).set_index(parameter_name)
