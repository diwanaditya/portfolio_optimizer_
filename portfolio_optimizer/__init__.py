"""
Portfolio Optimizer — Institutional-Grade Asset Allocation Engine
===================================================================

A production-ready portfolio construction toolkit built for ADC
(Aditya Diwan Capital) style systematic / quantitative workflows.

Core engines
------------
- Mean-Variance (Markowitz) optimization + full efficient frontier
- Black-Litterman with investor views
- Risk Parity: Equal Risk Contribution (ERC) + Hierarchical Risk Parity (HRP)
- CVaR / CDaR optimization (Rockafellar-Uryasev LP formulation)

Research-grade extensions (the "legend" layer)
-----------------------------------------------
1. Entropy Pooling          — generalized view-blending beyond BL's linear views
2. CVaR Risk Parity         — equalizes *tail-risk* contribution, not variance
3. Regime-Switching Overlay — HMM-based bull/bear detection, regime-conditional allocation
4. Robust / Resampled Frontier (Michaud) — Monte Carlo resampling to kill estimation-error noise
5. Factor Risk Model        — statistical (PCA) or fundamental factor exposure constraints
6. Walk-Forward Backtester  — realistic, cost-aware, look-ahead-bias-free simulation
7. Stress Testing Engine    — historical scenarios + Student-t fat-tailed Monte Carlo

Everything operates on plain pandas DataFrames of returns, so it plugs into
whatever data pipeline you already have (ADC's own feeds, CSVs, a DB, etc).
"""

from .estimators.expected_returns import (
    mean_historical_return,
    ewma_return,
    capm_return,
)
from .estimators.covariance import (
    sample_covariance,
    ledoit_wolf_shrinkage,
    ewma_covariance,
    semicovariance,
)
from .optimizers.markowitz import MarkowitzOptimizer
from .optimizers.black_litterman import BlackLitterman
from .optimizers.risk_parity import RiskParity, HierarchicalRiskParity
from .optimizers.cvar import CVaROptimizer

from .advanced.entropy_pooling import EntropyPooling
from .advanced.cvar_risk_parity import CVaRRiskParity
from .advanced.regime_switching import RegimeSwitchingOverlay
from .advanced.robust_frontier import ResampledEfficientFrontier
from .advanced.factor_risk_model import FactorRiskModel
from .advanced.stress_testing import StressTester

from .backtester import WalkForwardBacktester
from .reporting.tearsheet import Tearsheet

from .bayesian.bayesian_mean_variance import BayesianMeanVariance, bayes_stein_shrinkage
from .bayesian.hierarchical_bayes import HierarchicalBayesianPortfolio

from .multiperiod.multi_period_optimizer import LiNgMultiPeriod, ScenarioMPCOptimizer
from .execution.almgren_chriss import optimal_execution_trajectory, AlmgrenChrissCostModel
from .data.adapters import get_adapter, LiveDataAdapter

from .attribution.brinson import brinson_attribution, multi_period_brinson
from .attribution.factor_attribution import factor_attribution, factor_attribution_over_backtest
from .attribution.risk_attribution import variance_risk_attribution, factor_risk_attribution

from .explainability.dashboard import BlackLittermanExplainer
from .gpu.accel import available_backends, best_backend, GPUAcceleratedMonteCarlo, GPUAcceleratedResampling

__version__ = "3.0.0"

# Note: portfolio_optimizer.research, .benchmarking, .validation, and
# .infra are intentionally NOT imported into the top-level namespace.
# They pull in heavier/optional dependencies (pypfopt, riskfolio-lib for
# benchmarking) or are meant to be used as standalone toolkits
# (`from portfolio_optimizer.validation.bootstrap_ci import ...`,
# `from portfolio_optimizer.infra.oms import OrderManager`, etc.) rather
# than always-loaded core functionality.

__all__ = [
    "mean_historical_return", "ewma_return", "capm_return",
    "sample_covariance", "ledoit_wolf_shrinkage", "ewma_covariance", "semicovariance",
    "MarkowitzOptimizer", "BlackLitterman", "RiskParity", "HierarchicalRiskParity",
    "CVaROptimizer",
    "EntropyPooling", "CVaRRiskParity", "RegimeSwitchingOverlay",
    "ResampledEfficientFrontier", "FactorRiskModel", "StressTester",
    "WalkForwardBacktester", "Tearsheet",
    # v2.0 additions
    "BayesianMeanVariance", "bayes_stein_shrinkage", "HierarchicalBayesianPortfolio",
    "LiNgMultiPeriod", "ScenarioMPCOptimizer",
    "optimal_execution_trajectory", "AlmgrenChrissCostModel",
    "get_adapter", "LiveDataAdapter",
    "brinson_attribution", "multi_period_brinson",
    "factor_attribution", "factor_attribution_over_backtest",
    "variance_risk_attribution", "factor_risk_attribution",
    "BlackLittermanExplainer",
    "available_backends", "best_backend", "GPUAcceleratedMonteCarlo", "GPUAcceleratedResampling",
]
