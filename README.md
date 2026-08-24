# ADC Optimization & Allocation Engine

A quantitative portfolio research and allocation project I built for systematic investing work at Aditya Diwan Capital (ADC).

The project started as a collection of portfolio optimizers and grew into a larger research platform with backtesting, risk analysis, data adapters, execution models, a paper-trading loop, an API, and supporting infrastructure.

It is still a research and engineering project. It is **not a hedge fund, investment vehicle, or finished production trading operation**.

---

## What is in the project

The main parts are:

- Portfolio optimization
- Risk estimation and portfolio risk controls
- Historical and walk-forward backtesting
- Statistical validation
- Stress testing
- Factor and regime analysis
- Bayesian portfolio methods
- Multi-period optimization
- Transaction-cost and execution models
- Market-data adapters
- Paper trading through Alpaca
- Order-management and broker abstractions
- FastAPI service
- SaaS/tenant and billing scaffolding
- Portfolio attribution and explainability
- HTML reports and dashboards
- Optional GPU acceleration

The code works primarily with `pandas` DataFrames, so the individual optimizers can also be used without running the API or trading components.

## Quick start

```bash
pip install -r requirements.txt

python examples/full_demo.py
python examples/v2_demo.py
python examples/novel_contribution_validation.py
python examples/benchmark_report.py
```

For the development workflow:

```bash
make install
make test
```

The test suite is intended to be run from a clean environment with the project dependencies installed.

---

## Project layout

```text
portfolio_optimizer/
├── optimizers/       Core portfolio construction methods
├── estimators/       Return and covariance estimators
├── advanced/         CVaR, regimes, factors, stress tests, etc.
├── bayesian/         Bayesian portfolio methods
├── multiperiod/      Multi-period portfolio optimization
├── execution/        Transaction-cost and execution models
├── backtester.py     Walk-forward backtesting
├── validation/       Statistical tests and robustness checks
├── data/             Market-data adapters
├── live/             Paper-trading loop and live-data validation
├── infra/            OMS, audit log, persistence, risk, fault tolerance
├── api/              FastAPI service
├── saas/             Tenant and billing components
├── explainability/   Portfolio and weight-change explanations
├── reporting/        HTML tearsheets and reports
├── dashboard/        Dashboard components
├── gpu/              Optional GPU acceleration
├── rl/               Experimental RL portfolio management
├── examples/         Runnable examples and research checks
└── tests/            Automated tests
```

---

# 1. Portfolio optimizers

### Markowitz Mean-Variance

`optimizers/markowitz.py`

Supports:

- Maximum Sharpe
- Minimum volatility
- Target return
- Target risk
- Quadratic utility
- Efficient-frontier generation
- Per-asset bounds
- Sector/group constraints

### Black-Litterman

`optimizers/black_litterman.py`

Combines market-implied equilibrium returns with absolute or relative views using the He-Litterman framework.

### Risk Parity

`optimizers/risk_parity.py`

Includes:

- Equal Risk Contribution
- Custom risk budgets
- Hierarchical Risk Parity

### CVaR / CDaR

`optimizers/cvar.py`

Uses the Rockafellar-Uryasev linear-program formulation to optimize directly against tail loss or drawdown scenarios.

---

# 2. Additional portfolio methods

The `advanced/` package contains several methods that go beyond the four core optimizers.

### Entropy Pooling

`advanced/entropy_pooling.py`

Allows return-distribution views to be expressed as equality or inequality constraints and solves for the closest probability distribution using KL divergence.

### CVaR Risk Parity

`advanced/cvar_risk_parity.py`

Allocates risk using CVaR contributions instead of variance contributions.

### Regime Switching

`advanced/regime_switching.py`

Uses a Gaussian HMM to identify latent market regimes and produces regime-specific or probability-weighted return and covariance estimates.

### Robust / Resampled Frontier

`advanced/robust_frontier.py`

Uses block bootstrap resampling and repeated optimization to reduce sensitivity to a single historical sample.

### Factor Risk Model

`advanced/factor_risk_model.py`

Supports PCA-based statistical factors as well as supplied fundamental factors. Portfolio risk can be decomposed into factor and idiosyncratic components.

### Stress Testing

`advanced/stress_testing.py`

Includes historical scenarios and Student-t Monte Carlo simulations for tail-risk analysis.

### Reports

`reporting/tearsheet.py` generates portable HTML tearsheets with embedded charts.

---

# 3. Bayesian methods

The `bayesian/` package contains several approaches to parameter uncertainty.

- Jorion-style empirical-Bayes shrinkage for expected returns
- Normal-Inverse-Wishart Bayesian mean-variance estimation
- Hierarchical partial-pooling across assets and groups

The goal here is not to assume that estimated returns and covariance matrices are known with certainty. The posterior and posterior-predictive implementations make that uncertainty explicit.

---

# 4. Reinforcement learning

The `rl/` directory is **experimental research code**.

It contains:

- `PortfolioEnv`
- PPO
- SAC
- DQN
- RL benchmarking against classical portfolio methods

The agents are implemented in PyTorch. The DQN implementation uses a discrete meta-action space because DQN is not a continuous-action algorithm.

RL is intentionally separated from the normal production path. It should not be treated as equivalent in maturity to the convex optimization methods.

To enable the experimental production/API path explicitly:

```bash
PORTFOLIO_OPTIMIZER_ENABLE_RL=1
```

This is disabled by default.

---

# 5. Backtesting

`backtester.py` provides walk-forward testing with a lookback window and configurable rebalance schedules.

The backtesting layer includes checks and models for:

- Look-ahead bias
- Point-in-time universe membership
- Survivorship bias
- Corporate-action-aware price data
- Transaction costs
- Bid/ask spread
- Commission
- Market impact
- Borrow cost
- Slippage
- ADV/liquidity limits
- Participation-rate limits
- No-trade bands

The intent is to make the assumptions visible instead of treating a historical backtest as if it were automatically representative of live trading.

A strategy only receives information that would have been available at the decision point.

---

# 6. Statistical validation

The `validation/` package contains tools for evaluating portfolio results beyond a single Sharpe ratio.

Current components include:

- Bootstrap confidence intervals
- Sharpe, Sortino, drawdown and CVaR statistics
- Jobson-Korkie-Memmel Sharpe comparison
- Holm-Bonferroni correction
- Benjamini-Hochberg correction
- Parameter sensitivity analysis
- Multi-regime robustness testing

The examples include bull, bear, high-volatility, crash/recovery and stagflation-style regimes.

---

# 7. Research contribution

One of the research components is:

`research/shrinkage_cvar_risk_parity.py`

It tests a Shrinkage-Adaptive CVaR Risk Parity approach. The idea is to shrink a small-sample tail-conditional-mean estimate toward the unconditional mean.

I did not treat the idea as successful just because it was new. The implementation was tested against pre-defined predictions using multiple random seeds and confidence intervals.

The current result is a **null result**: there is no statistically detectable improvement over plain CVaR Risk Parity at the tested sample size.

The full experiment is in:

```bash
python examples/novel_contribution_validation.py
```

That result is kept in the repository because it is part of the research record.

---

# 8. External benchmarking

`examples/benchmark_report.py` compares the implementation with:

- PyPortfolioOpt
- Riskfolio-Lib

The benchmark covers runtime, objective values and several difficult covariance/return inputs.

QuantLib is not included because it is primarily a derivatives and fixed-income pricing library rather than a directly comparable portfolio-optimization package.

Run it with:

```bash
python examples/benchmark_report.py
```

The benchmark output should be treated as a comparison for the tested problem sizes and configurations, not as a general ranking of the libraries.

---

# 9. Execution and multi-period optimization

## Multi-period optimization

`multiperiod/` contains:

- Li-Ng multi-period optimization
- Scenario-based Model Predictive Control

The scenario optimizer generates joint future paths, optimizes the sequence of portfolio decisions and includes transaction costs in the objective.

## Execution costs

`execution/` contains:

- Almgren-Chriss optimal execution
- Temporary market impact
- Permanent market impact
- Volatility/ADV-based cost estimates

These models can also be used by the backtester instead of relying on a single fixed basis-point assumption.

---

# 10. Market data

`data/adapters.py` provides a common interface for:

- Polygon
- Alpaca
- Yahoo Finance / `yfinance`
- Binance

The adapters use lazy imports where possible so users do not need every vendor SDK just to use the core optimizers.

The data layer also validates timestamps, duplicates, missing symbols, invalid prices and stale observations.

For live trading, stale or invalid data causes the trading cycle to stop rather than continuing with an old price.

---

# 11. Paper trading

The project includes a paper-trading loop using Alpaca.

It is designed around:

```text
Market data
    ↓
Portfolio optimizer
    ↓
Risk checks
    ↓
Order management
    ↓
Broker
    ↓
Audit log
```

The live path includes controls for:

- Maximum position weight
- Gross leverage
- Turnover
- Daily loss
- Drawdown
- Maximum order notional
- Maximum order count
- Stale market data
- Broker/local-state reconciliation
- Emergency stop
- Deterministic client order IDs

It is still **paper trading**. No part of this repository should be interpreted as a recommendation to trade real capital.

### Running the paper loop

Create the environment file:

```bash
cp .env.example .env
```

Add your Alpaca paper-trading credentials, then run:

```bash
python start_live_trading.py --once
```

Or:

```bash
python -m portfolio_optimizer.live.paper_trading_loop \
  --symbols AAPL MSFT GOOGL \
  --once
```

Use an Alpaca **paper** account for testing.

---

# 12. Trading infrastructure

The `infra/` package contains the lower-level components used by the trading and API layers.

### Order management

A state machine controls legal order transitions and rejects invalid transitions.

### Persistence

SQLite-backed portfolio state is available for local use.

### Audit log

Audit records use a structured schema and hash chaining so changes to the chain can be detected.

The API audit layer records fields such as:

- Timestamp
- Request ID
- Tenant ID
- API key ID
- User ID
- Endpoint
- Strategy
- Assets
- Risk limits
- Optimization parameters
- Result hash
- Risk decision
- Request latency
- Status

An optional signing secret can be used for stronger audit-record integrity.

### Fault tolerance

The infrastructure includes retry/backoff and circuit-breaker components for transient failures.

### FIX support

There is FIX message encoding/decoding with checksum verification. This is **not** a complete FIX session layer: logon, heartbeat, sequence recovery and exchange connectivity are outside the current scope.

---

# 13. API

The FastAPI service exposes the portfolio engines and supporting functionality through REST endpoints.

For local development:

```bash
export PORTFOLIO_OPTIMIZER_API_KEYS="your-key-here"
uvicorn portfolio_optimizer.api.service:app --reload
```

Production configuration uses hashed API keys rather than storing plaintext keys.

The API also includes:

- Request IDs
- API-key-aware rate limiting
- Structured audit logging
- Fail-closed authentication
- Configuration validation
- Health checks

The API should be deployed behind a proper TLS-terminating reverse proxy in a real environment.

---

# 14. SaaS components

The `saas/` package contains the application-level pieces needed for a multi-tenant service, including tenant separation and billing integration.

This is application infrastructure rather than a claim that the repository is a complete hosted SaaS business.

Billing tests cover the expected API behavior without requiring production billing credentials.

---

# 15. Dashboard and reporting

There are two dashboard/reporting areas in the project:

- `dashboard/` for portfolio views
- `reporting/` for standalone HTML reports

The workflow dashboard brings together portfolio construction, results and explanations in one interface.

The report generator produces a self-contained HTML file so the result can be opened without running the application.

---

# 16. Architecture

The project is intentionally split into several layers.

```text
                    ┌──────────────────────┐
                    │       API / SaaS     │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │ Production / Trading │
                    │ risk • OMS • audit   │
                    └──────────┬───────────┘
                               │
              ┌────────────────▼────────────────┐
              │        Core Portfolio           │
              │ optimizers • estimators • risk  │
              └────────────────┬────────────────┘
                               │
        ┌──────────────────────▼──────────────────────┐
        │ Research / Experimental                     │
        │ Bayesian • RL • regimes • multi-period      │
        └─────────────────────────────────────────────┘
```

The production-facing code does not depend on the experimental RL package. The architecture checks in the repository are intended to prevent that dependency from being introduced accidentally.

---

# 17. Configuration

The main live-trading settings include:

```text
LIVE_MAX_POSITION_WEIGHT
LIVE_MAX_GROSS_LEVERAGE
LIVE_MAX_TURNOVER
LIVE_DAILY_LOSS_LIMIT
LIVE_MAX_DRAWDOWN
LIVE_MAX_ORDER_NOTIONAL
LIVE_MAX_ORDERS_PER_CYCLE
LIVE_LOOKBACK_DAYS
LIVE_TRADING_SYMBOLS
LIVE_MAX_STALE_DATA_SECONDS
LIVE_EMERGENCY_STOP
```

Values are validated at startup. Unsafe ranges or incomplete live configuration cause the application to stop rather than silently using an invalid value.

For production API deployments, authentication and audit-signing configuration are also validated before startup.

---

# 18. Development checks

The repository includes a small Make-based development workflow.

```bash
make install
make test
make lint
make typecheck
make security
make benchmark
```

The CI workflow also checks the project on supported Python versions, runs the test suite and coverage checks, performs dependency/security checks, builds the Docker image and runs an API smoke test.

Exact dependency versions are kept in the lock files so a clean environment can reproduce the development setup.

---

# 19. Testing

The test suite covers the main research and application layers, including:

- Core optimizers
- Advanced optimizers
- Bayesian methods
- RL components
- Backtesting
- Validation
- Data adapters
- Broker abstractions
- Paper trading
- Risk controls
- Order management
- Audit logging
- API authentication
- SaaS behavior
- Billing endpoints
- Dashboard behavior
- Production configuration
- Optional dependencies

Run the tests with:

```bash
pytest -q
```

For coverage:

```bash
pytest -q --cov=portfolio_optimizer --cov-report=term-missing
```

The repository includes coverage gates for the overall project and higher minimums for critical production modules.

---

# 20. Production scope

There is a clear difference between the parts that are ready to support an application and the parts that would still need work before being used with real capital.

### In the repository

- Portfolio construction
- Risk controls
- Backtesting
- Statistical validation
- Data interfaces
- Paper trading
- Broker abstraction
- Order-management state machine
- Audit logging
- API authentication
- Configuration validation
- CI checks
- Dependency locking

### Still outside the scope

- Exchange/FIX session management
- Direct exchange connectivity
- Distributed execution across multiple machines
- Production market-data infrastructure
- Full operational monitoring and paging
- Formal compliance program
- Fund administration
- Independent strategy validation
- Real-capital deployment approval

Those are separate engineering, operational and legal projects.

---

# 21. Compliance and real-money use

This repository is software. It is not an investment fund and does not manage third-party money.

Using the paper-trading loop does not create a fund or provide investment advice.

Any move from research or paper trading to real capital would require separate work covering brokerage setup, legal structure, risk limits, compliance, monitoring, operational procedures and independent testing.

---

# 22. Useful examples

### Full optimizer demo

```bash
python examples/full_demo.py
```

### Extended methods

```bash
python examples/v2_demo.py
```

### Research contribution test

```bash
python examples/novel_contribution_validation.py
```

### Benchmark report

```bash
python examples/benchmark_report.py
```

### Synthetic data example

```bash
python examples/synthetic_data.py
```

---

# 23. Notes on the research results

Not every method in this repository is expected to outperform every other method.

Some components are included because they are useful portfolio-construction techniques. Others are included to test an idea, understand a model, or compare different assumptions.

The research contribution mentioned above is one example: the method was implemented, tested, and the result did not support the original hypothesis strongly enough to claim an improvement.

That distinction is important when reading the benchmarks and examples. A benchmark result is a result for the stated data and configuration, not a universal statement about a method.

---

# 24. License

See the repository license file for the terms that apply to this project.

---

## Author

**Aditya Diwan**

Aditya Diwan Capital (ADC)

This repository is primarily a personal quant-research and engineering project. The goal is to keep the implementation useful, testable and clear enough to inspect and extend.
