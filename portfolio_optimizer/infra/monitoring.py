"""
Live Monitoring & Alerting.

SCOPE HONESTY: this is a real, working rules engine that evaluates
portfolio/risk metrics against thresholds and generates structured
alerts — it does NOT include a dashboard UI, a push-notification
integration (Slack/PagerDuty/email), or a scheduler to run checks on a
timer. Wiring `MonitoringEngine.check_all()` into a cron job / scheduled
task and piping `Alert` objects into whatever notification channel you
use is the last, deployment-specific step, deliberately left out since
it depends entirely on your actual ops stack.
"""
from __future__ import annotations
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Callable


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    rule_name: str
    severity: AlertSeverity
    message: str
    value: float
    threshold: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class MonitoringRule:
    name: str
    metric_fn: Callable[[dict], float]   # takes a state dict, returns a scalar
    threshold: float
    comparison: str = "greater_than"      # "greater_than" or "less_than"
    severity: AlertSeverity = AlertSeverity.WARNING
    message_template: str = "{name}: value {value:.4f} breached threshold {threshold:.4f}"

    def evaluate(self, state: dict) -> Alert | None:
        value = self.metric_fn(state)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        breached = (value > self.threshold if self.comparison == "greater_than"
                    else value < self.threshold)
        if not breached:
            return None
        return Alert(
            rule_name=self.name, severity=self.severity,
            message=self.message_template.format(name=self.name, value=value, threshold=self.threshold),
            value=float(value), threshold=self.threshold,
        )


class MonitoringEngine:
    def __init__(self):
        self.rules: list = []
        self.alert_history: list = []

    def add_rule(self, rule: MonitoringRule):
        self.rules.append(rule)
        return self

    def check_all(self, state: dict) -> list:
        """`state` is a dict of whatever the rules' metric_fns need --
        typically current weights, returns history, NAV, drawdown, etc.
        Evaluates every registered rule and returns any triggered alerts.
        """
        triggered = []
        for rule in self.rules:
            try:
                alert = rule.evaluate(state)
            except Exception as e:
                alert = Alert(rule_name=rule.name, severity=AlertSeverity.WARNING,
                               message=f"Rule '{rule.name}' failed to evaluate: {e}",
                               value=float("nan"), threshold=rule.threshold)
            if alert:
                triggered.append(alert)
                self.alert_history.append(alert)
        return triggered

    def alerts_by_severity(self, severity: AlertSeverity) -> list:
        return [a for a in self.alert_history if a.severity == severity]


# --------------------------------------------------------------------- #
# Standard rule library — ready-to-use rules for common risk thresholds
# --------------------------------------------------------------------- #

def drawdown_rule(max_drawdown: float = -0.15) -> MonitoringRule:
    def metric_fn(state: dict) -> float:
        equity = state["equity_curve"]
        dd = equity / equity.cummax() - 1
        return float(dd.iloc[-1])
    return MonitoringRule(
        name="current_drawdown", metric_fn=metric_fn, threshold=max_drawdown,
        comparison="less_than", severity=AlertSeverity.CRITICAL,
        message_template="Drawdown alert: current drawdown {value:.2%} breached limit {threshold:.2%}",
    )


def concentration_rule(max_weight: float = 0.35) -> MonitoringRule:
    def metric_fn(state: dict) -> float:
        return float(state["weights"].abs().max())
    return MonitoringRule(
        name="max_position_concentration", metric_fn=metric_fn, threshold=max_weight,
        comparison="greater_than", severity=AlertSeverity.WARNING,
        message_template="Concentration alert: largest position {value:.2%} exceeds {threshold:.2%}",
    )


def var_breach_rule(var_limit: float = 0.05) -> MonitoringRule:
    def metric_fn(state: dict) -> float:
        recent_returns = state["recent_returns"]
        return float((-recent_returns).quantile(0.95))
    return MonitoringRule(
        name="var_95_breach", metric_fn=metric_fn, threshold=var_limit,
        comparison="greater_than", severity=AlertSeverity.CRITICAL,
        message_template="VaR alert: realized 95% VaR {value:.2%} exceeds limit {threshold:.2%}",
    )


def turnover_spike_rule(max_turnover: float = 0.50) -> MonitoringRule:
    def metric_fn(state: dict) -> float:
        return float(state.get("last_turnover", 0.0))
    return MonitoringRule(
        name="turnover_spike", metric_fn=metric_fn, threshold=max_turnover,
        comparison="greater_than", severity=AlertSeverity.WARNING,
        message_template="Turnover alert: last rebalance turnover {value:.2%} exceeds {threshold:.2%}",
    )


def stale_data_rule(max_staleness_seconds: float = 300.0) -> MonitoringRule:
    def metric_fn(state: dict) -> float:
        last_update = state.get("last_data_update")
        if last_update is None:
            return float("inf")
        elapsed = (datetime.now(timezone.utc) - last_update).total_seconds()
        return elapsed
    return MonitoringRule(
        name="stale_market_data", metric_fn=metric_fn, threshold=max_staleness_seconds,
        comparison="greater_than", severity=AlertSeverity.CRITICAL,
        message_template="Data staleness alert: market data is {value:.0f}s old (limit {threshold:.0f}s)",
    )
