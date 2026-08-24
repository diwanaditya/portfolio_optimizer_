"""
Pre-Trade Operational Risk Controls.

The rules engine that would sit between "the optimizer says trade to
these weights" and "orders actually go to market" — checking hard
operational limits that have nothing to do with whether the optimization
math is correct (a mathematically optimal portfolio can still violate a
mandate's leverage limit, concentration limit, or a restricted-securities
list). Every real trading operation has some version of this layer;
skipping it is how "the optimizer said so" becomes a post-mortem.
"""
from __future__ import annotations
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum


class CheckSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    BLOCK = "block"


@dataclass
class RiskCheckResult:
    check_name: str
    passed: bool
    severity: CheckSeverity
    message: str
    details: dict = field(default_factory=dict)


@dataclass
class PreTradeLimits:
    max_position_weight: float = 0.40         # single-asset concentration limit
    max_gross_leverage: float = 1.0            # sum(|weights|); 1.0 = no leverage
    max_sector_weight: dict = field(default_factory=dict)  # {sector: max_weight}
    restricted_symbols: set = field(default_factory=set)
    max_single_trade_turnover: float = 1.0     # max |target - current| for any one asset
    max_total_turnover: float = 2.0            # max sum |target - current| across the book


class PreTradeRiskEngine:
    def __init__(self, limits: PreTradeLimits, sector_map: dict | None = None):
        self.limits = limits
        self.sector_map = sector_map or {}

    def check_target_portfolio(self, target_weights: pd.Series,
                                 current_weights: pd.Series | None = None) -> list:
        results = []
        current_weights = current_weights if current_weights is not None else pd.Series(0.0, index=target_weights.index)

        # 1. Restricted list
        held_restricted = [s for s in target_weights.index
                            if s in self.limits.restricted_symbols and abs(target_weights[s]) > 1e-9]
        results.append(RiskCheckResult(
            check_name="restricted_symbols", passed=len(held_restricted) == 0,
            severity=CheckSeverity.BLOCK,
            message=(f"Target portfolio holds restricted symbols: {held_restricted}"
                     if held_restricted else "No restricted symbols held"),
            details={"restricted_held": held_restricted},
        ))

        # 2. Single-position concentration limit
        breaches = target_weights[target_weights.abs() > self.limits.max_position_weight]
        results.append(RiskCheckResult(
            check_name="max_position_weight", passed=len(breaches) == 0,
            severity=CheckSeverity.BLOCK,
            message=(f"Positions exceed {self.limits.max_position_weight:.0%} limit: {breaches.to_dict()}"
                     if len(breaches) else "All positions within concentration limit"),
            details={"breaches": breaches.to_dict()},
        ))

        # 3. Gross leverage
        gross = target_weights.abs().sum()
        results.append(RiskCheckResult(
            check_name="max_gross_leverage", passed=gross <= self.limits.max_gross_leverage + 1e-6,
            severity=CheckSeverity.BLOCK,
            message=f"Gross exposure {gross:.2%} vs limit {self.limits.max_gross_leverage:.0%}",
            details={"gross_exposure": float(gross)},
        ))

        # 4. Sector limits
        if self.limits.max_sector_weight and self.sector_map:
            sector_totals = {}
            for asset, w in target_weights.items():
                sector = self.sector_map.get(asset)
                if sector:
                    sector_totals[sector] = sector_totals.get(sector, 0.0) + w
            sector_breaches = {s: v for s, v in sector_totals.items()
                                if s in self.limits.max_sector_weight and v > self.limits.max_sector_weight[s]}
            results.append(RiskCheckResult(
                check_name="max_sector_weight", passed=len(sector_breaches) == 0,
                severity=CheckSeverity.BLOCK,
                message=(f"Sector limits breached: {sector_breaches}" if sector_breaches
                         else "All sector exposures within limits"),
                details={"sector_totals": sector_totals, "breaches": sector_breaches},
            ))

        # 5. Turnover limits
        aligned_target = target_weights.reindex(current_weights.index.union(target_weights.index)).fillna(0.0)
        aligned_current = current_weights.reindex(aligned_target.index).fillna(0.0)
        trade_sizes = (aligned_target - aligned_current).abs()
        total_turnover = trade_sizes.sum()
        max_single = trade_sizes.max() if len(trade_sizes) else 0.0

        results.append(RiskCheckResult(
            check_name="max_total_turnover", passed=total_turnover <= self.limits.max_total_turnover + 1e-6,
            severity=CheckSeverity.WARNING,
            message=f"Total turnover {total_turnover:.2%} vs limit {self.limits.max_total_turnover:.0%}",
            details={"total_turnover": float(total_turnover)},
        ))
        results.append(RiskCheckResult(
            check_name="max_single_trade_turnover",
            passed=max_single <= self.limits.max_single_trade_turnover + 1e-6,
            severity=CheckSeverity.WARNING,
            message=f"Largest single trade {max_single:.2%} vs limit {self.limits.max_single_trade_turnover:.0%}",
            details={"max_single_trade": float(max_single)},
        ))

        return results

    def is_approved(self, results: list) -> bool:
        """Overall approval: any BLOCK-severity failure vetoes the trade;
        WARNING-severity failures do not, but should still be surfaced.
        """
        return all(r.passed for r in results if r.severity == CheckSeverity.BLOCK)

    def summary_report(self, results: list) -> pd.DataFrame:
        return pd.DataFrame([{
            "check": r.check_name, "passed": r.passed, "severity": r.severity.value,
            "message": r.message,
        } for r in results])
