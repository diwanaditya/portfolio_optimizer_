"""
Broker Abstraction -- decouples "where do I get prices" from "who executes
my trades", which the previous version of the live loop did not do (it
imported Alpaca's trading client directly, hard-wiring data and execution
together). This module fixes that: `BrokerAdapter` is a common interface,
and there are two implementations --

    SimulatedBroker -- a fully local, zero-dependency, zero-signup paper
        account. No API key, no external service, no network call for
        execution at all. Orders fill immediately at whatever price you
        pass in (the live loop passes in the current real market price
        from whichever data adapter you're using -- Yahoo Finance by
        default). This is the new default: the lowest-friction way to
        run the live loop with genuinely zero account setup.

    AlpacaBroker -- wraps the real Alpaca paper-trading API (unchanged
        from before), for when you specifically want real broker order
        mechanics (partial fills, real order book interaction, a genuine
        second system of record) rather than a local simulation. Kept
        available, no longer required.

Either one implements the same three methods the live loop needs:
    get_account()      -> AccountState(equity, cash)
    get_positions()    -> dict[symbol, PositionState]
    submit_order(...)  -> BrokerOrderResult
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class AccountState:
    equity: float
    cash: float


@dataclass
class PositionState:
    symbol: str
    quantity: float
    market_value: float
    avg_entry_price: float


@dataclass
class BrokerOrderResult:
    broker_order_id: str
    symbol: str
    side: str
    filled_quantity: float
    filled_price: float
    status: str = "filled"


class BrokerAdapter(ABC):
    @abstractmethod
    def get_account(self) -> AccountState: ...

    @abstractmethod
    def get_positions(self) -> dict: ...

    @abstractmethod
    def submit_order(self, symbol: str, side: str, notional: float,
                      current_price: float | None = None) -> BrokerOrderResult: ...


class SimulatedBroker(BrokerAdapter):
    """Local, in-memory (optionally SQLite-persisted) simulated paper
    account. No external account, no API key, no network call to execute
    a trade -- orders fill instantly and completely at `current_price`,
    which the caller (the live loop) supplies from real live market data.

    This is deliberately simple (no partial fills, no slippage model, no
    order book) -- it's meant as the zero-friction default for someone
    getting started, not a substitute for realistic execution-cost
    modeling (see `execution/almgren_chriss.py` for that, which can be
    layered on top of this if you want simulated fills to reflect
    market-impact costs rather than filling at the exact quoted price).
    """

    def __init__(self, starting_cash: float = 100_000.0, store=None, portfolio_id: str = "sim_default"):
        self.cash = starting_cash
        self.positions: dict = {}   # symbol -> PositionState
        self.store = store          # optional infra.persistence.PortfolioStateStore
        self.portfolio_id = portfolio_id
        self._order_counter = 0

        if self.store is not None:
            self._load_from_store()

    def _load_from_store(self):
        existing = self.store.get_positions(self.portfolio_id)
        for symbol, pos in existing.items():
            self.positions[symbol] = PositionState(
                symbol=symbol, quantity=pos["quantity"],
                market_value=pos["quantity"] * pos["avg_cost"], avg_entry_price=pos["avg_cost"],
            )
        history = self.store.get_nav_history(self.portfolio_id)
        if history:
            # cash = last recorded NAV minus current mark-to-cost position value
            # (an approximation since we don't have live prices here; the
            # live loop re-marks positions to the live price on each cycle)
            total_position_cost = sum(p.market_value for p in self.positions.values())
            self.cash = history[-1]["cash"] if "cash" in history[-1] else history[-1]["nav"] - total_position_cost

    def get_account(self) -> AccountState:
        equity = self.cash + sum(p.market_value for p in self.positions.values())
        return AccountState(equity=equity, cash=self.cash)

    def get_positions(self) -> dict:
        return dict(self.positions)

    def submit_order(self, symbol: str, side: str, notional: float,
                      current_price: float | None = None) -> BrokerOrderResult:
        if current_price is None or current_price <= 0:
            raise ValueError(f"SimulatedBroker requires a valid current_price to fill an order "
                              f"(got {current_price!r} for {symbol}) -- pass the live market price.")

        quantity = notional / current_price
        if side.lower() == "sell":
            quantity = -quantity

        if side.lower() == "buy" and notional > self.cash + 1e-6:
            raise ValueError(f"Insufficient simulated cash: order notional ${notional:,.2f} "
                              f"exceeds available cash ${self.cash:,.2f}")

        existing = self.positions.get(symbol)
        if existing is None:
            new_qty = quantity
            new_avg_price = current_price
        else:
            new_qty = existing.quantity + quantity
            if new_qty == 0:
                new_avg_price = 0.0
            elif (existing.quantity >= 0) == (quantity >= 0):
                # adding to the position in the same direction -> blend cost basis
                total_cost = existing.quantity * existing.avg_entry_price + quantity * current_price
                new_avg_price = total_cost / new_qty
            else:
                # reducing/flipping -> keep existing cost basis until fully closed
                new_avg_price = existing.avg_entry_price

        self.cash -= quantity * current_price
        self.positions[symbol] = PositionState(
            symbol=symbol, quantity=new_qty, market_value=new_qty * current_price,
            avg_entry_price=new_avg_price,
        )
        if abs(new_qty) < 1e-9:
            del self.positions[symbol]

        if self.store is not None:
            self.store.upsert_position(self.portfolio_id, symbol, new_qty, new_avg_price)

        self._order_counter += 1
        return BrokerOrderResult(
            broker_order_id=f"SIM-{self._order_counter}", symbol=symbol, side=side,
            filled_quantity=abs(quantity), filled_price=current_price, status="filled",
        )


class AlpacaBroker(BrokerAdapter):
    """Wraps the real Alpaca paper-trading API. Requires ALPACA_API_KEY /
    ALPACA_SECRET_KEY. Kept available for anyone who specifically wants
    real broker order mechanics rather than the local simulation above --
    no longer required to run the live loop at all.
    """

    def __init__(self, api_key: str, api_secret: str, paper: bool = True):
        try:
            from alpaca.trading.client import TradingClient
        except ImportError as e:
            raise ImportError("pip install alpaca-py") from e
        self.client = TradingClient(api_key, api_secret, paper=paper)

    def get_account(self) -> AccountState:
        account = self.client.get_account()
        return AccountState(equity=float(account.equity), cash=float(account.cash))

    def get_positions(self) -> dict:
        positions = self.client.get_all_positions()
        return {
            p.symbol: PositionState(
                symbol=p.symbol, quantity=float(p.qty), market_value=float(p.market_value),
                avg_entry_price=float(p.avg_entry_price),
            ) for p in positions
        }

    def submit_order(self, symbol: str, side: str, notional: float,
                      current_price: float | None = None) -> BrokerOrderResult:
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        alpaca_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        req = MarketOrderRequest(symbol=symbol, notional=round(notional, 2),
                                  side=alpaca_side, time_in_force=TimeInForce.DAY)
        order = self.client.submit_order(req)
        filled_qty = float(order.filled_qty) if order.filled_qty else 0.0
        filled_price = float(order.filled_avg_price) if order.filled_avg_price else (current_price or 0.0)
        return BrokerOrderResult(
            broker_order_id=str(order.id), symbol=symbol, side=side,
            filled_quantity=filled_qty, filled_price=filled_price,
            status=str(order.status),
        )
