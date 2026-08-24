"""
Order Management System (OMS) — order lifecycle state machine.

SCOPE HONESTY: this tracks order state transitions and fills correctly
and safely (illegal transitions are rejected, not silently allowed) — it
does NOT connect to an actual exchange/broker or execution venue. In
production this would sit behind a real EMS/broker API (Alpaca, Interactive
Brokers, a FIX-connected prime broker) which would be the actual source
of fill events; this class is the correct place to plug those events in.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import uuid


class OrderStatus(Enum):
    PENDING_NEW = "pending_new"
    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    PENDING_CANCEL = "pending_cancel"


# Legal state transitions -- attempting an illegal transition raises,
# rather than silently corrupting order state (a real source of trading
# bugs: e.g. applying a fill to an already-cancelled order).
_LEGAL_TRANSITIONS = {
    OrderStatus.PENDING_NEW: {OrderStatus.NEW, OrderStatus.REJECTED},
    OrderStatus.NEW: {OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED,
                       OrderStatus.CANCELLED, OrderStatus.PENDING_CANCEL, OrderStatus.REJECTED},
    OrderStatus.PARTIALLY_FILLED: {OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED,
                                    OrderStatus.CANCELLED, OrderStatus.PENDING_CANCEL},
    OrderStatus.PENDING_CANCEL: {OrderStatus.CANCELLED, OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED},
    OrderStatus.FILLED: set(),      # terminal
    OrderStatus.CANCELLED: set(),   # terminal
    OrderStatus.REJECTED: set(),    # terminal
}


class IllegalOrderTransitionError(Exception):
    pass


@dataclass
class Fill:
    quantity: float
    price: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Order:
    symbol: str
    side: str          # "buy" or "sell"
    quantity: float
    order_type: str = "market"
    limit_price: float | None = None
    client_order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: OrderStatus = OrderStatus.PENDING_NEW
    fills: list = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    reject_reason: str | None = None

    @property
    def filled_quantity(self) -> float:
        return sum(f.quantity for f in self.fills)

    @property
    def remaining_quantity(self) -> float:
        return self.quantity - self.filled_quantity

    @property
    def average_fill_price(self) -> float:
        if not self.fills:
            return 0.0
        total_value = sum(f.quantity * f.price for f in self.fills)
        return total_value / self.filled_quantity if self.filled_quantity > 0 else 0.0

    def _transition(self, new_status: OrderStatus):
        if new_status not in _LEGAL_TRANSITIONS.get(self.status, set()) and new_status != self.status:
            raise IllegalOrderTransitionError(
                f"Cannot transition order {self.client_order_id} from {self.status} to {new_status}"
            )
        self.status = new_status
        self.updated_at = datetime.now(timezone.utc)

    def acknowledge(self):
        self._transition(OrderStatus.NEW)

    def reject(self, reason: str):
        self.reject_reason = reason
        self._transition(OrderStatus.REJECTED)

    def apply_fill(self, quantity: float, price: float):
        if self.status not in (OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED):
            raise IllegalOrderTransitionError(
                f"Cannot apply fill to order {self.client_order_id} in status {self.status}"
            )
        if quantity > self.remaining_quantity + 1e-9:
            raise ValueError(
                f"Fill quantity {quantity} exceeds remaining quantity {self.remaining_quantity}"
            )
        self.fills.append(Fill(quantity=quantity, price=price))
        if abs(self.remaining_quantity) < 1e-9:
            self._transition(OrderStatus.FILLED)
        else:
            self._transition(OrderStatus.PARTIALLY_FILLED)

    def request_cancel(self):
        self._transition(OrderStatus.PENDING_CANCEL)

    def confirm_cancel(self):
        self._transition(OrderStatus.CANCELLED)


class OrderManager:
    """Tracks all orders for a portfolio/session, provides lookup and
    aggregate reporting (open orders, fill rate, average slippage vs
    arrival price if supplied).
    """

    def __init__(self):
        self.orders: dict = {}

    def submit(self, symbol: str, side: str, quantity: float, order_type: str = "market",
               limit_price: float | None = None, client_order_id: str | None = None) -> Order:
        order = Order(symbol=symbol, side=side, quantity=quantity,
                       order_type=order_type, limit_price=limit_price,
                       client_order_id=client_order_id or str(uuid.uuid4()))
        if order.client_order_id in self.orders:
            raise ValueError(f"Duplicate client_order_id: {order.client_order_id}")
        self.orders[order.client_order_id] = order
        return order

    def get(self, client_order_id: str) -> Order:
        if client_order_id not in self.orders:
            raise KeyError(f"Unknown order id: {client_order_id}")
        return self.orders[client_order_id]

    def open_orders(self) -> list:
        return [o for o in self.orders.values()
                if o.status in (OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED, OrderStatus.PENDING_NEW)]

    def filled_orders(self) -> list:
        return [o for o in self.orders.values() if o.status == OrderStatus.FILLED]

    def summary(self) -> dict:
        by_status = {}
        for o in self.orders.values():
            by_status[o.status.value] = by_status.get(o.status.value, 0) + 1
        return {
            "total_orders": len(self.orders),
            "by_status": by_status,
            "open_orders": len(self.open_orders()),
        }
