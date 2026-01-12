"""
Position Manager - Authoritative Order/Position/Exposure Tracking.

Maintains authoritative record of all orders, positions, fills, and exposure
with full reconciliation against IB at all times. Handles fill mismatches
and provides complete audit trail for all position changes.

Key Features:
- Authoritative position tracking (single source of truth)
- Order lifecycle management (pending, filled, cancelled)
- Fill tracking with mismatch detection
- Real-time exposure calculation
- Full IB reconciliation with mismatch handling
- Complete audit trail for all changes
- P&L tracking per position

CRITICAL: This is the AUTHORITATIVE source for positions.
All other components must query this module for position state.
"""

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Awaitable

# Support both ib_async and ib_insync for compatibility
try:
    from ib_async import Contract, Trade, Fill, Position as IBPosition
    IB_LIBRARY = "ib_async"
except ImportError:
    try:
        from ib_insync import Contract, Trade, Fill, Position as IBPosition
        IB_LIBRARY = "ib_insync"
    except ImportError:
        Contract = Any  # type: ignore
        Trade = Any  # type: ignore
        Fill = Any  # type: ignore
        IBPosition = Any  # type: ignore
        IB_LIBRARY = None

from config import get_config


class OrderState(Enum):
    """Order lifecycle state."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL_FILL = "partial_fill"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    ERROR = "error"


class PositionSide(Enum):
    """Position side."""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class ReconciliationResult(Enum):
    """Result of reconciliation check."""
    MATCH = "match"
    MISMATCH = "mismatch"
    IB_ONLY = "ib_only"  # Position exists in IB but not locally
    LOCAL_ONLY = "local_only"  # Position exists locally but not in IB


@dataclass
class OrderRecord:
    """
    Complete order record with full lifecycle tracking.

    Tracks order from creation through execution or cancellation
    with all relevant metadata.
    """
    order_id: str  # Internal tracking ID
    ib_order_id: Optional[int] = None  # IB's order ID
    contract_id: int = 0
    symbol: str = ""
    exchange: str = ""

    # Order details
    side: str = ""  # BUY or SELL
    quantity: int = 0
    order_type: str = "market"  # market, limit, etc.
    limit_price: Optional[float] = None

    # State
    state: OrderState = OrderState.PENDING

    # Fill tracking
    filled_quantity: int = 0
    remaining_quantity: int = 0
    average_fill_price: float = 0.0
    fills: List[dict] = field(default_factory=list)

    # Commission
    commission: float = 0.0

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Trade reference
    ib_trade: Optional[Trade] = None

    # Metadata
    reason: str = ""
    prediction_id: Optional[str] = None

    def record_fill(
        self,
        filled_quantity: int,
        fill_price: float,
        commission: float = 0.0,
        fill_time: Optional[datetime] = None,
    ) -> None:
        """Record a fill for this order."""
        fill_time = fill_time or datetime.now(timezone.utc)

        # Record fill details
        self.fills.append({
            'quantity': filled_quantity,
            'price': fill_price,
            'commission': commission,
            'timestamp': fill_time.isoformat(),
        })

        # Update aggregate fill tracking
        previous_filled = self.filled_quantity
        self.filled_quantity += filled_quantity
        self.remaining_quantity = self.quantity - self.filled_quantity
        self.commission += commission

        # Calculate weighted average price
        if self.filled_quantity > 0:
            if previous_filled > 0:
                self.average_fill_price = (
                    (previous_filled * self.average_fill_price +
                     filled_quantity * fill_price) / self.filled_quantity
                )
            else:
                self.average_fill_price = fill_price

        # Update state
        if self.filled_quantity >= self.quantity:
            self.state = OrderState.FILLED
            self.filled_at = fill_time
        else:
            self.state = OrderState.PARTIAL_FILL

        self.last_updated = fill_time

    @property
    def is_complete(self) -> bool:
        """Check if order is complete (filled, cancelled, rejected, or error)."""
        return self.state in (
            OrderState.FILLED,
            OrderState.CANCELLED,
            OrderState.REJECTED,
            OrderState.ERROR,
        )

    @property
    def is_active(self) -> bool:
        """Check if order is still active (pending, submitted, partial)."""
        return self.state in (
            OrderState.PENDING,
            OrderState.SUBMITTED,
            OrderState.PARTIAL_FILL,
        )

    def to_dict(self) -> dict:
        """Export as dictionary."""
        return {
            'order_id': self.order_id,
            'ib_order_id': self.ib_order_id,
            'contract_id': self.contract_id,
            'symbol': self.symbol,
            'exchange': self.exchange,
            'side': self.side,
            'quantity': self.quantity,
            'order_type': self.order_type,
            'limit_price': self.limit_price,
            'state': self.state.value,
            'filled_quantity': self.filled_quantity,
            'remaining_quantity': self.remaining_quantity,
            'average_fill_price': self.average_fill_price,
            'commission': self.commission,
            'fills_count': len(self.fills),
            'created_at': self.created_at.isoformat(),
            'submitted_at': self.submitted_at.isoformat() if self.submitted_at else None,
            'filled_at': self.filled_at.isoformat() if self.filled_at else None,
            'cancelled_at': self.cancelled_at.isoformat() if self.cancelled_at else None,
            'reason': self.reason,
            'prediction_id': self.prediction_id,
        }


@dataclass
class PositionRecord:
    """
    Position record for a single contract.

    Tracks current position, P&L, and history for audit trail.
    """
    contract_id: int
    symbol: str = ""
    exchange: str = ""

    # Position state
    quantity: int = 0  # Positive = long, Negative = short
    average_entry_price: float = 0.0

    # P&L tracking
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_commission: float = 0.0

    # Metadata
    first_entry_time: Optional[datetime] = None
    last_update_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # History for audit
    position_changes: deque = field(default_factory=lambda: deque(maxlen=100))

    # Contract reference (for exposure calculation)
    contract: Optional[Contract] = None
    multiplier: float = 1.0  # Contract multiplier for notional calculation

    @property
    def side(self) -> PositionSide:
        """Get position side."""
        if self.quantity > 0:
            return PositionSide.LONG
        elif self.quantity < 0:
            return PositionSide.SHORT
        return PositionSide.FLAT

    @property
    def market_value(self) -> float:
        """Get position market value at entry."""
        return abs(self.quantity) * self.average_entry_price * self.multiplier

    @property
    def is_flat(self) -> bool:
        """Check if position is flat."""
        return self.quantity == 0

    def update_position(
        self,
        quantity_change: int,
        price: float,
        commission: float = 0.0,
        timestamp: Optional[datetime] = None,
    ) -> float:
        """
        Update position with a trade.

        Handles opening, adding to, reducing, and closing positions
        with proper P&L calculation.

        Args:
            quantity_change: Signed quantity (positive=buy, negative=sell)
            price: Execution price
            commission: Commission paid
            timestamp: Time of update

        Returns:
            Realized P&L from this trade (if position reduced/closed)
        """
        timestamp = timestamp or datetime.now(timezone.utc)
        realized_pnl = 0.0

        old_quantity = self.quantity
        new_quantity = self.quantity + quantity_change

        # Track commission
        self.total_commission += commission

        # Record position change for audit
        self.position_changes.append({
            'old_quantity': old_quantity,
            'change': quantity_change,
            'new_quantity': new_quantity,
            'price': price,
            'commission': commission,
            'timestamp': timestamp.isoformat(),
        })

        # Set first entry time if opening position
        if old_quantity == 0 and new_quantity != 0:
            self.first_entry_time = timestamp

        # Calculate P&L for position changes
        if old_quantity == 0:
            # Opening new position
            self.average_entry_price = price
        elif (old_quantity > 0 and quantity_change > 0) or \
             (old_quantity < 0 and quantity_change < 0):
            # Adding to existing position - calculate new average
            total_cost = (
                abs(old_quantity) * self.average_entry_price +
                abs(quantity_change) * price
            )
            self.average_entry_price = total_cost / (abs(old_quantity) + abs(quantity_change))
        elif (old_quantity > 0 and quantity_change < 0) or \
             (old_quantity < 0 and quantity_change > 0):
            # Reducing position - calculate realized P&L
            closing_quantity = min(abs(old_quantity), abs(quantity_change))

            if old_quantity > 0:
                # Closing long - profit if price > entry
                realized_pnl = closing_quantity * (price - self.average_entry_price) * self.multiplier
            else:
                # Closing short - profit if price < entry
                realized_pnl = closing_quantity * (self.average_entry_price - price) * self.multiplier

            self.realized_pnl += realized_pnl

            # Check if flipping position
            if abs(quantity_change) > abs(old_quantity):
                # Flipping to opposite side - reset entry price for new portion
                self.average_entry_price = price

        # Update quantity and timestamp
        self.quantity = new_quantity
        self.last_update_time = timestamp

        # Reset if flat
        if self.quantity == 0:
            self.average_entry_price = 0.0
            self.first_entry_time = None

        return realized_pnl

    def update_unrealized_pnl(self, current_price: float) -> None:
        """
        Update unrealized P&L based on current market price.

        Args:
            current_price: Current market price
        """
        if self.quantity == 0 or self.average_entry_price == 0:
            self.unrealized_pnl = 0.0
            return

        if self.quantity > 0:
            # Long position
            self.unrealized_pnl = (
                self.quantity * (current_price - self.average_entry_price) * self.multiplier
            )
        else:
            # Short position
            self.unrealized_pnl = (
                abs(self.quantity) * (self.average_entry_price - current_price) * self.multiplier
            )

    @property
    def total_pnl(self) -> float:
        """Get total P&L (realized + unrealized - commission)."""
        return self.realized_pnl + self.unrealized_pnl - self.total_commission

    def to_dict(self) -> dict:
        """Export as dictionary."""
        return {
            'contract_id': self.contract_id,
            'symbol': self.symbol,
            'exchange': self.exchange,
            'quantity': self.quantity,
            'side': self.side.value,
            'average_entry_price': self.average_entry_price,
            'market_value': self.market_value,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'total_pnl': self.total_pnl,
            'total_commission': self.total_commission,
            'multiplier': self.multiplier,
            'first_entry_time': (
                self.first_entry_time.isoformat()
                if self.first_entry_time else None
            ),
            'last_update_time': self.last_update_time.isoformat(),
            'changes_count': len(self.position_changes),
        }


@dataclass
class ReconciliationReport:
    """Report from IB reconciliation."""
    timestamp: datetime
    matches: int = 0
    mismatches: int = 0
    ib_only: int = 0
    local_only: int = 0
    details: List[dict] = field(default_factory=list)
    corrections_applied: int = 0

    @property
    def is_clean(self) -> bool:
        """Check if reconciliation found no issues."""
        return self.mismatches == 0 and self.ib_only == 0 and self.local_only == 0

    def to_dict(self) -> dict:
        """Export as dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'is_clean': self.is_clean,
            'matches': self.matches,
            'mismatches': self.mismatches,
            'ib_only': self.ib_only,
            'local_only': self.local_only,
            'corrections_applied': self.corrections_applied,
            'details': self.details,
        }


class PositionManager:
    """
    Authoritative Position Manager.

    Maintains the single source of truth for all orders, positions, and exposure.
    Provides reconciliation with IB to detect and handle mismatches.

    Key Responsibilities:
    - Track all orders from creation to completion
    - Maintain authoritative position state
    - Calculate real-time exposure
    - Reconcile with IB positions
    - Provide complete audit trail

    Usage:
        manager = PositionManager()

        # Record order
        order = manager.create_order(
            contract_id=123,
            symbol="ES",
            side="BUY",
            quantity=1,
        )

        # Record fill
        manager.record_fill(
            order_id=order.order_id,
            quantity=1,
            price=4500.0,
        )

        # Get position
        position = manager.get_position(contract_id=123)

        # Reconcile with IB
        report = await manager.reconcile_with_ib(ib_client)
    """

    def __init__(
        self,
        max_exposure_contracts: int = None,
        max_position_per_contract: int = None,
    ):
        """
        Initialize PositionManager.

        Args:
            max_exposure_contracts: Maximum total contracts across all symbols
            max_position_per_contract: Maximum contracts per symbol
        """
        self._config = get_config()
        self._logger = logging.getLogger(__name__)

        # Risk limits
        self._max_exposure = max_exposure_contracts or 50
        self._max_position = max_position_per_contract or 10

        # Order tracking
        self._orders: Dict[str, OrderRecord] = {}  # order_id -> OrderRecord
        self._ib_order_map: Dict[int, str] = {}  # ib_order_id -> order_id
        self._active_orders: Dict[str, OrderRecord] = {}  # Active orders only

        # Position tracking (authoritative)
        self._positions: Dict[int, PositionRecord] = {}  # contract_id -> PositionRecord

        # Reconciliation history
        self._reconciliation_history: deque = field(default_factory=lambda: deque(maxlen=50))
        self._last_reconciliation: Optional[ReconciliationReport] = None

        # Audit log
        self._audit_log: deque = deque(maxlen=1000)

        # Callbacks
        self._on_position_change: List[Callable[[int, int, int], None]] = []
        self._on_order_update: List[Callable[[OrderRecord], None]] = []
        self._on_mismatch: List[Callable[[ReconciliationReport], Awaitable[None]]] = []

        # Statistics
        self._total_orders = 0
        self._total_fills = 0
        self._total_realized_pnl = 0.0

        self._logger.info(
            f"PositionManager initialized: max_exposure={self._max_exposure}, "
            f"max_position={self._max_position}"
        )

    def create_order(
        self,
        contract_id: int,
        symbol: str,
        side: str,
        quantity: int,
        exchange: str = "",
        order_type: str = "market",
        limit_price: Optional[float] = None,
        reason: str = "",
        prediction_id: Optional[str] = None,
    ) -> OrderRecord:
        """
        Create a new order record.

        Args:
            contract_id: IB contract ID
            symbol: Contract symbol
            side: BUY or SELL
            quantity: Number of contracts
            exchange: Exchange code
            order_type: Order type (market, limit)
            limit_price: Limit price for limit orders
            reason: Reason for order
            prediction_id: Optional prediction ID that triggered this

        Returns:
            Created OrderRecord
        """
        import uuid
        order_id = str(uuid.uuid4())

        order = OrderRecord(
            order_id=order_id,
            contract_id=contract_id,
            symbol=symbol,
            exchange=exchange,
            side=side.upper(),
            quantity=quantity,
            remaining_quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            reason=reason,
            prediction_id=prediction_id,
        )

        self._orders[order_id] = order
        self._active_orders[order_id] = order
        self._total_orders += 1

        self._audit("ORDER_CREATED", {
            'order_id': order_id,
            'contract_id': contract_id,
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
        })

        self._logger.info(
            f"Order created: {side} {quantity} {symbol} "
            f"(order_id={order_id[:8]}...)"
        )

        return order

    def update_order_submitted(
        self,
        order_id: str,
        ib_order_id: int,
        ib_trade: Optional[Trade] = None,
    ) -> Optional[OrderRecord]:
        """
        Update order as submitted to IB.

        Args:
            order_id: Internal order ID
            ib_order_id: IB's order ID
            ib_trade: Optional IB Trade object

        Returns:
            Updated OrderRecord or None if not found
        """
        if order_id not in self._orders:
            self._logger.warning(f"Order not found: {order_id}")
            return None

        order = self._orders[order_id]
        order.ib_order_id = ib_order_id
        order.ib_trade = ib_trade
        order.state = OrderState.SUBMITTED
        order.submitted_at = datetime.now(timezone.utc)
        order.last_updated = order.submitted_at

        # Map IB order ID to internal ID
        self._ib_order_map[ib_order_id] = order_id

        self._audit("ORDER_SUBMITTED", {
            'order_id': order_id,
            'ib_order_id': ib_order_id,
        })

        self._notify_order_update(order)
        return order

    def record_fill(
        self,
        order_id: str = None,
        ib_order_id: int = None,
        quantity: int = 0,
        price: float = 0.0,
        commission: float = 0.0,
        fill_time: Optional[datetime] = None,
    ) -> Optional[float]:
        """
        Record a fill for an order.

        Updates both order record and position record.

        Args:
            order_id: Internal order ID (or use ib_order_id)
            ib_order_id: IB's order ID (if order_id not provided)
            quantity: Filled quantity
            price: Fill price
            commission: Commission paid
            fill_time: Time of fill

        Returns:
            Realized P&L from this fill (if position reduced)
        """
        # Resolve order ID
        if order_id is None and ib_order_id is not None:
            order_id = self._ib_order_map.get(ib_order_id)

        if order_id is None or order_id not in self._orders:
            self._logger.warning(f"Order not found for fill: {order_id or ib_order_id}")
            return None

        order = self._orders[order_id]

        # Record fill on order
        order.record_fill(quantity, price, commission, fill_time)
        self._total_fills += 1

        # Update position
        position = self._get_or_create_position(
            order.contract_id,
            order.symbol,
            order.exchange,
        )

        # Determine signed quantity change
        if order.side == "BUY":
            quantity_change = quantity
        else:
            quantity_change = -quantity

        # Update position and get realized P&L
        realized_pnl = position.update_position(
            quantity_change=quantity_change,
            price=price,
            commission=commission,
            timestamp=fill_time,
        )

        self._total_realized_pnl += realized_pnl

        # Remove from active if complete
        if order.is_complete and order_id in self._active_orders:
            del self._active_orders[order_id]

        self._audit("FILL_RECORDED", {
            'order_id': order_id,
            'quantity': quantity,
            'price': price,
            'commission': commission,
            'realized_pnl': realized_pnl,
            'new_position': position.quantity,
        })

        self._logger.info(
            f"Fill recorded: {order.side} {quantity} {order.symbol} @ {price:.2f} "
            f"(position={position.quantity}, realized_pnl={realized_pnl:.2f})"
        )

        # Notify position change
        self._notify_position_change(
            order.contract_id,
            position.quantity - quantity_change,
            position.quantity,
        )

        self._notify_order_update(order)

        return realized_pnl

    def record_cancellation(
        self,
        order_id: str = None,
        ib_order_id: int = None,
    ) -> Optional[OrderRecord]:
        """
        Record order cancellation.

        Args:
            order_id: Internal order ID
            ib_order_id: IB's order ID (if order_id not provided)

        Returns:
            Updated OrderRecord or None if not found
        """
        # Resolve order ID
        if order_id is None and ib_order_id is not None:
            order_id = self._ib_order_map.get(ib_order_id)

        if order_id is None or order_id not in self._orders:
            self._logger.warning(f"Order not found for cancellation: {order_id or ib_order_id}")
            return None

        order = self._orders[order_id]
        order.state = OrderState.CANCELLED
        order.cancelled_at = datetime.now(timezone.utc)
        order.last_updated = order.cancelled_at

        # Remove from active
        if order_id in self._active_orders:
            del self._active_orders[order_id]

        self._audit("ORDER_CANCELLED", {
            'order_id': order_id,
            'filled_quantity': order.filled_quantity,
        })

        self._logger.info(
            f"Order cancelled: {order.side} {order.symbol} "
            f"(filled={order.filled_quantity}/{order.quantity})"
        )

        self._notify_order_update(order)
        return order

    def _get_or_create_position(
        self,
        contract_id: int,
        symbol: str = "",
        exchange: str = "",
    ) -> PositionRecord:
        """Get or create position record for a contract."""
        if contract_id not in self._positions:
            self._positions[contract_id] = PositionRecord(
                contract_id=contract_id,
                symbol=symbol,
                exchange=exchange,
            )
        return self._positions[contract_id]

    def get_position(self, contract_id: int) -> Optional[PositionRecord]:
        """
        Get position for a contract.

        Args:
            contract_id: Contract ID

        Returns:
            PositionRecord or None if no position
        """
        return self._positions.get(contract_id)

    def get_position_quantity(self, contract_id: int) -> int:
        """
        Get position quantity for a contract.

        Args:
            contract_id: Contract ID

        Returns:
            Position quantity (positive=long, negative=short, 0=flat)
        """
        position = self._positions.get(contract_id)
        return position.quantity if position else 0

    def get_order(self, order_id: str) -> Optional[OrderRecord]:
        """Get order by ID."""
        return self._orders.get(order_id)

    def get_order_by_ib_id(self, ib_order_id: int) -> Optional[OrderRecord]:
        """Get order by IB order ID."""
        order_id = self._ib_order_map.get(ib_order_id)
        if order_id:
            return self._orders.get(order_id)
        return None

    def get_active_orders(self) -> List[OrderRecord]:
        """Get all active orders."""
        return list(self._active_orders.values())

    def get_active_orders_for_contract(self, contract_id: int) -> List[OrderRecord]:
        """Get active orders for a specific contract."""
        return [
            order for order in self._active_orders.values()
            if order.contract_id == contract_id
        ]

    def get_all_positions(self) -> List[PositionRecord]:
        """Get all positions (including flat)."""
        return list(self._positions.values())

    def get_open_positions(self) -> List[PositionRecord]:
        """Get positions with non-zero quantity."""
        return [p for p in self._positions.values() if not p.is_flat]

    def get_total_exposure(self) -> int:
        """
        Get total absolute exposure across all contracts.

        Returns:
            Sum of absolute position quantities
        """
        return sum(abs(p.quantity) for p in self._positions.values())

    def get_net_exposure(self) -> int:
        """
        Get net exposure (long - short).

        Returns:
            Net exposure (positive=net long, negative=net short)
        """
        return sum(p.quantity for p in self._positions.values())

    def get_exposure_by_side(self) -> dict:
        """
        Get exposure breakdown by side.

        Returns:
            Dict with 'long' and 'short' total quantities
        """
        long_exposure = sum(p.quantity for p in self._positions.values() if p.quantity > 0)
        short_exposure = sum(abs(p.quantity) for p in self._positions.values() if p.quantity < 0)
        return {
            'long': long_exposure,
            'short': short_exposure,
            'net': long_exposure - short_exposure,
            'gross': long_exposure + short_exposure,
        }

    def check_position_limit(self, contract_id: int, additional_quantity: int) -> bool:
        """
        Check if adding quantity would exceed position limit.

        Args:
            contract_id: Contract ID
            additional_quantity: Signed quantity to add

        Returns:
            True if within limits, False if would exceed
        """
        current = self.get_position_quantity(contract_id)
        new_position = current + additional_quantity
        return abs(new_position) <= self._max_position

    def check_exposure_limit(self, additional_contracts: int) -> bool:
        """
        Check if adding contracts would exceed exposure limit.

        Args:
            additional_contracts: Additional contracts to add

        Returns:
            True if within limits, False if would exceed
        """
        current = self.get_total_exposure()
        return (current + additional_contracts) <= self._max_exposure

    def update_market_prices(self, prices: Dict[int, float]) -> None:
        """
        Update unrealized P&L with current market prices.

        Args:
            prices: Dict mapping contract_id to current price
        """
        for contract_id, price in prices.items():
            if contract_id in self._positions:
                self._positions[contract_id].update_unrealized_pnl(price)

    def set_contract_multiplier(self, contract_id: int, multiplier: float) -> None:
        """
        Set contract multiplier for notional calculations.

        Args:
            contract_id: Contract ID
            multiplier: Contract multiplier (e.g., 50 for ES)
        """
        position = self._get_or_create_position(contract_id)
        position.multiplier = multiplier

    async def reconcile_with_ib(
        self,
        ib_client: Any,
        auto_correct: bool = True,
    ) -> ReconciliationReport:
        """
        Reconcile positions with IB.

        Compares local position tracking with IB's positions and
        optionally corrects mismatches.

        Args:
            ib_client: IBClient instance
            auto_correct: If True, automatically correct mismatches

        Returns:
            ReconciliationReport with results
        """
        report = ReconciliationReport(timestamp=datetime.now(timezone.utc))

        try:
            # Get positions from IB
            ib_positions = await ib_client.request_positions()

            # Build map of IB positions by contract ID
            ib_position_map: Dict[int, Any] = {}
            for pos in ib_positions:
                contract_id = pos.contract.conId
                ib_position_map[contract_id] = pos

            # Check all local positions against IB
            for contract_id, local_pos in self._positions.items():
                if contract_id in ib_position_map:
                    ib_pos = ib_position_map[contract_id]
                    ib_qty = int(ib_pos.position)

                    if local_pos.quantity == ib_qty:
                        report.matches += 1
                        report.details.append({
                            'contract_id': contract_id,
                            'symbol': local_pos.symbol,
                            'result': ReconciliationResult.MATCH.value,
                            'local': local_pos.quantity,
                            'ib': ib_qty,
                        })
                    else:
                        report.mismatches += 1
                        detail = {
                            'contract_id': contract_id,
                            'symbol': local_pos.symbol,
                            'result': ReconciliationResult.MISMATCH.value,
                            'local': local_pos.quantity,
                            'ib': ib_qty,
                            'difference': ib_qty - local_pos.quantity,
                        }
                        report.details.append(detail)

                        self._logger.warning(
                            f"Position mismatch for {local_pos.symbol}: "
                            f"local={local_pos.quantity}, IB={ib_qty}"
                        )

                        # Auto-correct if enabled
                        if auto_correct:
                            self._correct_position(contract_id, ib_qty, ib_pos)
                            report.corrections_applied += 1

                    # Remove from IB map (processed)
                    del ib_position_map[contract_id]

                elif local_pos.quantity != 0:
                    # Position exists locally but not in IB
                    report.local_only += 1
                    report.details.append({
                        'contract_id': contract_id,
                        'symbol': local_pos.symbol,
                        'result': ReconciliationResult.LOCAL_ONLY.value,
                        'local': local_pos.quantity,
                        'ib': 0,
                    })

                    self._logger.warning(
                        f"Position exists locally but not in IB: "
                        f"{local_pos.symbol}={local_pos.quantity}"
                    )

                    if auto_correct:
                        self._correct_position(contract_id, 0, None)
                        report.corrections_applied += 1

            # Check remaining IB positions (not in local tracking)
            for contract_id, ib_pos in ib_position_map.items():
                ib_qty = int(ib_pos.position)
                if ib_qty != 0:
                    report.ib_only += 1
                    report.details.append({
                        'contract_id': contract_id,
                        'symbol': ib_pos.contract.symbol,
                        'result': ReconciliationResult.IB_ONLY.value,
                        'local': 0,
                        'ib': ib_qty,
                    })

                    self._logger.warning(
                        f"Position exists in IB but not locally: "
                        f"{ib_pos.contract.symbol}={ib_qty}"
                    )

                    if auto_correct:
                        self._add_ib_position(ib_pos)
                        report.corrections_applied += 1

            # Store report
            self._last_reconciliation = report

            if not report.is_clean:
                # Notify of mismatch
                for callback in self._on_mismatch:
                    try:
                        await callback(report)
                    except Exception as e:
                        self._logger.error(f"Mismatch callback error: {e}")

            self._audit("RECONCILIATION", report.to_dict())

            self._logger.info(
                f"Reconciliation complete: matches={report.matches}, "
                f"mismatches={report.mismatches}, corrections={report.corrections_applied}"
            )

        except Exception as e:
            self._logger.error(f"Reconciliation failed: {e}")
            report.details.append({'error': str(e)})

        return report

    def _correct_position(
        self,
        contract_id: int,
        correct_quantity: int,
        ib_pos: Any,
    ) -> None:
        """
        Correct local position to match IB.

        Args:
            contract_id: Contract ID
            correct_quantity: Correct quantity from IB
            ib_pos: IB position object (may be None)
        """
        position = self._get_or_create_position(contract_id)

        old_quantity = position.quantity
        position.quantity = correct_quantity
        position.last_update_time = datetime.now(timezone.utc)

        # Try to get entry price from IB
        if ib_pos and correct_quantity != 0:
            position.average_entry_price = ib_pos.avgCost / position.multiplier
            position.symbol = ib_pos.contract.symbol
            position.exchange = ib_pos.contract.exchange
        elif correct_quantity == 0:
            position.average_entry_price = 0.0

        self._audit("POSITION_CORRECTED", {
            'contract_id': contract_id,
            'old_quantity': old_quantity,
            'new_quantity': correct_quantity,
        })

        self._logger.info(
            f"Position corrected: {position.symbol} "
            f"{old_quantity} -> {correct_quantity}"
        )

    def _add_ib_position(self, ib_pos: Any) -> None:
        """
        Add position from IB that doesn't exist locally.

        Args:
            ib_pos: IB position object
        """
        contract_id = ib_pos.contract.conId
        position = self._get_or_create_position(
            contract_id,
            ib_pos.contract.symbol,
            ib_pos.contract.exchange,
        )

        position.quantity = int(ib_pos.position)

        # Get multiplier from contract if available
        if hasattr(ib_pos.contract, 'multiplier') and ib_pos.contract.multiplier:
            position.multiplier = float(ib_pos.contract.multiplier)

        # Calculate average price from average cost
        if position.multiplier > 0 and ib_pos.avgCost:
            position.average_entry_price = ib_pos.avgCost / position.multiplier

        position.last_update_time = datetime.now(timezone.utc)
        position.first_entry_time = position.last_update_time  # Unknown, use now

        self._audit("POSITION_ADDED_FROM_IB", {
            'contract_id': contract_id,
            'symbol': position.symbol,
            'quantity': position.quantity,
            'avg_price': position.average_entry_price,
        })

        self._logger.info(
            f"Position added from IB: {position.symbol}={position.quantity} "
            f"@ {position.average_entry_price:.2f}"
        )

    def _audit(self, event_type: str, data: dict) -> None:
        """Record event to audit log."""
        self._audit_log.append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'event': event_type,
            'data': data,
        })

    def _notify_position_change(
        self,
        contract_id: int,
        old_quantity: int,
        new_quantity: int,
    ) -> None:
        """Notify callbacks of position change."""
        for callback in self._on_position_change:
            try:
                callback(contract_id, old_quantity, new_quantity)
            except Exception as e:
                self._logger.error(f"Position change callback error: {e}")

    def _notify_order_update(self, order: OrderRecord) -> None:
        """Notify callbacks of order update."""
        for callback in self._on_order_update:
            try:
                callback(order)
            except Exception as e:
                self._logger.error(f"Order update callback error: {e}")

    def register_position_callback(
        self,
        callback: Callable[[int, int, int], None],
    ) -> None:
        """
        Register callback for position changes.

        Args:
            callback: Function(contract_id, old_quantity, new_quantity)
        """
        self._on_position_change.append(callback)

    def register_order_callback(
        self,
        callback: Callable[[OrderRecord], None],
    ) -> None:
        """
        Register callback for order updates.

        Args:
            callback: Function(order_record)
        """
        self._on_order_update.append(callback)

    def register_mismatch_callback(
        self,
        callback: Callable[[ReconciliationReport], Awaitable[None]],
    ) -> None:
        """
        Register async callback for reconciliation mismatches.

        Args:
            callback: Async function(report)
        """
        self._on_mismatch.append(callback)

    def get_state(self) -> dict:
        """
        Get complete manager state for monitoring.

        Returns:
            Dictionary with all state information
        """
        exposure = self.get_exposure_by_side()

        return {
            'total_orders': self._total_orders,
            'active_orders': len(self._active_orders),
            'total_fills': self._total_fills,
            'positions_count': len(self._positions),
            'open_positions_count': len(self.get_open_positions()),
            'exposure': exposure,
            'total_realized_pnl': self._total_realized_pnl,
            'total_unrealized_pnl': sum(p.unrealized_pnl for p in self._positions.values()),
            'total_commission': sum(p.total_commission for p in self._positions.values()),
            'max_exposure': self._max_exposure,
            'max_position': self._max_position,
            'last_reconciliation': (
                self._last_reconciliation.to_dict()
                if self._last_reconciliation else None
            ),
            'positions': {
                cid: pos.to_dict()
                for cid, pos in self._positions.items()
                if not pos.is_flat
            },
        }

    def get_summary(self) -> dict:
        """
        Get concise summary for dashboard.

        Returns:
            Summary dictionary
        """
        exposure = self.get_exposure_by_side()

        return {
            'active_orders': len(self._active_orders),
            'open_positions': len(self.get_open_positions()),
            'gross_exposure': exposure['gross'],
            'net_exposure': exposure['net'],
            'realized_pnl': self._total_realized_pnl,
            'unrealized_pnl': sum(p.unrealized_pnl for p in self._positions.values()),
            'total_pnl': (
                self._total_realized_pnl +
                sum(p.unrealized_pnl for p in self._positions.values())
            ),
            'reconciliation_clean': (
                self._last_reconciliation.is_clean
                if self._last_reconciliation else True
            ),
        }

    def get_audit_log(self, count: int = 100) -> List[dict]:
        """
        Get recent audit log entries.

        Args:
            count: Number of entries to return

        Returns:
            List of audit log entries
        """
        return list(self._audit_log)[-count:]

    def reset(self) -> None:
        """Reset all tracking data."""
        self._orders.clear()
        self._ib_order_map.clear()
        self._active_orders.clear()
        self._positions.clear()
        self._audit_log.clear()
        self._last_reconciliation = None
        self._total_orders = 0
        self._total_fills = 0
        self._total_realized_pnl = 0.0

        self._logger.info("PositionManager reset")

    @property
    def total_realized_pnl(self) -> float:
        """Get total realized P&L."""
        return self._total_realized_pnl

    @property
    def total_unrealized_pnl(self) -> float:
        """Get total unrealized P&L."""
        return sum(p.unrealized_pnl for p in self._positions.values())

    @property
    def total_pnl(self) -> float:
        """Get total P&L (realized + unrealized)."""
        return self._total_realized_pnl + self.total_unrealized_pnl
