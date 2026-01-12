"""
Confidence-Gated Order Executor with IB Integration.

Executes orders through Interactive Brokers when confidence thresholds
are met. Rejects low-confidence signals and tracks all execution
decisions for learning feedback.

Key Features:
- Confidence gating before order execution
- IB integration for order placement and tracking
- Session-aware execution (respects market hours)
- Position scaling based on confidence
- Complete execution tracking for learning
- Risk checks before execution

CRITICAL: Only execute when confidence meets threshold.
Low-confidence signals MUST be rejected, not executed with smaller size.
"""

import asyncio
import logging
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Awaitable

# Support both ib_async and ib_insync for compatibility
try:
    from ib_async import Order, MarketOrder, LimitOrder, Contract, Trade, OrderStatus
    IB_LIBRARY = "ib_async"
except ImportError:
    try:
        from ib_insync import Order, MarketOrder, LimitOrder, Contract, Trade, OrderStatus
        IB_LIBRARY = "ib_insync"
    except ImportError:
        Order = Any  # type: ignore
        MarketOrder = Any  # type: ignore
        LimitOrder = Any  # type: ignore
        Contract = Any  # type: ignore
        Trade = Any  # type: ignore
        OrderStatus = Any  # type: ignore
        IB_LIBRARY = None

from config import get_config
from src.core.ib_client import IBClient, ConnectionState
from src.trading.confidence import (
    ConfidenceTracker,
    ConfidenceLevel,
    GateDecision,
    ThresholdMode,
)
from src.trading.predictor import PredictionResult, PredictionSignal


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"


class ExecutionStatus(Enum):
    """Execution status enumeration."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL_FILL = "partial_fill"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    ERROR = "error"


class ExecutionDecision(Enum):
    """Decision from execution gating."""
    EXECUTE = "execute"
    REJECT_LOW_CONFIDENCE = "reject_low_confidence"
    REJECT_SESSION_CLOSED = "reject_session_closed"
    REJECT_RISK_LIMIT = "reject_risk_limit"
    REJECT_NO_CONNECTION = "reject_no_connection"
    REJECT_DATA_STALE = "reject_data_stale"
    REJECT_POSITION_LIMIT = "reject_position_limit"
    REJECT_DISABLED = "reject_disabled"


@dataclass
class ExecutionRequest:
    """
    Request for order execution.

    Contains all information needed to execute an order including
    the prediction that generated it.
    """
    request_id: str
    contract: Contract
    side: OrderSide
    quantity: int
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None

    # Source prediction
    prediction: Optional[PredictionResult] = None
    confidence: Optional[ConfidenceLevel] = None

    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    reason: str = ""

    def to_dict(self) -> dict:
        """Export as dictionary."""
        return {
            'request_id': self.request_id,
            'contract_id': self.contract.conId if self.contract else None,
            'symbol': getattr(self.contract, 'symbol', 'unknown'),
            'side': self.side.value,
            'quantity': self.quantity,
            'order_type': self.order_type.value,
            'limit_price': self.limit_price,
            'prediction_id': self.prediction.prediction_id if self.prediction else None,
            'confidence': self.confidence.value if self.confidence else None,
            'timestamp': self.timestamp.isoformat(),
            'reason': self.reason,
        }


@dataclass
class ExecutionResult:
    """
    Result of an execution attempt.

    Contains the final status and all execution details for tracking.
    """
    request_id: str
    request: ExecutionRequest
    decision: ExecutionDecision
    status: ExecutionStatus
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # IB order details (populated after submission)
    order_id: Optional[int] = None
    trade: Optional[Trade] = None

    # Fill details
    filled_quantity: int = 0
    average_fill_price: float = 0.0
    commission: float = 0.0

    # Timing
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None

    # Rejection details
    rejection_reason: Optional[str] = None

    def to_dict(self) -> dict:
        """Export as dictionary."""
        return {
            'request_id': self.request_id,
            'decision': self.decision.value,
            'status': self.status.value,
            'timestamp': self.timestamp.isoformat(),
            'order_id': self.order_id,
            'filled_quantity': self.filled_quantity,
            'average_fill_price': self.average_fill_price,
            'commission': self.commission,
            'submitted_at': self.submitted_at.isoformat() if self.submitted_at else None,
            'filled_at': self.filled_at.isoformat() if self.filled_at else None,
            'rejection_reason': self.rejection_reason,
        }


@dataclass
class ExecutionStats:
    """
    Statistics tracking for execution performance.

    Maintains running statistics for monitoring and learning feedback.
    """
    total_requests: int = 0
    executed_count: int = 0
    rejected_count: int = 0

    # Rejection breakdown
    rejections_by_reason: Dict[str, int] = field(default_factory=dict)

    # Fill statistics
    total_filled_quantity: int = 0
    total_commission: float = 0.0

    # Confidence tracking
    executed_confidence_sum: float = 0.0
    rejected_confidence_sum: float = 0.0

    # Recent executions
    recent_executions: deque = field(default_factory=lambda: deque(maxlen=100))

    # Timing
    last_execution_time: Optional[datetime] = None

    def record_execution(self, result: ExecutionResult) -> None:
        """Record an execution result."""
        self.total_requests += 1
        self.last_execution_time = result.timestamp

        if result.decision == ExecutionDecision.EXECUTE:
            self.executed_count += 1
            self.total_filled_quantity += result.filled_quantity
            self.total_commission += result.commission

            if result.request.confidence:
                self.executed_confidence_sum += result.request.confidence.value
        else:
            self.rejected_count += 1
            reason = result.decision.value
            self.rejections_by_reason[reason] = (
                self.rejections_by_reason.get(reason, 0) + 1
            )

            if result.request.confidence:
                self.rejected_confidence_sum += result.request.confidence.value

        self.recent_executions.append(result.to_dict())

    @property
    def execution_rate(self) -> float:
        """Get rate of executed orders."""
        if self.total_requests == 0:
            return 0.0
        return self.executed_count / self.total_requests

    @property
    def average_executed_confidence(self) -> float:
        """Get average confidence of executed orders."""
        if self.executed_count == 0:
            return 0.0
        return self.executed_confidence_sum / self.executed_count

    @property
    def average_rejected_confidence(self) -> float:
        """Get average confidence of rejected orders."""
        if self.rejected_count == 0:
            return 0.0
        return self.rejected_confidence_sum / self.rejected_count

    def to_dict(self) -> dict:
        """Export statistics as dictionary."""
        return {
            'total_requests': self.total_requests,
            'executed_count': self.executed_count,
            'rejected_count': self.rejected_count,
            'execution_rate': self.execution_rate,
            'rejections_by_reason': dict(self.rejections_by_reason),
            'total_filled_quantity': self.total_filled_quantity,
            'total_commission': self.total_commission,
            'average_executed_confidence': self.average_executed_confidence,
            'average_rejected_confidence': self.average_rejected_confidence,
            'last_execution_time': (
                self.last_execution_time.isoformat()
                if self.last_execution_time else None
            ),
        }

    def reset(self) -> None:
        """Reset all statistics."""
        self.total_requests = 0
        self.executed_count = 0
        self.rejected_count = 0
        self.rejections_by_reason.clear()
        self.total_filled_quantity = 0
        self.total_commission = 0.0
        self.executed_confidence_sum = 0.0
        self.rejected_confidence_sum = 0.0
        self.recent_executions.clear()
        self.last_execution_time = None


class Executor:
    """
    Confidence-Gated Order Executor.

    Executes orders through IB only when confidence thresholds are met.
    Tracks all execution decisions for learning feedback.

    CRITICAL BEHAVIOR:
    - Confidence below threshold = REJECT (not scale down)
    - No execution without IB connection
    - All decisions are logged and tracked
    - Outcomes are fed back to confidence tracker

    Usage:
        executor = Executor(ib_client, confidence_tracker)

        # Execute from prediction
        result = await executor.execute_prediction(prediction, confidence)

        # Or create manual request
        request = ExecutionRequest(
            request_id=str(uuid.uuid4()),
            contract=contract,
            side=OrderSide.BUY,
            quantity=1,
        )
        result = await executor.execute(request)

        # Record outcome for learning
        executor.record_outcome(result.request_id, success=True, pnl=150.0)
    """

    def __init__(
        self,
        ib_client: Optional[IBClient] = None,
        confidence_tracker: Optional[ConfidenceTracker] = None,
        session_manager: Any = None,
        max_position_per_contract: int = None,
        max_total_exposure: int = None,
    ):
        """
        Initialize Executor.

        Args:
            ib_client: IBClient instance for order execution
            confidence_tracker: ConfidenceTracker for gating decisions
            session_manager: Optional SessionManager for session checks
            max_position_per_contract: Maximum contracts per symbol
            max_total_exposure: Maximum total contracts across all symbols
        """
        self._config = get_config()
        self._logger = logging.getLogger(__name__)

        # Dependencies
        self._ib_client = ib_client
        self._confidence_tracker = confidence_tracker
        self._session_manager = session_manager

        # Risk limits (configurable)
        self._max_position_per_contract = max_position_per_contract or 10
        self._max_total_exposure = max_total_exposure or 50

        # State
        self._enabled = True
        self._pending_orders: Dict[str, ExecutionResult] = {}
        self._completed_orders: Dict[str, ExecutionResult] = {}
        self._current_positions: Dict[int, int] = {}  # contract_id -> quantity

        # Statistics
        self._stats = ExecutionStats()
        self._per_contract_stats: Dict[int, ExecutionStats] = {}

        # Callbacks
        self._on_execution_callbacks: List[Callable[[ExecutionResult], None]] = []
        self._on_fill_callbacks: List[Callable[[ExecutionResult], Awaitable[None]]] = []
        self._on_rejection_callbacks: List[Callable[[ExecutionResult], None]] = []

        # Outcome tracking for learning
        self._execution_outcomes: Dict[str, dict] = {}

        self._logger.info(
            f"Executor initialized: max_position={self._max_position_per_contract}, "
            f"max_exposure={self._max_total_exposure}"
        )

    def set_ib_client(self, ib_client: IBClient) -> None:
        """Set or update the IB client."""
        self._ib_client = ib_client
        self._logger.info("IB client updated")

    def set_confidence_tracker(self, confidence_tracker: ConfidenceTracker) -> None:
        """Set or update the confidence tracker."""
        self._confidence_tracker = confidence_tracker
        self._logger.info("Confidence tracker updated")

    def set_session_manager(self, session_manager: Any) -> None:
        """Set or update the session manager."""
        self._session_manager = session_manager
        self._logger.info("Session manager updated")

    def enable(self) -> None:
        """Enable order execution."""
        self._enabled = True
        self._logger.info("Executor enabled")

    def disable(self) -> None:
        """Disable order execution."""
        self._enabled = False
        self._logger.info("Executor disabled")

    @property
    def is_enabled(self) -> bool:
        """Check if executor is enabled."""
        return self._enabled

    @property
    def is_ready(self) -> bool:
        """Check if executor is ready to execute orders."""
        if not self._enabled:
            return False
        if self._ib_client is None:
            return False
        if not self._ib_client.is_connected:
            return False
        return True

    def _generate_request_id(self) -> str:
        """Generate unique request ID."""
        return str(uuid.uuid4())

    def _get_contract_stats(self, contract_id: int) -> ExecutionStats:
        """Get or create stats for a contract."""
        if contract_id not in self._per_contract_stats:
            self._per_contract_stats[contract_id] = ExecutionStats()
        return self._per_contract_stats[contract_id]

    def _check_execution_allowed(
        self,
        request: ExecutionRequest,
    ) -> tuple[ExecutionDecision, Optional[str]]:
        """
        Check if execution is allowed based on all gates.

        Returns:
            Tuple of (decision, rejection_reason)
        """
        # Check if executor is enabled
        if not self._enabled:
            return ExecutionDecision.REJECT_DISABLED, "Executor is disabled"

        # Check IB connection
        if self._ib_client is None:
            return ExecutionDecision.REJECT_NO_CONNECTION, "No IB client configured"

        if not self._ib_client.is_connected:
            return ExecutionDecision.REJECT_NO_CONNECTION, "Not connected to IB"

        # Check session (if session manager available)
        if self._session_manager is not None:
            contract = request.contract
            exchange = getattr(contract, 'exchange', 'CME')

            if hasattr(self._session_manager, 'should_allow_entry'):
                if not self._session_manager.should_allow_entry(exchange):
                    return (
                        ExecutionDecision.REJECT_SESSION_CLOSED,
                        f"Session closed or closing soon for {exchange}"
                    )
            elif hasattr(self._session_manager, 'is_market_open'):
                if not self._session_manager.is_market_open(exchange):
                    return (
                        ExecutionDecision.REJECT_SESSION_CLOSED,
                        f"Market closed for {exchange}"
                    )

        # Check confidence gate (if confidence provided)
        if request.confidence is not None and self._confidence_tracker is not None:
            gate_decision = self._confidence_tracker.gate_entry(
                request.confidence,
                contract_id=request.contract.conId if request.contract else None,
            )

            if gate_decision != GateDecision.ALLOWED:
                return (
                    ExecutionDecision.REJECT_LOW_CONFIDENCE,
                    f"Confidence {request.confidence.value:.3f} below threshold"
                )

        # Check position limits
        contract_id = request.contract.conId if request.contract else 0
        current_position = self._current_positions.get(contract_id, 0)

        if request.side == OrderSide.BUY:
            new_position = current_position + request.quantity
        else:
            new_position = current_position - request.quantity

        if abs(new_position) > self._max_position_per_contract:
            return (
                ExecutionDecision.REJECT_POSITION_LIMIT,
                f"Would exceed position limit: {abs(new_position)} > {self._max_position_per_contract}"
            )

        # Check total exposure
        total_exposure = sum(abs(p) for p in self._current_positions.values())
        exposure_change = request.quantity
        if total_exposure + exposure_change > self._max_total_exposure:
            return (
                ExecutionDecision.REJECT_RISK_LIMIT,
                f"Would exceed total exposure: {total_exposure + exposure_change} > {self._max_total_exposure}"
            )

        return ExecutionDecision.EXECUTE, None

    async def execute(
        self,
        request: ExecutionRequest,
    ) -> ExecutionResult:
        """
        Execute an order request.

        Performs all confidence and risk checks before execution.

        Args:
            request: ExecutionRequest with order details

        Returns:
            ExecutionResult with execution status
        """
        # Check if execution is allowed
        decision, rejection_reason = self._check_execution_allowed(request)

        if decision != ExecutionDecision.EXECUTE:
            result = ExecutionResult(
                request_id=request.request_id,
                request=request,
                decision=decision,
                status=ExecutionStatus.REJECTED,
                rejection_reason=rejection_reason,
            )

            self._record_result(result)
            self._logger.info(
                f"Order rejected: {rejection_reason} "
                f"(request_id={request.request_id})"
            )

            # Invoke rejection callbacks
            for callback in self._on_rejection_callbacks:
                try:
                    callback(result)
                except Exception as e:
                    self._logger.error(f"Rejection callback error: {e}")

            return result

        # Create order
        try:
            order = self._create_order(request)
        except Exception as e:
            self._logger.error(f"Error creating order: {e}")
            return ExecutionResult(
                request_id=request.request_id,
                request=request,
                decision=ExecutionDecision.EXECUTE,
                status=ExecutionStatus.ERROR,
                rejection_reason=str(e),
            )

        # Submit order to IB
        try:
            trade = self._ib_client.ib.placeOrder(request.contract, order)

            result = ExecutionResult(
                request_id=request.request_id,
                request=request,
                decision=ExecutionDecision.EXECUTE,
                status=ExecutionStatus.SUBMITTED,
                order_id=trade.order.orderId,
                trade=trade,
                submitted_at=datetime.now(timezone.utc),
            )

            # Track pending order
            self._pending_orders[request.request_id] = result

            # Set up fill callback
            trade.filledEvent += lambda t: asyncio.create_task(
                self._handle_fill(request.request_id, t)
            )
            trade.cancelledEvent += lambda t: self._handle_cancelled(
                request.request_id, t
            )

            self._logger.info(
                f"Order submitted: {request.side.value} {request.quantity} "
                f"{getattr(request.contract, 'symbol', 'unknown')} "
                f"(order_id={trade.order.orderId})"
            )

            # Invoke execution callbacks
            for callback in self._on_execution_callbacks:
                try:
                    callback(result)
                except Exception as e:
                    self._logger.error(f"Execution callback error: {e}")

            return result

        except Exception as e:
            self._logger.error(f"Error submitting order: {e}")
            result = ExecutionResult(
                request_id=request.request_id,
                request=request,
                decision=ExecutionDecision.EXECUTE,
                status=ExecutionStatus.ERROR,
                rejection_reason=str(e),
            )
            self._record_result(result)
            return result

    async def execute_prediction(
        self,
        prediction: PredictionResult,
        confidence: ConfidenceLevel,
        contract: Contract,
        quantity: int = 1,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
    ) -> ExecutionResult:
        """
        Execute an order based on a prediction.

        Convenience method to create request from prediction and execute.

        Args:
            prediction: PredictionResult from predictor
            confidence: ConfidenceLevel for gating
            contract: IB Contract to trade
            quantity: Number of contracts
            order_type: Market or limit order
            limit_price: Limit price for limit orders

        Returns:
            ExecutionResult with execution status
        """
        # Determine side from prediction signal
        if prediction.signal == PredictionSignal.BUY:
            side = OrderSide.BUY
        elif prediction.signal == PredictionSignal.SELL:
            side = OrderSide.SELL
        else:
            # Neutral signal - reject
            return ExecutionResult(
                request_id=self._generate_request_id(),
                request=ExecutionRequest(
                    request_id=self._generate_request_id(),
                    contract=contract,
                    side=OrderSide.BUY,  # Placeholder
                    quantity=quantity,
                    prediction=prediction,
                    confidence=confidence,
                    reason="Neutral prediction signal",
                ),
                decision=ExecutionDecision.REJECT_LOW_CONFIDENCE,
                status=ExecutionStatus.REJECTED,
                rejection_reason="Neutral prediction signal - no directional bias",
            )

        request = ExecutionRequest(
            request_id=self._generate_request_id(),
            contract=contract,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            prediction=prediction,
            confidence=confidence,
            reason=f"Prediction {prediction.prediction_id}: {prediction.signal.name}",
        )

        return await self.execute(request)

    def _create_order(self, request: ExecutionRequest) -> Order:
        """Create IB Order from request."""
        action = request.side.value

        if request.order_type == OrderType.MARKET:
            order = MarketOrder(action, request.quantity)
        elif request.order_type == OrderType.LIMIT:
            if request.limit_price is None:
                raise ValueError("Limit price required for limit orders")
            order = LimitOrder(action, request.quantity, request.limit_price)
        else:
            raise ValueError(f"Unknown order type: {request.order_type}")

        return order

    async def _handle_fill(self, request_id: str, trade: Trade) -> None:
        """Handle order fill event."""
        if request_id not in self._pending_orders:
            self._logger.warning(f"Fill for unknown request: {request_id}")
            return

        result = self._pending_orders.pop(request_id)
        result.status = ExecutionStatus.FILLED
        result.filled_quantity = int(trade.orderStatus.filled)
        result.average_fill_price = trade.orderStatus.avgFillPrice
        result.filled_at = datetime.now(timezone.utc)

        # Calculate commission from fills
        if trade.fills:
            result.commission = sum(f.commissionReport.commission for f in trade.fills)

        # Update position tracking
        contract_id = result.request.contract.conId if result.request.contract else 0
        current = self._current_positions.get(contract_id, 0)
        if result.request.side == OrderSide.BUY:
            self._current_positions[contract_id] = current + result.filled_quantity
        else:
            self._current_positions[contract_id] = current - result.filled_quantity

        # Move to completed
        self._completed_orders[request_id] = result
        self._record_result(result)

        self._logger.info(
            f"Order filled: {result.request.side.value} {result.filled_quantity} "
            f"@ {result.average_fill_price:.2f} "
            f"(order_id={result.order_id}, commission={result.commission:.2f})"
        )

        # Invoke fill callbacks
        for callback in self._on_fill_callbacks:
            try:
                await callback(result)
            except Exception as e:
                self._logger.error(f"Fill callback error: {e}")

    def _handle_cancelled(self, request_id: str, trade: Trade) -> None:
        """Handle order cancellation event."""
        if request_id not in self._pending_orders:
            return

        result = self._pending_orders.pop(request_id)
        result.status = ExecutionStatus.CANCELLED

        self._completed_orders[request_id] = result
        self._record_result(result)

        self._logger.info(f"Order cancelled: (order_id={result.order_id})")

    def _record_result(self, result: ExecutionResult) -> None:
        """Record execution result to statistics."""
        self._stats.record_execution(result)

        contract_id = (
            result.request.contract.conId if result.request.contract else 0
        )
        self._get_contract_stats(contract_id).record_execution(result)

    def record_outcome(
        self,
        request_id: str,
        success: bool,
        pnl: float = 0.0,
        exit_price: Optional[float] = None,
    ) -> None:
        """
        Record trade outcome for learning feedback.

        Args:
            request_id: ID of the execution request
            success: Whether trade was profitable
            pnl: Profit/loss amount
            exit_price: Exit price of the trade
        """
        # Store outcome
        self._execution_outcomes[request_id] = {
            'success': success,
            'pnl': pnl,
            'exit_price': exit_price,
            'timestamp': datetime.now(timezone.utc),
        }

        # Update confidence tracker
        if self._confidence_tracker is not None:
            result = self._completed_orders.get(request_id)
            if result and result.request.confidence:
                confidence_used = result.request.confidence.value
                contract_id = (
                    result.request.contract.conId
                    if result.request.contract else None
                )
                self._confidence_tracker.record_outcome(
                    success=success,
                    confidence_used=confidence_used,
                    contract_id=contract_id,
                )
                self._logger.debug(
                    f"Recorded outcome for learning: success={success}, "
                    f"confidence={confidence_used:.3f}, pnl={pnl:.2f}"
                )

    async def cancel_order(self, request_id: str) -> bool:
        """
        Cancel a pending order.

        Args:
            request_id: ID of the execution request

        Returns:
            True if cancellation was sent
        """
        if request_id not in self._pending_orders:
            self._logger.warning(f"No pending order found for: {request_id}")
            return False

        result = self._pending_orders[request_id]
        if result.trade is None:
            return False

        try:
            self._ib_client.ib.cancelOrder(result.trade.order)
            self._logger.info(f"Cancel sent for order_id={result.order_id}")
            return True
        except Exception as e:
            self._logger.error(f"Error cancelling order: {e}")
            return False

    async def cancel_all_orders(self) -> int:
        """
        Cancel all pending orders.

        Returns:
            Number of cancellations sent
        """
        cancelled = 0
        for request_id in list(self._pending_orders.keys()):
            if await self.cancel_order(request_id):
                cancelled += 1
        return cancelled

    def update_position(self, contract_id: int, quantity: int) -> None:
        """
        Update position tracking externally (e.g., from IB reconciliation).

        Args:
            contract_id: IB contract ID
            quantity: Current position quantity (positive=long, negative=short)
        """
        self._current_positions[contract_id] = quantity
        self._logger.debug(f"Position updated: contract_id={contract_id}, qty={quantity}")

    def get_position(self, contract_id: int) -> int:
        """Get current position for a contract."""
        return self._current_positions.get(contract_id, 0)

    def get_total_exposure(self) -> int:
        """Get total absolute exposure across all contracts."""
        return sum(abs(p) for p in self._current_positions.values())

    def register_execution_callback(
        self,
        callback: Callable[[ExecutionResult], None],
    ) -> None:
        """Register callback for execution events."""
        self._on_execution_callbacks.append(callback)

    def register_fill_callback(
        self,
        callback: Callable[[ExecutionResult], Awaitable[None]],
    ) -> None:
        """Register async callback for fill events."""
        self._on_fill_callbacks.append(callback)

    def register_rejection_callback(
        self,
        callback: Callable[[ExecutionResult], None],
    ) -> None:
        """Register callback for rejection events."""
        self._on_rejection_callbacks.append(callback)

    def get_pending_orders(self) -> List[ExecutionResult]:
        """Get all pending orders."""
        return list(self._pending_orders.values())

    def get_recent_executions(self, count: int = 10) -> List[dict]:
        """Get recent executions."""
        executions = list(self._stats.recent_executions)
        return executions[-count:]

    def get_state(self) -> dict:
        """
        Get complete executor state for monitoring.

        Returns:
            Dictionary with all state information
        """
        return {
            'enabled': self._enabled,
            'is_ready': self.is_ready,
            'max_position_per_contract': self._max_position_per_contract,
            'max_total_exposure': self._max_total_exposure,
            'pending_orders_count': len(self._pending_orders),
            'completed_orders_count': len(self._completed_orders),
            'current_positions': dict(self._current_positions),
            'total_exposure': self.get_total_exposure(),
            'stats': self._stats.to_dict(),
        }

    def get_summary(self) -> dict:
        """
        Get concise summary for dashboard display.

        Returns:
            Summary dictionary
        """
        return {
            'enabled': self._enabled,
            'is_ready': self.is_ready,
            'pending_orders': len(self._pending_orders),
            'total_executions': self._stats.executed_count,
            'total_rejections': self._stats.rejected_count,
            'execution_rate': self._stats.execution_rate,
            'total_exposure': self.get_total_exposure(),
            'positions_count': len([p for p in self._current_positions.values() if p != 0]),
            'avg_executed_confidence': self._stats.average_executed_confidence,
            'total_commission': self._stats.total_commission,
        }

    def get_contract_summary(self, contract_id: int) -> Optional[dict]:
        """
        Get summary for a specific contract.

        Args:
            contract_id: Contract ID

        Returns:
            Summary dict or None if not tracked
        """
        if contract_id not in self._per_contract_stats:
            return None

        stats = self._per_contract_stats[contract_id]
        return {
            'position': self._current_positions.get(contract_id, 0),
            'stats': stats.to_dict(),
        }

    def reset_stats(self) -> None:
        """Reset all statistics."""
        self._stats.reset()
        self._per_contract_stats.clear()
        self._execution_outcomes.clear()
        self._logger.info("Executor statistics reset")

    def clear_completed_orders(self, older_than_hours: int = 24) -> int:
        """
        Clear old completed orders.

        Args:
            older_than_hours: Clear orders older than this

        Returns:
            Number of orders cleared
        """
        now = datetime.now(timezone.utc)
        cutoff = now.timestamp() - (older_than_hours * 3600)
        cleared = 0

        for request_id in list(self._completed_orders.keys()):
            result = self._completed_orders[request_id]
            if result.timestamp.timestamp() < cutoff:
                del self._completed_orders[request_id]
                cleared += 1

        if cleared > 0:
            self._logger.info(f"Cleared {cleared} old completed orders")

        return cleared

    @property
    def stats(self) -> ExecutionStats:
        """Get global execution statistics."""
        return self._stats

    @property
    def positions(self) -> Dict[int, int]:
        """Get current positions."""
        return dict(self._current_positions)
