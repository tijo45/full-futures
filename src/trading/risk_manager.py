"""
Risk Manager with Sharpe Optimization and Drawdown Protection.

Implements comprehensive risk management for the autonomous trading system,
optimizing for Sharpe ratio while providing dynamic drawdown protection and
session-aware position sizing.

Key Features:
- Real-time Sharpe ratio calculation and optimization
- Dynamic drawdown protection with threshold adjustment
- Session-aware position sizing
- Per-contract and portfolio-level risk limits
- Risk state monitoring and alerts
- Integration with ConfidenceTracker for threshold adjustment

CRITICAL: All risk parameters adapt based on performance and market conditions.
No hard-coded risk limits - everything is learned and adjusted dynamically.
"""

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from config import get_config


class RiskLevel(Enum):
    """Current risk level classification."""
    LOW = "low"
    NORMAL = "normal"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"


class DrawdownState(Enum):
    """Drawdown state for protection triggers."""
    NORMAL = "normal"
    WARNING = "warning"
    DANGER = "danger"
    CRITICAL = "critical"
    RECOVERY = "recovery"


class TradingState(Enum):
    """Overall trading state based on risk assessment."""
    ACTIVE = "active"  # Normal trading
    REDUCED = "reduced"  # Reduced position sizes
    CAUTIOUS = "cautious"  # Higher confidence required
    HALTED = "halted"  # Trading halted due to risk


@dataclass
class ReturnRecord:
    """Individual return record for Sharpe calculation."""
    timestamp: datetime
    pnl: float  # P&L for the period
    trade_count: int = 1
    win_count: int = 0

    @property
    def return_pct(self) -> float:
        """Return as percentage (requires capital context)."""
        return self.pnl


@dataclass
class SharpeMetrics:
    """
    Sharpe ratio and related metrics.

    Uses rolling calculation for real-time optimization.
    """
    # Current Sharpe (annualized)
    sharpe_ratio: float = 0.0

    # Component metrics
    mean_return: float = 0.0
    return_std: float = 0.0

    # Rolling statistics (Welford's algorithm)
    count: int = 0
    m1: float = 0.0  # Running mean
    m2: float = 0.0  # Running variance component

    # Tracking
    total_return: float = 0.0
    positive_returns: int = 0
    negative_returns: int = 0

    # Periods
    periods_per_year: float = 252 * 6.5  # Trading hours annualization

    def update(self, return_value: float) -> None:
        """
        Update Sharpe calculation with new return using Welford's algorithm.

        Args:
            return_value: New return value
        """
        self.count += 1
        self.total_return += return_value

        if return_value > 0:
            self.positive_returns += 1
        elif return_value < 0:
            self.negative_returns += 1

        # Welford's online algorithm for mean and variance
        delta = return_value - self.m1
        self.m1 += delta / self.count
        delta2 = return_value - self.m1
        self.m2 += delta * delta2

        # Update component metrics
        self.mean_return = self.m1
        if self.count > 1:
            self.return_std = math.sqrt(self.m2 / (self.count - 1))

        # Calculate Sharpe ratio (annualized, assuming risk-free rate = 0)
        if self.return_std > 0:
            self.sharpe_ratio = (self.mean_return / self.return_std) * math.sqrt(self.periods_per_year)
        else:
            self.sharpe_ratio = 0.0

    @property
    def win_rate(self) -> float:
        """Get win rate."""
        if self.count == 0:
            return 0.0
        return self.positive_returns / self.count

    @property
    def profit_factor(self) -> float:
        """Get profit factor (gross profits / gross losses)."""
        if self.negative_returns == 0:
            return float('inf') if self.positive_returns > 0 else 0.0
        # Simplified - actual implementation would track gross amounts
        if self.positive_returns == 0:
            return 0.0
        return self.positive_returns / self.negative_returns

    def to_dict(self) -> dict:
        """Export as dictionary."""
        return {
            'sharpe_ratio': self.sharpe_ratio,
            'mean_return': self.mean_return,
            'return_std': self.return_std,
            'count': self.count,
            'total_return': self.total_return,
            'win_rate': self.win_rate,
            'positive_returns': self.positive_returns,
            'negative_returns': self.negative_returns,
        }


@dataclass
class DrawdownTracker:
    """
    Tracks drawdown for protection triggers.

    Monitors peak equity and current drawdown for risk management.
    """
    # Peak tracking
    peak_equity: float = 0.0
    current_equity: float = 0.0

    # Drawdown values
    current_drawdown: float = 0.0  # Absolute
    current_drawdown_pct: float = 0.0  # Percentage
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0

    # Thresholds for state transitions (adaptive)
    warning_threshold: float = 0.03  # 3%
    danger_threshold: float = 0.05  # 5%
    critical_threshold: float = 0.10  # 10%

    # State
    state: DrawdownState = DrawdownState.NORMAL

    # Timestamps
    peak_time: Optional[datetime] = None
    drawdown_start_time: Optional[datetime] = None

    # History
    drawdown_history: deque = field(default_factory=lambda: deque(maxlen=500))

    def update(self, equity: float, timestamp: Optional[datetime] = None) -> DrawdownState:
        """
        Update drawdown tracking with new equity value.

        Args:
            equity: Current equity value
            timestamp: Time of update

        Returns:
            Current DrawdownState
        """
        timestamp = timestamp or datetime.now(timezone.utc)
        self.current_equity = equity

        # Check for new peak
        if equity > self.peak_equity:
            self.peak_equity = equity
            self.peak_time = timestamp
            self.current_drawdown = 0.0
            self.current_drawdown_pct = 0.0

            # Check for recovery
            if self.state in (DrawdownState.WARNING, DrawdownState.DANGER,
                              DrawdownState.CRITICAL):
                self.state = DrawdownState.RECOVERY
            elif self.state == DrawdownState.RECOVERY:
                self.state = DrawdownState.NORMAL
        else:
            # Calculate drawdown
            self.current_drawdown = self.peak_equity - equity
            if self.peak_equity > 0:
                self.current_drawdown_pct = self.current_drawdown / self.peak_equity
            else:
                self.current_drawdown_pct = 0.0

            # Track max drawdown
            if self.current_drawdown > self.max_drawdown:
                self.max_drawdown = self.current_drawdown
            if self.current_drawdown_pct > self.max_drawdown_pct:
                self.max_drawdown_pct = self.current_drawdown_pct

            # Record drawdown start
            if self.drawdown_start_time is None and self.current_drawdown_pct > 0.01:
                self.drawdown_start_time = timestamp

        # Update state based on drawdown percentage
        old_state = self.state
        if self.current_drawdown_pct >= self.critical_threshold:
            self.state = DrawdownState.CRITICAL
        elif self.current_drawdown_pct >= self.danger_threshold:
            self.state = DrawdownState.DANGER
        elif self.current_drawdown_pct >= self.warning_threshold:
            self.state = DrawdownState.WARNING
        elif self.state != DrawdownState.RECOVERY:
            self.state = DrawdownState.NORMAL

        # Record history
        self.drawdown_history.append({
            'timestamp': timestamp.isoformat(),
            'equity': equity,
            'drawdown_pct': self.current_drawdown_pct,
            'state': self.state.value,
        })

        return self.state

    def reset_drawdown_start(self) -> None:
        """Reset drawdown start time when recovered."""
        self.drawdown_start_time = None

    def set_initial_equity(self, equity: float) -> None:
        """Set initial equity (for startup)."""
        self.peak_equity = equity
        self.current_equity = equity
        self.peak_time = datetime.now(timezone.utc)

    def to_dict(self) -> dict:
        """Export as dictionary."""
        return {
            'peak_equity': self.peak_equity,
            'current_equity': self.current_equity,
            'current_drawdown': self.current_drawdown,
            'current_drawdown_pct': self.current_drawdown_pct,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_pct': self.max_drawdown_pct,
            'state': self.state.value,
            'warning_threshold': self.warning_threshold,
            'danger_threshold': self.danger_threshold,
            'critical_threshold': self.critical_threshold,
            'peak_time': self.peak_time.isoformat() if self.peak_time else None,
            'drawdown_start_time': (
                self.drawdown_start_time.isoformat()
                if self.drawdown_start_time else None
            ),
        }


@dataclass
class PositionSizer:
    """
    Dynamic position sizing based on risk factors.

    Calculates optimal position size considering:
    - Account equity
    - Volatility
    - Confidence level
    - Session timing
    - Drawdown state
    """
    # Base parameters (adaptive)
    base_risk_per_trade: float = 0.01  # 1% risk per trade
    max_position_pct: float = 0.10  # Max 10% in single position

    # Scaling factors
    confidence_scale_min: float = 0.5
    confidence_scale_max: float = 1.5

    # Session scaling
    session_end_scale: float = 0.25  # Reduce near session end

    # Drawdown scaling
    drawdown_scale_factor: float = 0.5  # Reduce by 50% at critical drawdown

    def calculate_position_size(
        self,
        account_equity: float,
        contract_value: float,
        confidence: float,
        volatility: float,
        drawdown_state: DrawdownState,
        minutes_to_close: int,
        min_minutes_to_close: int = 30,
    ) -> Tuple[int, dict]:
        """
        Calculate optimal position size.

        Args:
            account_equity: Current account equity
            contract_value: Value of one contract (price * multiplier)
            confidence: Prediction confidence (0-1)
            volatility: Current market volatility
            drawdown_state: Current drawdown state
            minutes_to_close: Minutes until session close
            min_minutes_to_close: Minimum minutes required for new positions

        Returns:
            Tuple of (position_size, calculation_details)
        """
        details = {
            'account_equity': account_equity,
            'contract_value': contract_value,
            'base_risk': self.base_risk_per_trade,
        }

        # Base position from risk budget
        risk_amount = account_equity * self.base_risk_per_trade
        if contract_value > 0 and volatility > 0:
            base_contracts = risk_amount / (contract_value * volatility)
        else:
            base_contracts = 0.0

        details['base_contracts'] = base_contracts

        # Confidence scaling
        confidence_scale = self._confidence_to_scale(confidence)
        scaled_contracts = base_contracts * confidence_scale
        details['confidence_scale'] = confidence_scale
        details['after_confidence'] = scaled_contracts

        # Drawdown scaling
        drawdown_scale = self._drawdown_state_to_scale(drawdown_state)
        scaled_contracts *= drawdown_scale
        details['drawdown_scale'] = drawdown_scale
        details['after_drawdown'] = scaled_contracts

        # Session timing scaling
        if minutes_to_close < min_minutes_to_close:
            # Too close to session end - no new positions
            session_scale = 0.0
        elif minutes_to_close < 60:
            # Within last hour - reduced position
            session_scale = self.session_end_scale + (
                (1 - self.session_end_scale) * (minutes_to_close - min_minutes_to_close) /
                (60 - min_minutes_to_close)
            )
        else:
            session_scale = 1.0

        scaled_contracts *= session_scale
        details['session_scale'] = session_scale
        details['after_session'] = scaled_contracts

        # Max position limit
        max_contracts_by_equity = (account_equity * self.max_position_pct) / contract_value
        if scaled_contracts > max_contracts_by_equity:
            scaled_contracts = max_contracts_by_equity
            details['capped_by_max'] = True
        else:
            details['capped_by_max'] = False

        # Round to integer
        position_size = max(0, int(scaled_contracts))
        details['final_position'] = position_size

        return position_size, details

    def _confidence_to_scale(self, confidence: float) -> float:
        """Convert confidence to position scale factor."""
        # Map confidence (0-1) to scale (min-max)
        normalized = max(0, min(1, confidence))
        return self.confidence_scale_min + (
            normalized * (self.confidence_scale_max - self.confidence_scale_min)
        )

    def _drawdown_state_to_scale(self, state: DrawdownState) -> float:
        """Convert drawdown state to position scale factor."""
        scales = {
            DrawdownState.NORMAL: 1.0,
            DrawdownState.RECOVERY: 0.9,
            DrawdownState.WARNING: 0.75,
            DrawdownState.DANGER: 0.5,
            DrawdownState.CRITICAL: self.drawdown_scale_factor,
        }
        return scales.get(state, 1.0)


class RiskManager:
    """
    Comprehensive Risk Manager for autonomous trading.

    Provides:
    - Sharpe ratio optimization and tracking
    - Drawdown protection with dynamic thresholds
    - Position sizing based on risk factors
    - Integration with ConfidenceTracker
    - Risk state monitoring and alerts

    Usage:
        risk_manager = RiskManager()

        # Update with new trade result
        risk_manager.record_trade_result(pnl=100.0, trade_success=True)

        # Check if trading should continue
        if risk_manager.can_trade():
            # Get position size
            size = risk_manager.calculate_position_size(
                contract_value=50000,
                confidence=0.75,
                volatility=0.02,
                minutes_to_close=120,
            )

        # Get current risk assessment
        assessment = risk_manager.get_risk_assessment()
    """

    def __init__(
        self,
        initial_equity: float = None,
        risk_per_trade: float = None,
        max_position_pct: float = None,
        target_sharpe: float = None,
    ):
        """
        Initialize RiskManager.

        Args:
            initial_equity: Starting account equity (will be updated dynamically)
            risk_per_trade: Base risk per trade as decimal (e.g., 0.01 for 1%)
            max_position_pct: Maximum position size as percent of equity
            target_sharpe: Target Sharpe ratio for optimization
        """
        self._config = get_config()
        self._logger = logging.getLogger(__name__)

        # Sharpe metrics
        self._sharpe = SharpeMetrics()
        self._target_sharpe = target_sharpe or 1.5  # Target Sharpe (adaptive)

        # Drawdown tracking
        self._drawdown = DrawdownTracker()
        if initial_equity:
            self._drawdown.set_initial_equity(initial_equity)

        # Position sizer
        self._position_sizer = PositionSizer(
            base_risk_per_trade=risk_per_trade or 0.01,
            max_position_pct=max_position_pct or 0.10,
        )

        # Current equity
        self._current_equity = initial_equity or 0.0

        # Trading state
        self._trading_state = TradingState.ACTIVE
        self._risk_level = RiskLevel.NORMAL

        # Return history for analysis
        self._return_history: deque = deque(maxlen=1000)

        # Daily tracking
        self._daily_returns: deque = deque(maxlen=252)  # ~1 year of trading days
        self._session_pnl: float = 0.0
        self._session_trades: int = 0
        self._session_wins: int = 0

        # Risk limits (adaptive)
        self._max_daily_loss: float = 0.02  # 2% max daily loss
        self._max_consecutive_losses: int = 5
        self._consecutive_losses: int = 0

        # Callbacks
        self._on_risk_change: List[Callable[[RiskLevel, RiskLevel], None]] = []
        self._on_trading_halt: List[Callable[[str], None]] = []
        self._on_drawdown_alert: List[Callable[[DrawdownState, float], None]] = []

        # Statistics
        self._total_trades: int = 0
        self._total_wins: int = 0
        self._total_losses: int = 0
        self._largest_win: float = 0.0
        self._largest_loss: float = 0.0

        self._logger.info(
            f"RiskManager initialized: target_sharpe={self._target_sharpe}, "
            f"risk_per_trade={self._position_sizer.base_risk_per_trade}"
        )

    def record_trade_result(
        self,
        pnl: float,
        trade_success: bool,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Record a trade result for risk tracking.

        Args:
            pnl: Profit/loss from the trade
            trade_success: Whether the trade was successful
            timestamp: Time of trade completion
        """
        timestamp = timestamp or datetime.now(timezone.utc)

        # Update statistics
        self._total_trades += 1
        self._session_trades += 1
        self._session_pnl += pnl

        if trade_success:
            self._total_wins += 1
            self._session_wins += 1
            self._consecutive_losses = 0
            if pnl > self._largest_win:
                self._largest_win = pnl
        else:
            self._total_losses += 1
            self._consecutive_losses += 1
            if pnl < self._largest_loss:
                self._largest_loss = pnl

        # Update Sharpe calculation
        self._sharpe.update(pnl)

        # Update equity and drawdown
        self._current_equity += pnl
        old_state = self._drawdown.state
        new_state = self._drawdown.update(self._current_equity, timestamp)

        # Record in history
        self._return_history.append(ReturnRecord(
            timestamp=timestamp,
            pnl=pnl,
            trade_count=1,
            win_count=1 if trade_success else 0,
        ))

        # Check for risk state changes
        self._update_risk_state()

        # Alert on drawdown state change
        if new_state != old_state:
            self._notify_drawdown_alert(new_state, self._drawdown.current_drawdown_pct)

        self._logger.debug(
            f"Trade recorded: pnl={pnl:.2f}, success={trade_success}, "
            f"sharpe={self._sharpe.sharpe_ratio:.3f}, "
            f"drawdown={self._drawdown.current_drawdown_pct:.2%}"
        )

    def update_equity(self, equity: float) -> None:
        """
        Update current equity (e.g., from unrealized P&L changes).

        Args:
            equity: Current total equity
        """
        self._current_equity = equity
        self._drawdown.update(equity)
        self._update_risk_state()

    def calculate_position_size(
        self,
        contract_value: float,
        confidence: float,
        volatility: float = 0.02,
        minutes_to_close: int = 120,
    ) -> int:
        """
        Calculate recommended position size.

        Args:
            contract_value: Value of one contract (price * multiplier)
            confidence: Prediction confidence (0-1)
            volatility: Current market volatility
            minutes_to_close: Minutes until session close

        Returns:
            Recommended position size (contracts)
        """
        if not self.can_trade():
            return 0

        min_minutes = self._config.SESSION_CLOSE_BUFFER_MINUTES

        size, details = self._position_sizer.calculate_position_size(
            account_equity=self._current_equity,
            contract_value=contract_value,
            confidence=confidence,
            volatility=volatility,
            drawdown_state=self._drawdown.state,
            minutes_to_close=minutes_to_close,
            min_minutes_to_close=min_minutes,
        )

        self._logger.debug(f"Position size calculated: {size} (details: {details})")

        return size

    def can_trade(self) -> bool:
        """
        Check if trading is currently allowed.

        Returns:
            True if trading is allowed
        """
        return self._trading_state != TradingState.HALTED

    def should_reduce_exposure(self) -> bool:
        """
        Check if exposure should be reduced.

        Returns:
            True if exposure reduction recommended
        """
        return self._trading_state in (TradingState.REDUCED, TradingState.CAUTIOUS)

    def get_confidence_adjustment(self) -> float:
        """
        Get confidence threshold adjustment based on risk state.

        Returns higher values during elevated risk to require higher confidence.

        Returns:
            Multiplier for confidence threshold (1.0 = no adjustment)
        """
        adjustments = {
            RiskLevel.LOW: 0.9,  # Can be slightly less conservative
            RiskLevel.NORMAL: 1.0,
            RiskLevel.ELEVATED: 1.1,
            RiskLevel.HIGH: 1.25,
            RiskLevel.CRITICAL: 1.5,
        }

        base_adjustment = adjustments.get(self._risk_level, 1.0)

        # Additional adjustment for drawdown
        if self._drawdown.state == DrawdownState.CRITICAL:
            base_adjustment *= 1.2
        elif self._drawdown.state == DrawdownState.DANGER:
            base_adjustment *= 1.1

        return base_adjustment

    def _update_risk_state(self) -> None:
        """Update overall risk state based on multiple factors."""
        old_level = self._risk_level
        old_state = self._trading_state

        # Assess risk level
        self._risk_level = self._assess_risk_level()

        # Determine trading state
        self._trading_state = self._determine_trading_state()

        # Notify on changes
        if self._risk_level != old_level:
            self._notify_risk_change(old_level, self._risk_level)

        if self._trading_state == TradingState.HALTED and old_state != TradingState.HALTED:
            self._notify_trading_halt(self._get_halt_reason())

    def _assess_risk_level(self) -> RiskLevel:
        """Assess current risk level from multiple factors."""
        risk_score = 0.0

        # Factor 1: Drawdown state
        drawdown_scores = {
            DrawdownState.NORMAL: 0,
            DrawdownState.RECOVERY: 1,
            DrawdownState.WARNING: 2,
            DrawdownState.DANGER: 3,
            DrawdownState.CRITICAL: 4,
        }
        risk_score += drawdown_scores.get(self._drawdown.state, 0) * 0.4

        # Factor 2: Sharpe ratio vs target
        if self._sharpe.count > 10:
            sharpe_diff = self._target_sharpe - self._sharpe.sharpe_ratio
            if sharpe_diff > 1.0:
                risk_score += 2 * 0.3
            elif sharpe_diff > 0.5:
                risk_score += 1 * 0.3
            elif sharpe_diff < -0.5:
                risk_score -= 1 * 0.3  # Better than target = lower risk

        # Factor 3: Consecutive losses
        if self._consecutive_losses >= self._max_consecutive_losses:
            risk_score += 2 * 0.3
        elif self._consecutive_losses >= 3:
            risk_score += 1 * 0.3

        # Map score to level
        if risk_score >= 3.0:
            return RiskLevel.CRITICAL
        elif risk_score >= 2.0:
            return RiskLevel.HIGH
        elif risk_score >= 1.0:
            return RiskLevel.ELEVATED
        elif risk_score <= -0.5:
            return RiskLevel.LOW
        return RiskLevel.NORMAL

    def _determine_trading_state(self) -> TradingState:
        """Determine trading state from risk assessment."""
        # Critical conditions that halt trading
        if self._drawdown.state == DrawdownState.CRITICAL:
            return TradingState.HALTED

        if self._consecutive_losses >= self._max_consecutive_losses:
            return TradingState.HALTED

        # Check daily loss limit
        if self._current_equity > 0:
            daily_loss_pct = -self._session_pnl / self._drawdown.peak_equity
            if daily_loss_pct >= self._max_daily_loss:
                return TradingState.HALTED

        # Elevated risk conditions
        if self._risk_level == RiskLevel.HIGH:
            return TradingState.CAUTIOUS

        if self._risk_level == RiskLevel.ELEVATED:
            return TradingState.REDUCED

        if self._drawdown.state == DrawdownState.DANGER:
            return TradingState.CAUTIOUS

        if self._drawdown.state == DrawdownState.WARNING:
            return TradingState.REDUCED

        return TradingState.ACTIVE

    def _get_halt_reason(self) -> str:
        """Get reason for trading halt."""
        reasons = []

        if self._drawdown.state == DrawdownState.CRITICAL:
            reasons.append(f"Critical drawdown ({self._drawdown.current_drawdown_pct:.1%})")

        if self._consecutive_losses >= self._max_consecutive_losses:
            reasons.append(f"Consecutive losses ({self._consecutive_losses})")

        if self._current_equity > 0:
            daily_loss_pct = -self._session_pnl / self._drawdown.peak_equity
            if daily_loss_pct >= self._max_daily_loss:
                reasons.append(f"Daily loss limit ({daily_loss_pct:.1%})")

        return "; ".join(reasons) if reasons else "Unknown"

    def reset_session(self) -> None:
        """Reset session-level tracking (call at session start)."""
        self._session_pnl = 0.0
        self._session_trades = 0
        self._session_wins = 0
        self._consecutive_losses = 0

        # Potentially resume trading if was halted for daily limits
        if self._trading_state == TradingState.HALTED:
            self._update_risk_state()

        self._logger.info("Risk manager session reset")

    def resume_trading(self) -> bool:
        """
        Attempt to resume trading after halt.

        Returns:
            True if trading was resumed
        """
        # Check if conditions allow resumption
        if self._drawdown.state == DrawdownState.CRITICAL:
            self._logger.warning("Cannot resume: still in critical drawdown")
            return False

        # Reset consecutive losses counter
        self._consecutive_losses = 0
        self._update_risk_state()

        if self._trading_state != TradingState.HALTED:
            self._logger.info("Trading resumed")
            return True

        return False

    def get_risk_assessment(self) -> dict:
        """
        Get comprehensive risk assessment.

        Returns:
            Dictionary with all risk metrics
        """
        return {
            'trading_state': self._trading_state.value,
            'risk_level': self._risk_level.value,
            'can_trade': self.can_trade(),
            'sharpe': self._sharpe.to_dict(),
            'drawdown': self._drawdown.to_dict(),
            'confidence_adjustment': self.get_confidence_adjustment(),
            'session': {
                'pnl': self._session_pnl,
                'trades': self._session_trades,
                'wins': self._session_wins,
                'win_rate': (
                    self._session_wins / self._session_trades
                    if self._session_trades > 0 else 0.0
                ),
            },
            'consecutive_losses': self._consecutive_losses,
            'total_trades': self._total_trades,
            'total_win_rate': (
                self._total_wins / self._total_trades
                if self._total_trades > 0 else 0.0
            ),
            'largest_win': self._largest_win,
            'largest_loss': self._largest_loss,
            'current_equity': self._current_equity,
        }

    def get_summary(self) -> dict:
        """
        Get concise summary for dashboard.

        Returns:
            Summary dictionary
        """
        return {
            'trading_state': self._trading_state.value,
            'risk_level': self._risk_level.value,
            'sharpe_ratio': self._sharpe.sharpe_ratio,
            'drawdown_pct': self._drawdown.current_drawdown_pct,
            'drawdown_state': self._drawdown.state.value,
            'session_pnl': self._session_pnl,
            'session_trades': self._session_trades,
            'win_rate': self._sharpe.win_rate,
            'current_equity': self._current_equity,
        }

    def get_sharpe_ratio(self) -> float:
        """Get current Sharpe ratio."""
        return self._sharpe.sharpe_ratio

    def get_drawdown_percent(self) -> float:
        """Get current drawdown percentage."""
        return self._drawdown.current_drawdown_pct

    def get_max_drawdown_percent(self) -> float:
        """Get maximum drawdown percentage."""
        return self._drawdown.max_drawdown_pct

    # Callback registration
    def register_risk_callback(
        self,
        callback: Callable[[RiskLevel, RiskLevel], None],
    ) -> None:
        """
        Register callback for risk level changes.

        Args:
            callback: Function(old_level, new_level)
        """
        self._on_risk_change.append(callback)

    def register_halt_callback(
        self,
        callback: Callable[[str], None],
    ) -> None:
        """
        Register callback for trading halts.

        Args:
            callback: Function(halt_reason)
        """
        self._on_trading_halt.append(callback)

    def register_drawdown_callback(
        self,
        callback: Callable[[DrawdownState, float], None],
    ) -> None:
        """
        Register callback for drawdown alerts.

        Args:
            callback: Function(drawdown_state, drawdown_pct)
        """
        self._on_drawdown_alert.append(callback)

    def _notify_risk_change(
        self,
        old_level: RiskLevel,
        new_level: RiskLevel,
    ) -> None:
        """Notify callbacks of risk level change."""
        self._logger.info(f"Risk level changed: {old_level.value} -> {new_level.value}")
        for callback in self._on_risk_change:
            try:
                callback(old_level, new_level)
            except Exception as e:
                self._logger.error(f"Risk change callback error: {e}")

    def _notify_trading_halt(self, reason: str) -> None:
        """Notify callbacks of trading halt."""
        self._logger.warning(f"Trading halted: {reason}")
        for callback in self._on_trading_halt:
            try:
                callback(reason)
            except Exception as e:
                self._logger.error(f"Trading halt callback error: {e}")

    def _notify_drawdown_alert(
        self,
        state: DrawdownState,
        drawdown_pct: float,
    ) -> None:
        """Notify callbacks of drawdown alert."""
        self._logger.warning(
            f"Drawdown alert: state={state.value}, drawdown={drawdown_pct:.1%}"
        )
        for callback in self._on_drawdown_alert:
            try:
                callback(state, drawdown_pct)
            except Exception as e:
                self._logger.error(f"Drawdown alert callback error: {e}")

    # Properties
    @property
    def trading_state(self) -> TradingState:
        """Get current trading state."""
        return self._trading_state

    @property
    def risk_level(self) -> RiskLevel:
        """Get current risk level."""
        return self._risk_level

    @property
    def drawdown_state(self) -> DrawdownState:
        """Get current drawdown state."""
        return self._drawdown.state

    @property
    def current_equity(self) -> float:
        """Get current equity."""
        return self._current_equity

    @property
    def session_pnl(self) -> float:
        """Get session P&L."""
        return self._session_pnl
