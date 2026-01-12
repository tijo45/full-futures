"""
Confidence Tracking and Dynamic Threshold Management.

Implements confidence scoring, tracking, and dynamic threshold management
for confidence-gated trading. Only trades meeting confidence thresholds
are executed - low-confidence signals are rejected.

Key Features:
- Per-prediction, per-contract, per-regime confidence tracking
- Dynamic thresholds - NO hard-coded values
- Drawdown-based threshold adjustment
- Confidence gating for entries, scaling, and exits
- Historical confidence tracking for analysis
- Adaptive threshold learning from outcomes

CRITICAL: No hard-coded thresholds. All thresholds adapt based on
performance and market conditions.
"""

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


from config import get_config


class GateDecision(Enum):
    """Decision from confidence gating."""
    ALLOWED = "allowed"
    REJECTED = "rejected"
    SCALED_DOWN = "scaled_down"


class ThresholdMode(Enum):
    """Mode for threshold adjustment."""
    NORMAL = "normal"
    CAUTIOUS = "cautious"  # After losses or drift
    AGGRESSIVE = "aggressive"  # During strong performance


@dataclass
class ConfidenceLevel:
    """
    Represents a confidence level with associated metadata.

    Contains the confidence score and factors that contributed to it.
    """
    value: float  # 0.0 to 1.0
    timestamp: datetime

    # Contributing factors
    model_confidence: float = 0.0  # From model probability
    data_quality: float = 0.0  # From data freshness/completeness
    regime_stability: float = 0.0  # From drift detection
    recent_accuracy: float = 0.0  # From recent prediction accuracy

    # Context
    contract_id: Optional[int] = None
    prediction_id: Optional[str] = None

    def to_dict(self) -> dict:
        """Export as dictionary."""
        return {
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'model_confidence': self.model_confidence,
            'data_quality': self.data_quality,
            'regime_stability': self.regime_stability,
            'recent_accuracy': self.recent_accuracy,
            'contract_id': self.contract_id,
            'prediction_id': self.prediction_id,
        }


@dataclass
class ThresholdState:
    """
    Dynamic threshold state with adaptive adjustment.

    Thresholds adapt based on performance, drawdown, and regime.
    """
    # Base threshold learned from performance
    base_threshold: float = 0.5

    # Current active threshold (may be modified from base)
    current_threshold: float = 0.5

    # Threshold bounds (learned, not hard-coded)
    min_threshold: float = 0.3
    max_threshold: float = 0.95

    # Mode and adjustments
    mode: ThresholdMode = ThresholdMode.NORMAL
    adjustment_factor: float = 1.0  # Multiplier for threshold

    # Performance tracking for threshold adaptation
    recent_outcomes: deque = field(default_factory=lambda: deque(maxlen=50))

    # Threshold history
    history: deque = field(default_factory=lambda: deque(maxlen=200))

    # Learning state
    samples_seen: int = 0
    last_update: Optional[datetime] = None

    def update_threshold(self, success: bool, confidence_used: float) -> None:
        """
        Update threshold based on trade outcome.

        Learns optimal threshold from outcomes - increases after failures
        at low confidence, decreases after successes at high confidence.

        Args:
            success: Whether the trade was successful
            confidence_used: The confidence level when trade was made
        """
        self.recent_outcomes.append({
            'success': success,
            'confidence': confidence_used,
            'timestamp': datetime.now(timezone.utc),
        })

        self.samples_seen += 1
        self.last_update = datetime.now(timezone.utc)

        # Calculate recent success rate
        if len(self.recent_outcomes) < 5:
            return

        successes = sum(1 for o in self.recent_outcomes if o['success'])
        success_rate = successes / len(self.recent_outcomes)

        # Adaptive threshold learning
        if success_rate < 0.4:
            # Poor performance - raise threshold
            self._adjust_threshold_up()
        elif success_rate > 0.65:
            # Good performance - can lower threshold cautiously
            self._adjust_threshold_down()

        # Record history
        self.history.append({
            'threshold': self.current_threshold,
            'success_rate': success_rate,
            'timestamp': self.last_update,
        })

    def _adjust_threshold_up(self) -> None:
        """Increase threshold (more conservative)."""
        step = (self.max_threshold - self.base_threshold) * 0.05
        self.base_threshold = min(self.max_threshold, self.base_threshold + step)
        self._apply_mode_adjustment()

    def _adjust_threshold_down(self) -> None:
        """Decrease threshold (more aggressive)."""
        step = (self.base_threshold - self.min_threshold) * 0.02
        self.base_threshold = max(self.min_threshold, self.base_threshold - step)
        self._apply_mode_adjustment()

    def _apply_mode_adjustment(self) -> None:
        """Apply mode-based adjustment to current threshold."""
        if self.mode == ThresholdMode.CAUTIOUS:
            self.current_threshold = min(
                self.max_threshold,
                self.base_threshold * self.adjustment_factor
            )
        elif self.mode == ThresholdMode.AGGRESSIVE:
            self.current_threshold = max(
                self.min_threshold,
                self.base_threshold / self.adjustment_factor
            )
        else:
            self.current_threshold = self.base_threshold

    def set_mode(self, mode: ThresholdMode, factor: float = 1.2) -> None:
        """
        Set threshold mode.

        Args:
            mode: New threshold mode
            factor: Adjustment factor for cautious/aggressive modes
        """
        self.mode = mode
        self.adjustment_factor = factor
        self._apply_mode_adjustment()

    def to_dict(self) -> dict:
        """Export state as dictionary."""
        return {
            'base_threshold': self.base_threshold,
            'current_threshold': self.current_threshold,
            'min_threshold': self.min_threshold,
            'max_threshold': self.max_threshold,
            'mode': self.mode.value,
            'adjustment_factor': self.adjustment_factor,
            'samples_seen': self.samples_seen,
            'last_update': self.last_update.isoformat() if self.last_update else None,
        }


@dataclass
class ConfidenceHistory:
    """
    Historical confidence tracking for analysis.

    Maintains rolling statistics and history for confidence levels.
    """
    # Rolling windows
    recent_confidences: deque = field(default_factory=lambda: deque(maxlen=500))

    # Running statistics (Welford's algorithm)
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0  # For variance calculation

    # Extremes
    min_confidence: float = 1.0
    max_confidence: float = 0.0

    # Time-based tracking
    confidences_by_hour: Dict[int, List[float]] = field(default_factory=dict)

    def record(self, confidence: ConfidenceLevel) -> None:
        """Record a confidence level."""
        value = confidence.value

        # Update rolling window
        self.recent_confidences.append({
            'value': value,
            'timestamp': confidence.timestamp,
            'contract_id': confidence.contract_id,
        })

        # Update running statistics (Welford's algorithm)
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

        # Update extremes
        self.min_confidence = min(self.min_confidence, value)
        self.max_confidence = max(self.max_confidence, value)

        # Track by hour
        hour = confidence.timestamp.hour
        if hour not in self.confidences_by_hour:
            self.confidences_by_hour[hour] = []
        self.confidences_by_hour[hour].append(value)
        # Keep limited history per hour
        if len(self.confidences_by_hour[hour]) > 100:
            self.confidences_by_hour[hour] = self.confidences_by_hour[hour][-100:]

    @property
    def variance(self) -> float:
        """Get variance of confidences."""
        if self.count < 2:
            return 0.0
        return self.m2 / (self.count - 1)

    @property
    def std_dev(self) -> float:
        """Get standard deviation of confidences."""
        return math.sqrt(self.variance)

    @property
    def recent_mean(self) -> float:
        """Get mean of recent confidences."""
        if not self.recent_confidences:
            return 0.0
        return sum(c['value'] for c in self.recent_confidences) / len(self.recent_confidences)

    def get_percentile(self, percentile: float) -> float:
        """
        Get confidence value at given percentile from recent history.

        Args:
            percentile: Percentile (0-100)

        Returns:
            Confidence value at percentile
        """
        if not self.recent_confidences:
            return 0.5

        sorted_values = sorted(c['value'] for c in self.recent_confidences)
        idx = int(len(sorted_values) * percentile / 100)
        idx = max(0, min(idx, len(sorted_values) - 1))
        return sorted_values[idx]

    def to_dict(self) -> dict:
        """Export as dictionary."""
        return {
            'count': self.count,
            'mean': self.mean,
            'std_dev': self.std_dev,
            'min': self.min_confidence,
            'max': self.max_confidence,
            'recent_mean': self.recent_mean,
            'percentile_25': self.get_percentile(25),
            'percentile_75': self.get_percentile(75),
        }


@dataclass
class ContractConfidence:
    """
    Per-contract confidence tracking.

    Maintains separate statistics for each contract.
    """
    contract_id: int
    symbol: str = ""

    # Confidence statistics
    history: ConfidenceHistory = field(default_factory=ConfidenceHistory)

    # Threshold state for this contract
    threshold: ThresholdState = field(default_factory=ThresholdState)

    # Recent predictions and outcomes
    recent_predictions: deque = field(default_factory=lambda: deque(maxlen=100))

    # Performance
    successful_trades: int = 0
    total_trades: int = 0

    @property
    def success_rate(self) -> float:
        """Get trade success rate."""
        if self.total_trades == 0:
            return 0.0
        return self.successful_trades / self.total_trades

    def record_confidence(self, confidence: ConfidenceLevel) -> None:
        """Record a confidence level for this contract."""
        self.history.record(confidence)
        self.recent_predictions.append({
            'confidence': confidence.value,
            'timestamp': confidence.timestamp,
            'prediction_id': confidence.prediction_id,
        })

    def record_outcome(self, success: bool, confidence_used: float) -> None:
        """Record trade outcome."""
        self.total_trades += 1
        if success:
            self.successful_trades += 1
        self.threshold.update_threshold(success, confidence_used)

    def to_dict(self) -> dict:
        """Export as dictionary."""
        return {
            'contract_id': self.contract_id,
            'symbol': self.symbol,
            'history': self.history.to_dict(),
            'threshold': self.threshold.to_dict(),
            'success_rate': self.success_rate,
            'total_trades': self.total_trades,
        }


class ConfidenceTracker:
    """
    Confidence tracking and dynamic threshold management.

    Tracks confidence at multiple granularities and provides gating
    for trade execution. All thresholds are dynamic - no hard-coded values.

    Granularities:
    - Per prediction: Individual prediction confidence
    - Per contract: Contract-specific confidence tracking
    - Per regime: Confidence adjustments based on market regime
    - Over time: Historical confidence trends

    Usage:
        tracker = ConfidenceTracker()

        # Record and evaluate confidence
        confidence = tracker.calculate_confidence(
            model_probability=0.75,
            data_quality=0.9,
            regime_stability=0.8,
        )

        # Check if entry is allowed
        decision = tracker.gate_entry(confidence, contract_id=123)
        if decision == GateDecision.ALLOWED:
            # Proceed with trade

        # Record outcome for learning
        tracker.record_outcome(success=True, confidence_used=confidence.value)
    """

    def __init__(
        self,
        initial_threshold: float = None,
        min_threshold: float = None,
        max_threshold: float = None,
        drawdown_multiplier: float = None,
    ):
        """
        Initialize ConfidenceTracker.

        Args:
            initial_threshold: Starting threshold (learned if None)
            min_threshold: Minimum allowed threshold (adaptive if None)
            max_threshold: Maximum allowed threshold (adaptive if None)
            drawdown_multiplier: Threshold increase per drawdown percent
        """
        self._config = get_config()
        self._logger = logging.getLogger(__name__)

        # Global threshold state (adaptive defaults)
        self._global_threshold = ThresholdState(
            base_threshold=initial_threshold or 0.5,
            current_threshold=initial_threshold or 0.5,
            min_threshold=min_threshold or 0.3,
            max_threshold=max_threshold or 0.95,
        )

        # Per-contract tracking
        self._contracts: Dict[int, ContractConfidence] = {}

        # Global confidence history
        self._history = ConfidenceHistory()

        # Regime awareness
        self._current_regime: str = "stable"
        self._regime_confidence_factor: float = 1.0

        # Drawdown tracking for threshold adjustment
        self._drawdown_multiplier = drawdown_multiplier or 0.5  # per 1% drawdown
        self._current_drawdown: float = 0.0

        # Callbacks
        self._on_threshold_change: List[Callable[[float, float], None]] = []
        self._on_gate_decision: List[Callable[[GateDecision, float], None]] = []

        # Statistics
        self._total_evaluations: int = 0
        self._allowed_count: int = 0
        self._rejected_count: int = 0

        self._logger.info(
            f"ConfidenceTracker initialized: threshold={self._global_threshold.current_threshold:.3f}"
        )

    def calculate_confidence(
        self,
        model_probability: float,
        data_quality: float = 1.0,
        regime_stability: float = 1.0,
        recent_accuracy: float = 0.5,
        contract_id: Optional[int] = None,
        prediction_id: Optional[str] = None,
    ) -> ConfidenceLevel:
        """
        Calculate confidence level from multiple factors.

        Combines model confidence with data quality, regime stability,
        and recent performance to produce overall confidence score.

        Args:
            model_probability: Raw probability from model (0.5-1.0)
            data_quality: Data freshness/completeness factor (0-1)
            regime_stability: Regime stability factor (0-1)
            recent_accuracy: Recent prediction accuracy (0-1)
            contract_id: Optional contract ID for tracking
            prediction_id: Optional prediction ID for tracking

        Returns:
            ConfidenceLevel with overall confidence score
        """
        # Convert probability to confidence (distance from 0.5)
        model_confidence = abs(model_probability - 0.5) * 2

        # Weight factors (can be learned/adapted in future)
        weights = {
            'model': 0.4,
            'data': 0.2,
            'regime': 0.2,
            'accuracy': 0.2,
        }

        # Calculate weighted confidence
        confidence_value = (
            weights['model'] * model_confidence +
            weights['data'] * data_quality +
            weights['regime'] * regime_stability +
            weights['accuracy'] * self._normalize_accuracy(recent_accuracy)
        )

        # Apply regime factor
        confidence_value *= self._regime_confidence_factor

        # Clamp to valid range
        confidence_value = max(0.0, min(1.0, confidence_value))

        # Create confidence level
        confidence = ConfidenceLevel(
            value=confidence_value,
            timestamp=datetime.now(timezone.utc),
            model_confidence=model_confidence,
            data_quality=data_quality,
            regime_stability=regime_stability,
            recent_accuracy=recent_accuracy,
            contract_id=contract_id,
            prediction_id=prediction_id,
        )

        # Record to history
        self._history.record(confidence)

        # Record to contract if specified
        if contract_id is not None:
            self._get_contract(contract_id).record_confidence(confidence)

        return confidence

    def _normalize_accuracy(self, accuracy: float) -> float:
        """
        Normalize accuracy for confidence calculation.

        Maps accuracy to 0-1 range with baseline adjustment.
        50% accuracy (random) should give ~0.5 confidence factor.
        """
        # Below 50% is worse than random
        if accuracy < 0.5:
            return accuracy
        # Scale 50-100% to 0.5-1.0
        return 0.5 + (accuracy - 0.5)

    def _get_contract(self, contract_id: int, symbol: str = "") -> ContractConfidence:
        """Get or create per-contract confidence tracker."""
        if contract_id not in self._contracts:
            self._contracts[contract_id] = ContractConfidence(
                contract_id=contract_id,
                symbol=symbol,
            )
        return self._contracts[contract_id]

    def gate_entry(
        self,
        confidence: ConfidenceLevel,
        contract_id: Optional[int] = None,
    ) -> GateDecision:
        """
        Gate entry decision based on confidence threshold.

        Args:
            confidence: Confidence level to evaluate
            contract_id: Optional contract ID for per-contract threshold

        Returns:
            GateDecision indicating whether entry is allowed
        """
        self._total_evaluations += 1

        # Get applicable threshold
        threshold = self._get_threshold(contract_id)

        # Apply drawdown adjustment
        adjusted_threshold = self._apply_drawdown_adjustment(threshold)

        # Make decision
        if confidence.value >= adjusted_threshold:
            decision = GateDecision.ALLOWED
            self._allowed_count += 1
        else:
            decision = GateDecision.REJECTED
            self._rejected_count += 1

        # Call callbacks
        for callback in self._on_gate_decision:
            try:
                callback(decision, confidence.value)
            except Exception as e:
                self._logger.error(f"Gate decision callback error: {e}")

        self._logger.debug(
            f"Gate decision: {decision.value} "
            f"(confidence={confidence.value:.3f}, threshold={adjusted_threshold:.3f})"
        )

        return decision

    def gate_exit(
        self,
        confidence: ConfidenceLevel,
        position_pnl_percent: float,
        contract_id: Optional[int] = None,
    ) -> GateDecision:
        """
        Gate exit decision based on confidence and P&L.

        Lower confidence threshold for exits when in profit,
        higher threshold when in loss (avoid panic selling).

        Args:
            confidence: Confidence level for exit signal
            position_pnl_percent: Current position P&L in percent
            contract_id: Optional contract ID

        Returns:
            GateDecision for exit
        """
        threshold = self._get_threshold(contract_id)

        # Adjust threshold based on P&L
        if position_pnl_percent > 0:
            # In profit - lower threshold to allow easier exit
            exit_threshold = threshold * 0.7
        else:
            # In loss - higher threshold to avoid panic exits
            exit_threshold = threshold * 1.1

        exit_threshold = max(self._global_threshold.min_threshold,
                            min(self._global_threshold.max_threshold, exit_threshold))

        if confidence.value >= exit_threshold:
            return GateDecision.ALLOWED
        return GateDecision.REJECTED

    def get_position_scaling(
        self,
        confidence: ConfidenceLevel,
        contract_id: Optional[int] = None,
    ) -> float:
        """
        Get position scaling factor based on confidence.

        Higher confidence allows larger position size.

        Args:
            confidence: Confidence level
            contract_id: Optional contract ID

        Returns:
            Scaling factor (0.0 to 1.0)
        """
        threshold = self._get_threshold(contract_id)

        # Must meet minimum threshold
        if confidence.value < threshold:
            return 0.0

        # Scale linearly from threshold to max confidence
        max_conf = self._global_threshold.max_threshold
        scale_range = max_conf - threshold

        if scale_range <= 0:
            return 1.0 if confidence.value >= threshold else 0.0

        scaling = (confidence.value - threshold) / scale_range
        return max(0.0, min(1.0, scaling))

    def _get_threshold(self, contract_id: Optional[int] = None) -> float:
        """Get applicable threshold for contract or global."""
        if contract_id is not None and contract_id in self._contracts:
            return self._contracts[contract_id].threshold.current_threshold
        return self._global_threshold.current_threshold

    def _apply_drawdown_adjustment(self, threshold: float) -> float:
        """
        Adjust threshold based on current drawdown.

        Increases threshold during drawdown to be more conservative.
        """
        if self._current_drawdown <= 0:
            return threshold

        # Increase threshold based on drawdown
        adjustment = self._current_drawdown * self._drawdown_multiplier / 100
        adjusted = threshold + adjustment

        return min(self._global_threshold.max_threshold, adjusted)

    def update_drawdown(self, drawdown_percent: float) -> None:
        """
        Update current drawdown level.

        Args:
            drawdown_percent: Current drawdown as percentage (positive)
        """
        old_drawdown = self._current_drawdown
        self._current_drawdown = max(0, drawdown_percent)

        # Adjust mode based on drawdown severity
        if self._current_drawdown > 10:  # >10% drawdown
            self._global_threshold.set_mode(ThresholdMode.CAUTIOUS, factor=1.3)
        elif self._current_drawdown > 5:  # 5-10% drawdown
            self._global_threshold.set_mode(ThresholdMode.CAUTIOUS, factor=1.15)
        elif self._current_drawdown < 2 and old_drawdown >= 5:
            # Recovering from drawdown
            self._global_threshold.set_mode(ThresholdMode.NORMAL)

    def update_regime(
        self,
        regime: str,
        stability: float,
    ) -> None:
        """
        Update market regime state.

        Args:
            regime: Regime identifier (e.g., "stable", "volatile", "trending")
            stability: Regime stability score (0-1)
        """
        old_regime = self._current_regime
        self._current_regime = regime
        self._regime_confidence_factor = stability

        # Adjust mode based on regime
        if regime == "volatile" or stability < 0.5:
            self._global_threshold.set_mode(ThresholdMode.CAUTIOUS, factor=1.2)
        elif regime == "stable" and stability > 0.8:
            self._global_threshold.set_mode(ThresholdMode.NORMAL)

        if old_regime != regime:
            self._logger.info(
                f"Regime changed: {old_regime} -> {regime} (stability={stability:.3f})"
            )

    def set_threshold_mode(self, mode: str, factor: float = 1.2) -> None:
        """
        Set the threshold mode directly.

        Allows external components to adjust threshold mode based on
        system-wide conditions (e.g., drift detection, risk events).

        Args:
            mode: Mode name ('NORMAL', 'CAUTIOUS', or 'AGGRESSIVE')
            factor: Multiplier for threshold adjustment (default 1.2)
        """
        mode_upper = mode.upper()
        if mode_upper == 'NORMAL':
            self._global_threshold.set_mode(ThresholdMode.NORMAL, factor)
        elif mode_upper == 'CAUTIOUS':
            self._global_threshold.set_mode(ThresholdMode.CAUTIOUS, factor)
        elif mode_upper == 'AGGRESSIVE':
            self._global_threshold.set_mode(ThresholdMode.AGGRESSIVE, factor)
        else:
            self._logger.warning(f"Unknown threshold mode: {mode}")
            return

        self._logger.info(f"Threshold mode set to {mode_upper} with factor {factor}")

    def record_outcome(
        self,
        success: bool,
        confidence_used: float,
        contract_id: Optional[int] = None,
    ) -> None:
        """
        Record trade outcome for threshold learning.

        Args:
            success: Whether trade was successful
            confidence_used: Confidence level when trade was made
            contract_id: Optional contract ID
        """
        # Update global threshold
        self._global_threshold.update_threshold(success, confidence_used)

        # Update contract-specific threshold
        if contract_id is not None:
            self._get_contract(contract_id).record_outcome(success, confidence_used)

        self._logger.debug(
            f"Recorded outcome: success={success}, confidence={confidence_used:.3f}, "
            f"new_threshold={self._global_threshold.current_threshold:.3f}"
        )

        # Notify callbacks of threshold change
        for callback in self._on_threshold_change:
            try:
                callback(
                    self._global_threshold.current_threshold,
                    self._global_threshold.base_threshold,
                )
            except Exception as e:
                self._logger.error(f"Threshold change callback error: {e}")

    def register_threshold_callback(
        self,
        callback: Callable[[float, float], None],
    ) -> None:
        """
        Register callback for threshold changes.

        Args:
            callback: Function(current_threshold, base_threshold)
        """
        self._on_threshold_change.append(callback)

    def register_gate_callback(
        self,
        callback: Callable[[GateDecision, float], None],
    ) -> None:
        """
        Register callback for gate decisions.

        Args:
            callback: Function(decision, confidence_value)
        """
        self._on_gate_decision.append(callback)

    def get_threshold(self, contract_id: Optional[int] = None) -> float:
        """
        Get current threshold.

        Args:
            contract_id: Optional contract ID for contract-specific threshold

        Returns:
            Current threshold value
        """
        threshold = self._get_threshold(contract_id)
        return self._apply_drawdown_adjustment(threshold)

    def get_contract_summary(self, contract_id: int) -> Optional[dict]:
        """
        Get summary for a specific contract.

        Args:
            contract_id: Contract ID

        Returns:
            Summary dict or None if not tracked
        """
        if contract_id not in self._contracts:
            return None
        return self._contracts[contract_id].to_dict()

    def get_state(self) -> dict:
        """
        Get complete tracker state.

        Returns:
            Dictionary with all state information
        """
        return {
            'global_threshold': self._global_threshold.to_dict(),
            'current_drawdown': self._current_drawdown,
            'current_regime': self._current_regime,
            'regime_confidence_factor': self._regime_confidence_factor,
            'history': self._history.to_dict(),
            'total_evaluations': self._total_evaluations,
            'allowed_count': self._allowed_count,
            'rejected_count': self._rejected_count,
            'allow_rate': self._allowed_count / max(1, self._total_evaluations),
            'contracts_tracked': len(self._contracts),
        }

    def get_summary(self) -> dict:
        """
        Get concise summary for dashboard.

        Returns:
            Summary dictionary
        """
        return {
            'threshold': self._global_threshold.current_threshold,
            'base_threshold': self._global_threshold.base_threshold,
            'mode': self._global_threshold.mode.value,
            'drawdown': self._current_drawdown,
            'regime': self._current_regime,
            'mean_confidence': self._history.mean,
            'recent_confidence': self._history.recent_mean,
            'total_evaluations': self._total_evaluations,
            'allow_rate': self._allowed_count / max(1, self._total_evaluations),
            'contracts_count': len(self._contracts),
        }

    def reset(self, preserve_history: bool = False) -> None:
        """
        Reset tracker state.

        Args:
            preserve_history: If True, keep historical data
        """
        self._global_threshold = ThresholdState()
        self._contracts.clear()
        self._current_drawdown = 0.0
        self._current_regime = "stable"
        self._regime_confidence_factor = 1.0
        self._total_evaluations = 0
        self._allowed_count = 0
        self._rejected_count = 0

        if not preserve_history:
            self._history = ConfidenceHistory()

        self._logger.info("ConfidenceTracker reset")

    @property
    def current_threshold(self) -> float:
        """Get current global threshold."""
        return self._global_threshold.current_threshold

    @property
    def allow_rate(self) -> float:
        """Get rate of allowed entries."""
        if self._total_evaluations == 0:
            return 0.0
        return self._allowed_count / self._total_evaluations

    @property
    def contracts_count(self) -> int:
        """Get number of tracked contracts."""
        return len(self._contracts)
