"""
Drift Detection Module - Market regime change detection using ADWIN.

Implements concept drift detection using River's ADWIN algorithm for detecting
market regime changes in real-time. Markets exhibit concept drift where the
underlying data distribution changes over time, requiring adaptation of
trading models.

Key Features:
- ADWIN-based drift detection with configurable sensitivity
- Multi-stream monitoring (accuracy, returns, volatility, etc.)
- Per-contract drift tracking
- Drift severity classification
- Regime state tracking with callbacks

CRITICAL: Drift detection is essential for autonomous trading. Markets change
regimes and models must adapt. This detector signals when to increase caution,
reset models, or adjust confidence thresholds.
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from river import drift

from config import get_config


class DriftSeverity(Enum):
    """Classification of drift severity levels."""
    NONE = 0        # No drift detected
    MILD = 1        # Minor distribution shift
    MODERATE = 2    # Significant distribution change
    SEVERE = 3      # Major regime change


class RegimeState(Enum):
    """Market regime states based on drift patterns."""
    STABLE = "stable"           # No recent drift, stable market
    TRANSITIONING = "transitioning"  # Drift detected, regime changing
    VOLATILE = "volatile"       # Multiple drifts, high uncertainty
    RECOVERING = "recovering"   # After drift, stabilizing


@dataclass
class DriftEvent:
    """
    Record of a detected drift event.

    Captures metadata about when and where drift occurred for analysis.
    """
    timestamp: datetime
    stream_name: str
    severity: DriftSeverity
    window_size: int
    estimation: float
    variance: float
    detection_count: int

    def to_dict(self) -> dict:
        """Export drift event as dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'stream_name': self.stream_name,
            'severity': self.severity.name,
            'window_size': self.window_size,
            'estimation': self.estimation,
            'variance': self.variance,
            'detection_count': self.detection_count,
        }


@dataclass
class StreamState:
    """
    State tracking for a single monitored stream.

    Each stream (e.g., prediction_error, volatility) has its own ADWIN detector.
    """
    detector: drift.ADWIN = field(default_factory=lambda: drift.ADWIN())

    # Detection tracking
    detection_count: int = 0
    last_drift_time: Optional[datetime] = None
    samples_since_drift: int = 0

    # Current state
    in_drift: bool = False

    # History
    drift_history: deque = field(default_factory=lambda: deque(maxlen=100))
    value_history: deque = field(default_factory=lambda: deque(maxlen=1000))

    def update(self, value: float) -> bool:
        """
        Update stream with new value.

        Args:
            value: New data point for this stream

        Returns:
            True if drift was detected
        """
        self.detector.update(value)
        self.value_history.append(value)
        self.samples_since_drift += 1

        if self.detector.drift_detected:
            self.detection_count += 1
            self.last_drift_time = datetime.now(timezone.utc)
            self.drift_history.append(self.last_drift_time)
            self.in_drift = True
            self.samples_since_drift = 0
            return True

        # Recover from drift state after stable period
        if self.in_drift and self.samples_since_drift > 100:
            self.in_drift = False

        return False

    def reset(self) -> None:
        """Reset stream detector state."""
        self.detector = drift.ADWIN()
        self.detection_count = 0
        self.last_drift_time = None
        self.samples_since_drift = 0
        self.in_drift = False
        self.drift_history.clear()
        self.value_history.clear()

    @property
    def estimation(self) -> float:
        """Get current mean estimation from ADWIN."""
        return self.detector.estimation

    @property
    def variance(self) -> float:
        """Get current variance from ADWIN."""
        return self.detector.variance

    @property
    def width(self) -> int:
        """Get current window size."""
        return self.detector.width

    def to_dict(self) -> dict:
        """Export stream state as dictionary."""
        return {
            'detection_count': self.detection_count,
            'in_drift': self.in_drift,
            'samples_since_drift': self.samples_since_drift,
            'last_drift_time': self.last_drift_time.isoformat() if self.last_drift_time else None,
            'estimation': self.estimation,
            'variance': self.variance,
            'width': self.width,
        }


class DriftDetector:
    """
    Multi-stream drift detector for market regime change detection.

    Uses ADWIN (Adaptive Windowing) algorithm to detect concept drift across
    multiple monitored streams. Each stream can represent different market
    characteristics (prediction errors, returns, volatility, etc.).

    ADWIN maintains a variable-length window of recent data and detects
    statistically significant changes in distribution between window halves.

    Usage:
        detector = DriftDetector()

        # Update with prediction errors (most common use)
        drift_detected = detector.update_prediction_error(was_correct=False)

        # Or update specific streams
        detector.update_stream('returns', price_return)
        detector.update_stream('volatility', vol_estimate)

        # Check overall regime state
        if detector.regime_state == RegimeState.VOLATILE:
            # Increase caution, raise confidence threshold
            pass

        # Get severity for trading decisions
        if detector.overall_severity >= DriftSeverity.MODERATE:
            # Consider model reset or learning rate adjustment
            pass
    """

    # Default stream names for trading
    DEFAULT_STREAMS = [
        'prediction_error',  # Primary: tracks model accuracy drift
        'returns',           # Price returns drift
        'volatility',        # Volatility regime changes
        'spread',            # Bid-ask spread changes
        'volume',            # Volume pattern changes
    ]

    def __init__(
        self,
        delta: float = 0.002,
        clock: int = 32,
        min_window_length: int = 5,
        grace_period: int = 10,
        streams: Optional[List[str]] = None,
    ):
        """
        Initialize DriftDetector.

        Args:
            delta: Significance threshold for drift detection (lower = more sensitive)
            clock: Frequency of change detection checks
            min_window_length: Minimum subwindow size for evaluation
            grace_period: Initial samples before detection begins
            streams: List of stream names to monitor. Defaults to DEFAULT_STREAMS.
        """
        self._config = get_config()
        self._logger = logging.getLogger(__name__)

        # ADWIN configuration
        self._delta = delta
        self._clock = clock
        self._min_window_length = min_window_length
        self._grace_period = grace_period

        # Initialize streams
        stream_names = streams or self.DEFAULT_STREAMS
        self._streams: Dict[str, StreamState] = {
            name: self._create_stream_state()
            for name in stream_names
        }

        # Overall state tracking
        self._regime_state = RegimeState.STABLE
        self._last_regime_change: Optional[datetime] = None
        self._total_detections = 0

        # Event history
        self._events: deque = deque(maxlen=500)

        # Callbacks for drift events
        self._callbacks: List[Callable[[DriftEvent], None]] = []

        # Recovery tracking
        self._recovery_threshold = 200  # Samples needed for full recovery
        self._samples_since_any_drift = 0

        self._logger.info(
            f"DriftDetector initialized: delta={delta}, clock={clock}, "
            f"streams={list(self._streams.keys())}"
        )

    def _create_stream_state(self) -> StreamState:
        """Create a new stream state with configured ADWIN."""
        detector = drift.ADWIN(
            delta=self._delta,
            clock=self._clock,
            min_window_length=self._min_window_length,
            grace_period=self._grace_period,
        )
        state = StreamState()
        state.detector = detector
        return state

    @property
    def regime_state(self) -> RegimeState:
        """Get current market regime state."""
        return self._regime_state

    @property
    def overall_severity(self) -> DriftSeverity:
        """
        Get overall drift severity across all streams.

        Returns highest severity from any stream.
        """
        active_drifts = sum(1 for s in self._streams.values() if s.in_drift)

        if active_drifts == 0:
            return DriftSeverity.NONE
        elif active_drifts == 1:
            return DriftSeverity.MILD
        elif active_drifts <= 3:
            return DriftSeverity.MODERATE
        else:
            return DriftSeverity.SEVERE

    @property
    def is_drifting(self) -> bool:
        """Check if any stream is currently in drift."""
        return any(s.in_drift for s in self._streams.values())

    @property
    def total_detections(self) -> int:
        """Get total drift detections across all streams."""
        return self._total_detections

    @property
    def drifting_streams(self) -> List[str]:
        """Get names of streams currently in drift state."""
        return [name for name, state in self._streams.items() if state.in_drift]

    def update_stream(self, stream_name: str, value: float) -> bool:
        """
        Update a specific stream with a new value.

        Args:
            stream_name: Name of the stream to update
            value: New data point value

        Returns:
            True if drift was detected in this stream
        """
        if stream_name not in self._streams:
            # Auto-create new streams
            self._streams[stream_name] = self._create_stream_state()
            self._logger.info(f"Created new stream: {stream_name}")

        stream = self._streams[stream_name]
        drift_detected = stream.update(value)

        if drift_detected:
            self._handle_drift_event(stream_name, stream)
        else:
            self._samples_since_any_drift += 1

        # Update regime state based on overall conditions
        self._update_regime_state()

        return drift_detected

    def update_prediction_error(self, was_correct: bool) -> bool:
        """
        Update drift detector with prediction outcome.

        This is the primary interface for model accuracy drift detection.
        Feed in whether each prediction was correct to detect when the
        model's accuracy is degrading.

        Args:
            was_correct: True if prediction was correct, False otherwise

        Returns:
            True if drift was detected
        """
        # ADWIN expects error rate: 1 = error, 0 = correct
        error = 0.0 if was_correct else 1.0
        return self.update_stream('prediction_error', error)

    def update_returns(self, price_return: float) -> bool:
        """
        Update with price return for returns distribution drift.

        Args:
            price_return: Percentage price return

        Returns:
            True if drift was detected
        """
        return self.update_stream('returns', price_return)

    def update_volatility(self, volatility: float) -> bool:
        """
        Update with volatility estimate for regime change detection.

        Args:
            volatility: Volatility estimate (e.g., from rolling std)

        Returns:
            True if drift was detected
        """
        return self.update_stream('volatility', volatility)

    def update_multiple(self, values: Dict[str, float]) -> Dict[str, bool]:
        """
        Update multiple streams at once.

        Args:
            values: Dictionary of stream_name -> value

        Returns:
            Dictionary of stream_name -> drift_detected
        """
        results = {}
        for name, value in values.items():
            results[name] = self.update_stream(name, value)
        return results

    def _handle_drift_event(self, stream_name: str, stream: StreamState) -> None:
        """Handle a detected drift event."""
        self._total_detections += 1
        self._samples_since_any_drift = 0

        # Determine severity based on context
        severity = self._classify_severity(stream)

        # Create drift event
        event = DriftEvent(
            timestamp=datetime.now(timezone.utc),
            stream_name=stream_name,
            severity=severity,
            window_size=stream.width,
            estimation=stream.estimation,
            variance=stream.variance,
            detection_count=stream.detection_count,
        )

        self._events.append(event)

        self._logger.warning(
            f"Drift detected in '{stream_name}': severity={severity.name}, "
            f"window={stream.width}, estimation={stream.estimation:.4f}"
        )

        # Call registered callbacks
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception as e:
                self._logger.error(f"Drift callback error: {e}")

    def _classify_severity(self, stream: StreamState) -> DriftSeverity:
        """
        Classify drift severity based on stream characteristics.

        Args:
            stream: The stream that detected drift

        Returns:
            Severity classification
        """
        # Multiple recent drifts = more severe
        recent_drifts = len([
            t for t in stream.drift_history
            if (datetime.now(timezone.utc) - t).total_seconds() < 300  # 5 minutes
        ])

        if recent_drifts >= 3:
            return DriftSeverity.SEVERE
        elif recent_drifts >= 2:
            return DriftSeverity.MODERATE
        else:
            return DriftSeverity.MILD

    def _update_regime_state(self) -> None:
        """Update the overall market regime state."""
        old_state = self._regime_state
        drifting_count = sum(1 for s in self._streams.values() if s.in_drift)

        if drifting_count >= 3:
            self._regime_state = RegimeState.VOLATILE
        elif drifting_count >= 1:
            self._regime_state = RegimeState.TRANSITIONING
        elif self._samples_since_any_drift < self._recovery_threshold:
            self._regime_state = RegimeState.RECOVERING
        else:
            self._regime_state = RegimeState.STABLE

        if old_state != self._regime_state:
            self._last_regime_change = datetime.now(timezone.utc)
            self._logger.info(
                f"Regime state changed: {old_state.value} -> {self._regime_state.value}"
            )

    def register_callback(self, callback: Callable[[DriftEvent], None]) -> None:
        """
        Register a callback for drift events.

        Args:
            callback: Function to call when drift is detected.
                     Receives DriftEvent as argument.
        """
        self._callbacks.append(callback)

    def reset(self, stream_name: Optional[str] = None) -> None:
        """
        Reset drift detector state.

        Args:
            stream_name: If provided, reset only this stream. Otherwise reset all.
        """
        if stream_name:
            if stream_name in self._streams:
                self._streams[stream_name].reset()
                self._logger.info(f"Reset stream: {stream_name}")
        else:
            for stream in self._streams.values():
                stream.reset()
            self._total_detections = 0
            self._regime_state = RegimeState.STABLE
            self._samples_since_any_drift = 0
            self._events.clear()
            self._logger.info("Reset all streams")

    def get_stream_state(self, stream_name: str) -> Optional[dict]:
        """
        Get state of a specific stream.

        Args:
            stream_name: Name of stream

        Returns:
            Stream state dict or None if not found
        """
        if stream_name in self._streams:
            return self._streams[stream_name].to_dict()
        return None

    def get_recent_events(self, n: int = 20) -> List[dict]:
        """
        Get recent drift events.

        Args:
            n: Number of recent events to return

        Returns:
            List of event dictionaries
        """
        return [e.to_dict() for e in list(self._events)[-n:]]

    def get_state(self) -> dict:
        """
        Get complete detector state for persistence or monitoring.

        Returns:
            Dictionary with all state information
        """
        return {
            'regime_state': self._regime_state.value,
            'overall_severity': self.overall_severity.name,
            'is_drifting': self.is_drifting,
            'total_detections': self._total_detections,
            'drifting_streams': self.drifting_streams,
            'samples_since_any_drift': self._samples_since_any_drift,
            'last_regime_change': self._last_regime_change.isoformat() if self._last_regime_change else None,
            'streams': {
                name: state.to_dict()
                for name, state in self._streams.items()
            },
            'config': {
                'delta': self._delta,
                'clock': self._clock,
                'min_window_length': self._min_window_length,
                'grace_period': self._grace_period,
            },
        }

    def get_summary(self) -> dict:
        """
        Get summary for dashboard display.

        Returns:
            Concise summary of detector state
        """
        return {
            'regime': self._regime_state.value,
            'severity': self.overall_severity.name,
            'drifting': self.is_drifting,
            'total_detections': self._total_detections,
            'active_drifts': len(self.drifting_streams),
            'streams_monitored': len(self._streams),
        }


class MultiContractDriftDetector:
    """
    Manages drift detectors for multiple contracts.

    Each contract has its own DriftDetector to track regime changes
    independently, since different markets may exhibit different drift patterns.
    """

    def __init__(
        self,
        delta: float = 0.002,
        clock: int = 32,
        min_window_length: int = 5,
        grace_period: int = 10,
    ):
        """
        Initialize MultiContractDriftDetector.

        Args:
            delta: ADWIN delta for all detectors
            clock: ADWIN clock for all detectors
            min_window_length: ADWIN min window for all detectors
            grace_period: ADWIN grace period for all detectors
        """
        self._logger = logging.getLogger(__name__)

        self._delta = delta
        self._clock = clock
        self._min_window_length = min_window_length
        self._grace_period = grace_period

        # Per-contract detectors
        self._detectors: Dict[str, DriftDetector] = {}

        # Global drift tracking
        self._global_detections = 0

        self._logger.info("MultiContractDriftDetector initialized")

    def get_detector(self, contract_id: str) -> DriftDetector:
        """
        Get or create detector for a contract.

        Args:
            contract_id: Contract identifier

        Returns:
            DriftDetector for this contract
        """
        if contract_id not in self._detectors:
            self._detectors[contract_id] = DriftDetector(
                delta=self._delta,
                clock=self._clock,
                min_window_length=self._min_window_length,
                grace_period=self._grace_period,
            )
            self._logger.info(f"Created drift detector for: {contract_id}")

        return self._detectors[contract_id]

    def update_prediction_error(
        self,
        contract_id: str,
        was_correct: bool,
    ) -> bool:
        """
        Update prediction error for a contract.

        Args:
            contract_id: Contract identifier
            was_correct: Whether prediction was correct

        Returns:
            True if drift was detected
        """
        detector = self.get_detector(contract_id)
        result = detector.update_prediction_error(was_correct)

        if result:
            self._global_detections += 1

        return result

    @property
    def contract_count(self) -> int:
        """Get number of tracked contracts."""
        return len(self._detectors)

    @property
    def contracts_in_drift(self) -> List[str]:
        """Get list of contracts currently experiencing drift."""
        return [
            contract_id
            for contract_id, detector in self._detectors.items()
            if detector.is_drifting
        ]

    @property
    def overall_regime(self) -> RegimeState:
        """
        Get overall regime state across all contracts.

        Returns the most severe regime state among all contracts.
        """
        if not self._detectors:
            return RegimeState.STABLE

        states = [d.regime_state for d in self._detectors.values()]

        # Priority: VOLATILE > TRANSITIONING > RECOVERING > STABLE
        if RegimeState.VOLATILE in states:
            return RegimeState.VOLATILE
        if RegimeState.TRANSITIONING in states:
            return RegimeState.TRANSITIONING
        if RegimeState.RECOVERING in states:
            return RegimeState.RECOVERING
        return RegimeState.STABLE

    def get_all_summaries(self) -> Dict[str, dict]:
        """Get summaries for all contracts."""
        return {
            contract_id: detector.get_summary()
            for contract_id, detector in self._detectors.items()
        }

    def get_summary(self) -> dict:
        """Get aggregate summary across all contracts."""
        total_detections = sum(
            d.total_detections for d in self._detectors.values()
        )
        drifting_count = len(self.contracts_in_drift)

        return {
            'contract_count': self.contract_count,
            'contracts_in_drift': drifting_count,
            'overall_regime': self.overall_regime.value,
            'total_detections': total_detections,
            'global_detections': self._global_detections,
        }

    def reset_all(self) -> None:
        """Reset all contract detectors."""
        for detector in self._detectors.values():
            detector.reset()

        self._global_detections = 0
        self._logger.info("All drift detectors reset")
