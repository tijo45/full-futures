"""
Predictor Module - Integrates Feature Engine with Online Learner.

Provides a unified interface for making predictions from market data by
combining feature extraction with online machine learning. Handles the
complete prediction pipeline: market data -> features -> prediction.

Key Features:
- Integration of FeatureEngine and OnlineLearner
- Per-contract prediction support
- Confidence scoring with prediction tracking
- Delayed learning from trade outcomes
- Data quality gating (staleness, feature completeness)
- Prediction history and statistics

CRITICAL: Features must be Python dict for River compatibility.
Use learn_one() NOT fit_one() for model updates.
"""

import logging
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from config import get_config
from src.data.feature_engine import FeatureEngine
from src.data.market_data import TickData, DepthData
from src.learning.online_learner import OnlineLearner, MultiContractLearner, PredictionLabel


class PredictionSignal(Enum):
    """Trading signal from prediction."""
    NEUTRAL = 0
    BUY = 1
    SELL = -1


@dataclass
class PredictionResult:
    """
    Result of a prediction including confidence and metadata.

    Contains all information needed to make a trading decision.
    """
    prediction_id: str
    contract_id: int
    symbol: str
    timestamp: datetime

    # Prediction output
    signal: PredictionSignal
    direction: int  # 1 for up/buy, 0 for down/sell
    probability: float  # Probability of predicted direction

    # Confidence metrics
    confidence: float  # Overall confidence score (0.0 to 1.0)
    model_warm: bool  # Whether model has seen enough data

    # Feature metadata
    feature_count: int  # Number of features used
    data_age_seconds: float  # Age of underlying market data

    # Quality flags
    is_valid: bool = True  # Whether prediction is usable
    rejection_reason: Optional[str] = None

    def to_dict(self) -> dict:
        """Export prediction result as dictionary."""
        return {
            'prediction_id': self.prediction_id,
            'contract_id': self.contract_id,
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'signal': self.signal.name,
            'direction': self.direction,
            'probability': self.probability,
            'confidence': self.confidence,
            'model_warm': self.model_warm,
            'feature_count': self.feature_count,
            'data_age_seconds': self.data_age_seconds,
            'is_valid': self.is_valid,
            'rejection_reason': self.rejection_reason,
        }


@dataclass
class PredictionStats:
    """
    Statistics tracking for prediction performance.

    Maintains running statistics for monitoring and dashboard display.
    """
    total_predictions: int = 0
    valid_predictions: int = 0
    rejected_predictions: int = 0

    # Rejection reasons
    rejections_by_reason: Dict[str, int] = field(default_factory=dict)

    # Confidence tracking
    confidence_sum: float = 0.0
    high_confidence_count: int = 0  # Above threshold

    # Recent predictions
    recent_predictions: deque = field(default_factory=lambda: deque(maxlen=100))

    # Timing
    last_prediction_time: Optional[datetime] = None

    def record_prediction(
        self,
        result: PredictionResult,
        confidence_threshold: float = 0.6,
    ) -> None:
        """Record a prediction result."""
        self.total_predictions += 1
        self.last_prediction_time = result.timestamp

        if result.is_valid:
            self.valid_predictions += 1
            self.confidence_sum += result.confidence

            if result.confidence >= confidence_threshold:
                self.high_confidence_count += 1
        else:
            self.rejected_predictions += 1
            reason = result.rejection_reason or "unknown"
            self.rejections_by_reason[reason] = (
                self.rejections_by_reason.get(reason, 0) + 1
            )

        # Track recent predictions
        self.recent_predictions.append({
            'timestamp': result.timestamp,
            'symbol': result.symbol,
            'signal': result.signal.name,
            'confidence': result.confidence,
            'is_valid': result.is_valid,
        })

    @property
    def average_confidence(self) -> float:
        """Get average confidence of valid predictions."""
        if self.valid_predictions == 0:
            return 0.0
        return self.confidence_sum / self.valid_predictions

    @property
    def valid_rate(self) -> float:
        """Get rate of valid predictions."""
        if self.total_predictions == 0:
            return 0.0
        return self.valid_predictions / self.total_predictions

    def to_dict(self) -> dict:
        """Export statistics as dictionary."""
        return {
            'total_predictions': self.total_predictions,
            'valid_predictions': self.valid_predictions,
            'rejected_predictions': self.rejected_predictions,
            'valid_rate': self.valid_rate,
            'average_confidence': self.average_confidence,
            'high_confidence_count': self.high_confidence_count,
            'rejections_by_reason': dict(self.rejections_by_reason),
            'last_prediction_time': (
                self.last_prediction_time.isoformat()
                if self.last_prediction_time else None
            ),
        }

    def reset(self) -> None:
        """Reset all statistics."""
        self.total_predictions = 0
        self.valid_predictions = 0
        self.rejected_predictions = 0
        self.rejections_by_reason.clear()
        self.confidence_sum = 0.0
        self.high_confidence_count = 0
        self.recent_predictions.clear()
        self.last_prediction_time = None


class Predictor:
    """
    Unified prediction engine integrating features with online learning.

    Combines FeatureEngine for feature extraction and OnlineLearner for
    predictions, providing a complete pipeline from market data to
    actionable trading signals.

    CRITICAL PATTERNS:
    - Features are Python dicts, NOT numpy arrays
    - Use learn_one() NOT fit_one() for model updates
    - Data freshness is gated before predictions

    Usage:
        predictor = Predictor()

        # Make prediction from market data
        result = predictor.predict(tick_data, depth_data)

        if result.is_valid and result.confidence >= threshold:
            # Execute trade based on result.signal

        # Later, when outcome is known
        predictor.learn_from_outcome(result.prediction_id, outcome)

        # Or direct learning:
        predictor.predict_and_learn(tick_data, depth_data, y_true=outcome)
    """

    # Default thresholds
    DEFAULT_MIN_FEATURES = 5
    DEFAULT_MAX_DATA_AGE_SECONDS = 30.0
    DEFAULT_MIN_CONFIDENCE = 0.5
    DEFAULT_CONFIDENCE_THRESHOLD = 0.6

    def __init__(
        self,
        feature_engine: Optional[FeatureEngine] = None,
        learner: Optional[OnlineLearner] = None,
        use_multi_contract: bool = True,
        min_features: int = None,
        max_data_age_seconds: float = None,
        min_confidence: float = None,
    ):
        """
        Initialize Predictor.

        Args:
            feature_engine: FeatureEngine instance (created if not provided)
            learner: OnlineLearner instance (created if not provided)
            use_multi_contract: Use separate learners per contract
            min_features: Minimum features required for prediction
            max_data_age_seconds: Maximum age of data for prediction
            min_confidence: Minimum confidence to consider prediction valid
        """
        self._config = get_config()
        self._logger = logging.getLogger(__name__)

        # Feature extraction
        self._feature_engine = feature_engine or FeatureEngine()

        # Learning mode
        self._use_multi_contract = use_multi_contract

        # Single or multi-contract learner
        if use_multi_contract:
            self._multi_learner = MultiContractLearner()
            self._learner = None
        else:
            self._learner = learner or OnlineLearner()
            self._multi_learner = None

        # Quality thresholds
        self._min_features = min_features or self.DEFAULT_MIN_FEATURES
        self._max_data_age_seconds = (
            max_data_age_seconds or self.DEFAULT_MAX_DATA_AGE_SECONDS
        )
        self._min_confidence = min_confidence or self.DEFAULT_MIN_CONFIDENCE

        # Statistics
        self._stats = PredictionStats()
        self._per_contract_stats: Dict[int, PredictionStats] = {}

        # Pending predictions for delayed learning
        self._pending_predictions: Dict[str, dict] = {}

        # Callbacks
        self._on_prediction_callbacks: List[Callable[[PredictionResult], None]] = []
        self._on_drift_callbacks: List[Callable[[str, dict], None]] = []

        # Register drift callback on learner(s)
        self._setup_drift_callbacks()

        self._logger.info(
            f"Predictor initialized: multi_contract={use_multi_contract}, "
            f"min_features={self._min_features}, max_age={self._max_data_age_seconds}s"
        )

    def _setup_drift_callbacks(self) -> None:
        """Set up drift detection callbacks."""
        if self._learner:
            self._learner.register_drift_callback(
                lambda drift_info: self._handle_drift("global", drift_info)
            )
        # Note: Multi-contract learner drift callbacks are set per-contract

    def _handle_drift(self, contract_id: str, drift_info: dict) -> None:
        """Handle drift detection event."""
        self._logger.warning(f"Drift detected for {contract_id}: {drift_info}")
        for callback in self._on_drift_callbacks:
            try:
                callback(contract_id, drift_info)
            except Exception as e:
                self._logger.error(f"Drift callback error: {e}")

    def _get_learner(self, contract_id: int, symbol: str) -> OnlineLearner:
        """Get appropriate learner for contract."""
        if self._use_multi_contract:
            learner = self._multi_learner.get_learner(str(contract_id))
            # Set up drift callback for new learners
            if learner.samples_seen == 0:
                learner.register_drift_callback(
                    lambda info, cid=str(contract_id): self._handle_drift(cid, info)
                )
            return learner
        return self._learner

    def _get_contract_stats(self, contract_id: int) -> PredictionStats:
        """Get or create stats for a contract."""
        if contract_id not in self._per_contract_stats:
            self._per_contract_stats[contract_id] = PredictionStats()
        return self._per_contract_stats[contract_id]

    def _generate_prediction_id(self) -> str:
        """Generate unique prediction ID."""
        return str(uuid.uuid4())

    def _validate_data(
        self,
        tick_data: Optional[TickData],
        depth_data: Optional[DepthData],
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate market data for prediction.

        Args:
            tick_data: Level 1 tick data
            depth_data: Level 2 depth data

        Returns:
            Tuple of (is_valid, rejection_reason)
        """
        # Require at least one data source
        if tick_data is None and depth_data is None:
            return False, "no_market_data"

        # Check data staleness
        max_age = 0.0
        if tick_data is not None:
            max_age = max(max_age, tick_data.age_seconds)
        if depth_data is not None:
            max_age = max(max_age, depth_data.age_seconds)

        if max_age > self._max_data_age_seconds:
            return False, f"stale_data_{max_age:.1f}s"

        return True, None

    def _calculate_confidence(
        self,
        probability: float,
        learner: OnlineLearner,
        feature_count: int,
    ) -> float:
        """
        Calculate overall prediction confidence.

        Combines model probability with warmth and feature completeness.

        Args:
            probability: Raw probability from model
            learner: OnlineLearner instance
            feature_count: Number of features extracted

        Returns:
            Confidence score (0.0 to 1.0)
        """
        # Base confidence from probability (distance from 0.5)
        base_confidence = abs(probability - 0.5) * 2

        # Model warmth factor
        warmth_factor = 1.0 if learner.is_warm else (
            learner.samples_seen / learner._warmup_samples
        )

        # Feature completeness factor
        expected_features = 30  # Approximate expected feature count
        feature_factor = min(1.0, feature_count / expected_features)

        # Recent performance factor (if model is warm)
        perf_factor = 1.0
        if learner.is_warm:
            recent_acc = learner.metrics.recent_accuracy
            if recent_acc > 0:
                perf_factor = min(1.0, recent_acc / 0.55)  # Scale to baseline

        # Combine factors (weighted average)
        confidence = (
            base_confidence * 0.4 +
            warmth_factor * 0.2 +
            feature_factor * 0.2 +
            perf_factor * 0.2
        )

        return max(0.0, min(1.0, confidence))

    def predict(
        self,
        tick_data: Optional[TickData] = None,
        depth_data: Optional[DepthData] = None,
    ) -> PredictionResult:
        """
        Make a prediction from market data.

        Args:
            tick_data: Level 1 tick data
            depth_data: Level 2 order book data

        Returns:
            PredictionResult with signal, confidence, and metadata
        """
        prediction_id = self._generate_prediction_id()
        now = datetime.now(timezone.utc)

        # Determine contract from data
        if tick_data is not None:
            contract_id = tick_data.contract_id
            symbol = tick_data.symbol
        elif depth_data is not None:
            contract_id = depth_data.contract_id
            symbol = depth_data.symbol
        else:
            # Return invalid result for missing data
            return self._create_invalid_result(
                prediction_id, 0, "unknown", now, "no_market_data"
            )

        # Validate data
        is_valid, rejection_reason = self._validate_data(tick_data, depth_data)
        if not is_valid:
            return self._create_invalid_result(
                prediction_id, contract_id, symbol, now, rejection_reason
            )

        # Extract features
        features = self._feature_engine.extract_features(
            tick_data=tick_data,
            depth_data=depth_data,
        )

        # Check minimum features
        if len(features) < self._min_features:
            return self._create_invalid_result(
                prediction_id, contract_id, symbol, now,
                f"insufficient_features_{len(features)}"
            )

        # Get learner and make prediction
        learner = self._get_learner(contract_id, symbol)

        # Get prediction probabilities
        proba = learner.predict_proba_one(features)
        direction = learner.predict_one(features)

        # Get probability for predicted direction
        probability = proba.get(direction, 0.5)

        # Calculate confidence
        confidence = self._calculate_confidence(
            probability, learner, len(features)
        )

        # Determine signal
        if confidence < self._min_confidence:
            signal = PredictionSignal.NEUTRAL
        elif direction == 1:
            signal = PredictionSignal.BUY
        else:
            signal = PredictionSignal.SELL

        # Calculate data age
        data_age = 0.0
        if tick_data:
            data_age = max(data_age, tick_data.age_seconds)
        if depth_data:
            data_age = max(data_age, depth_data.age_seconds)

        # Create result
        result = PredictionResult(
            prediction_id=prediction_id,
            contract_id=contract_id,
            symbol=symbol,
            timestamp=now,
            signal=signal,
            direction=direction,
            probability=probability,
            confidence=confidence,
            model_warm=learner.is_warm,
            feature_count=len(features),
            data_age_seconds=data_age,
            is_valid=True,
        )

        # Store for delayed learning
        self._pending_predictions[prediction_id] = {
            'features': features,
            'direction': direction,
            'contract_id': contract_id,
            'symbol': symbol,
            'timestamp': now,
        }

        # Record to learner history
        learner._history.record_prediction(
            prediction_id=prediction_id,
            features=features,
            prediction=direction,
            probability=probability,
        )

        # Update statistics
        self._stats.record_prediction(result, self.DEFAULT_CONFIDENCE_THRESHOLD)
        self._get_contract_stats(contract_id).record_prediction(
            result, self.DEFAULT_CONFIDENCE_THRESHOLD
        )

        # Invoke callbacks
        for callback in self._on_prediction_callbacks:
            try:
                callback(result)
            except Exception as e:
                self._logger.error(f"Prediction callback error: {e}")

        return result

    def predict_and_learn(
        self,
        tick_data: Optional[TickData] = None,
        depth_data: Optional[DepthData] = None,
        y_true: Optional[int] = None,
    ) -> PredictionResult:
        """
        Make prediction and optionally learn from outcome.

        Args:
            tick_data: Level 1 tick data
            depth_data: Level 2 order book data
            y_true: True outcome if known (0 for down, 1 for up)

        Returns:
            PredictionResult with signal and confidence
        """
        # Make prediction
        result = self.predict(tick_data, depth_data)

        # Learn if outcome is known
        if y_true is not None and result.is_valid:
            self.learn_from_outcome(result.prediction_id, y_true)

        return result

    def learn_from_outcome(
        self,
        prediction_id: str,
        y_true: int,
    ) -> bool:
        """
        Learn from a delayed outcome.

        Use this when the true outcome becomes known after prediction
        (e.g., after a trade is closed).

        Args:
            prediction_id: ID of the original prediction
            y_true: Actual outcome (0 for down/sell, 1 for up/buy)

        Returns:
            True if learning was successful
        """
        if prediction_id not in self._pending_predictions:
            self._logger.warning(
                f"No pending prediction found for ID: {prediction_id}"
            )
            return False

        pending = self._pending_predictions.pop(prediction_id)
        features = pending['features']
        contract_id = pending['contract_id']

        # Get learner and update
        learner = self._get_learner(contract_id, pending['symbol'])

        # Use learner's delayed learning
        success = learner.learn_from_outcome(prediction_id, y_true)

        if success:
            self._logger.debug(
                f"Learned from outcome for {pending['symbol']}: "
                f"predicted={pending['direction']}, actual={y_true}"
            )
        else:
            # Direct learn if not in learner's history
            learner.learn_one(features, y_true)
            learner.metrics.update(y_true, pending['direction'])
            self._logger.debug(
                f"Direct learned for {pending['symbol']}: actual={y_true}"
            )

        return True

    def _create_invalid_result(
        self,
        prediction_id: str,
        contract_id: int,
        symbol: str,
        timestamp: datetime,
        rejection_reason: str,
    ) -> PredictionResult:
        """Create an invalid prediction result."""
        result = PredictionResult(
            prediction_id=prediction_id,
            contract_id=contract_id,
            symbol=symbol,
            timestamp=timestamp,
            signal=PredictionSignal.NEUTRAL,
            direction=0,
            probability=0.5,
            confidence=0.0,
            model_warm=False,
            feature_count=0,
            data_age_seconds=0.0,
            is_valid=False,
            rejection_reason=rejection_reason,
        )

        # Record rejection
        self._stats.record_prediction(result, self.DEFAULT_CONFIDENCE_THRESHOLD)
        if contract_id:
            self._get_contract_stats(contract_id).record_prediction(
                result, self.DEFAULT_CONFIDENCE_THRESHOLD
            )

        return result

    def register_prediction_callback(
        self,
        callback: Callable[[PredictionResult], None],
    ) -> None:
        """
        Register a callback for prediction events.

        Args:
            callback: Function called with PredictionResult on each prediction
        """
        self._on_prediction_callbacks.append(callback)

    def register_drift_callback(
        self,
        callback: Callable[[str, dict], None],
    ) -> None:
        """
        Register a callback for drift detection events.

        Args:
            callback: Function called with (contract_id, drift_info) on drift
        """
        self._on_drift_callbacks.append(callback)

    def get_pending_count(self) -> int:
        """Get count of predictions awaiting outcome."""
        return len(self._pending_predictions)

    def clear_stale_pending(self, max_age_seconds: int = 3600) -> int:
        """
        Clear old pending predictions.

        Args:
            max_age_seconds: Maximum age in seconds

        Returns:
            Number of cleared entries
        """
        now = datetime.now(timezone.utc)
        stale_ids = []

        for pred_id, pending in self._pending_predictions.items():
            age = (now - pending['timestamp']).total_seconds()
            if age > max_age_seconds:
                stale_ids.append(pred_id)

        for pred_id in stale_ids:
            del self._pending_predictions[pred_id]

        if stale_ids:
            self._logger.info(f"Cleared {len(stale_ids)} stale pending predictions")

        return len(stale_ids)

    def reset(self, preserve_metrics: bool = False) -> None:
        """
        Reset predictor state.

        Args:
            preserve_metrics: If True, keep statistics
        """
        self._feature_engine.reset_all()

        if self._use_multi_contract:
            self._multi_learner.reset_all(preserve_metrics)
        else:
            self._learner.reset_model(preserve_metrics)

        self._pending_predictions.clear()

        if not preserve_metrics:
            self._stats.reset()
            self._per_contract_stats.clear()

        self._logger.info("Predictor reset")

    def get_state(self) -> dict:
        """
        Get complete predictor state for monitoring.

        Returns:
            Dictionary with all state information
        """
        learner_state = (
            self._multi_learner.get_summary()
            if self._use_multi_contract
            else self._learner.get_state()
        )

        return {
            'use_multi_contract': self._use_multi_contract,
            'min_features': self._min_features,
            'max_data_age_seconds': self._max_data_age_seconds,
            'min_confidence': self._min_confidence,
            'pending_predictions': self.get_pending_count(),
            'feature_engine': self._feature_engine.get_summary(),
            'learner': learner_state,
            'stats': self._stats.to_dict(),
        }

    def get_summary(self) -> dict:
        """
        Get concise summary for dashboard display.

        Returns:
            Summary dictionary
        """
        if self._use_multi_contract:
            learner_summary = self._multi_learner.get_summary()
            samples = learner_summary['total_samples']
            accuracy = learner_summary['global_accuracy']
            warm_count = learner_summary['warm_learners']
            drift_count = learner_summary['total_drift_events']
        else:
            samples = self._learner.samples_seen
            accuracy = self._learner.metrics.accuracy.get()
            warm_count = 1 if self._learner.is_warm else 0
            drift_count = self._learner._drift.drift_count

        return {
            'total_predictions': self._stats.total_predictions,
            'valid_predictions': self._stats.valid_predictions,
            'valid_rate': self._stats.valid_rate,
            'average_confidence': self._stats.average_confidence,
            'samples_learned': samples,
            'model_accuracy': accuracy,
            'warm_learners': warm_count,
            'drift_events': drift_count,
            'pending_outcomes': self.get_pending_count(),
            'contracts_tracked': self._feature_engine.contract_count,
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

        learner_summary = {}
        if self._use_multi_contract:
            learner = self._multi_learner._learners.get(str(contract_id))
            if learner:
                learner_summary = learner.get_summary()

        features = self._feature_engine.get_contract_features(contract_id)
        feature_summary = {
            'feature_count': features.feature_count if features else 0,
            'price_observations': features.price_stats.count if features else 0,
        }

        return {
            'stats': stats.to_dict(),
            'learner': learner_summary,
            'features': feature_summary,
        }

    @property
    def feature_engine(self) -> FeatureEngine:
        """Get the feature engine."""
        return self._feature_engine

    @property
    def stats(self) -> PredictionStats:
        """Get global prediction statistics."""
        return self._stats
