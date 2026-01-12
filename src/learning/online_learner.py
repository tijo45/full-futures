"""
Online Learning Module - River-based continuous learning system.

Implements online machine learning using River's predict_one/learn_one pattern
for real-time adaptation from trading outcomes. Losses are treated as first-class
feedback signals for model improvement.

Key Features:
- River Pipeline with StandardScaler and LogisticRegression
- Incremental predict_one/learn_one updates
- Multiple metric tracking (accuracy, precision, recall, F1)
- ADWIN drift detection for market regime changes
- Per-contract model support
- Model state export/import for persistence

CRITICAL: Use learn_one() NOT fit_one(). Features must be dict, NOT numpy arrays.
"""

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from river import compose, drift, linear_model, metrics, optim, preprocessing

from config import get_config


class PredictionLabel(Enum):
    """Prediction labels for classification."""
    DOWN = 0
    UP = 1


@dataclass
class LearningMetrics:
    """
    Metrics tracking for model performance.

    Tracks multiple metrics to provide comprehensive performance view.
    """
    accuracy: metrics.Accuracy = field(default_factory=metrics.Accuracy)
    precision: metrics.Precision = field(default_factory=metrics.Precision)
    recall: metrics.Recall = field(default_factory=metrics.Recall)
    f1: metrics.F1 = field(default_factory=metrics.F1)

    # Confusion matrix tracking
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    # Cumulative counts
    total_predictions: int = 0
    total_correct: int = 0

    # Recent performance window
    recent_results: deque = field(default_factory=lambda: deque(maxlen=100))

    def update(self, y_true: int, y_pred: int) -> None:
        """
        Update all metrics with a new prediction result.

        Args:
            y_true: Actual outcome (0 or 1)
            y_pred: Predicted outcome (0 or 1)
        """
        # Update River metrics
        self.accuracy.update(y_true, y_pred)
        self.precision.update(y_true, y_pred)
        self.recall.update(y_true, y_pred)
        self.f1.update(y_true, y_pred)

        # Update confusion matrix
        if y_true == 1 and y_pred == 1:
            self.true_positives += 1
        elif y_true == 0 and y_pred == 0:
            self.true_negatives += 1
        elif y_true == 0 and y_pred == 1:
            self.false_positives += 1
        else:  # y_true == 1 and y_pred == 0
            self.false_negatives += 1

        # Update cumulative counts
        self.total_predictions += 1
        if y_true == y_pred:
            self.total_correct += 1

        # Track recent performance
        self.recent_results.append(int(y_true == y_pred))

    @property
    def recent_accuracy(self) -> float:
        """Get accuracy over recent predictions window."""
        if not self.recent_results:
            return 0.0
        return sum(self.recent_results) / len(self.recent_results)

    def to_dict(self) -> dict:
        """Export metrics as dictionary."""
        return {
            'accuracy': float(self.accuracy.get()),
            'precision': float(self.precision.get()),
            'recall': float(self.recall.get()),
            'f1': float(self.f1.get()),
            'recent_accuracy': self.recent_accuracy,
            'total_predictions': self.total_predictions,
            'total_correct': self.total_correct,
            'true_positives': self.true_positives,
            'true_negatives': self.true_negatives,
            'false_positives': self.false_positives,
            'false_negatives': self.false_negatives,
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self.accuracy = metrics.Accuracy()
        self.precision = metrics.Precision()
        self.recall = metrics.Recall()
        self.f1 = metrics.F1()
        self.true_positives = 0
        self.true_negatives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.total_predictions = 0
        self.total_correct = 0
        self.recent_results.clear()


@dataclass
class DriftState:
    """
    Drift detection state using ADWIN.

    Tracks concept drift to detect market regime changes.
    """
    detector: drift.ADWIN = field(default_factory=drift.ADWIN)

    # Drift event tracking
    drift_count: int = 0
    last_drift_time: Optional[datetime] = None
    drift_history: deque = field(default_factory=lambda: deque(maxlen=50))

    # Current state
    in_drift: bool = False
    samples_since_drift: int = 0

    def update(self, error: int) -> bool:
        """
        Update drift detector with prediction error.

        Args:
            error: 1 if prediction was wrong, 0 if correct

        Returns:
            True if drift was detected
        """
        self.detector.update(error)
        self.samples_since_drift += 1

        if self.detector.drift_detected:
            self.drift_count += 1
            self.last_drift_time = datetime.now(timezone.utc)
            self.drift_history.append(self.last_drift_time)
            self.in_drift = True
            self.samples_since_drift = 0
            return True

        # Recover from drift state after stable period
        if self.in_drift and self.samples_since_drift > 100:
            self.in_drift = False

        return False

    def to_dict(self) -> dict:
        """Export drift state as dictionary."""
        return {
            'drift_count': self.drift_count,
            'in_drift': self.in_drift,
            'samples_since_drift': self.samples_since_drift,
            'last_drift_time': self.last_drift_time.isoformat() if self.last_drift_time else None,
        }

    def reset(self) -> None:
        """Reset drift detector state."""
        self.detector = drift.ADWIN()
        self.drift_count = 0
        self.last_drift_time = None
        self.drift_history.clear()
        self.in_drift = False
        self.samples_since_drift = 0


@dataclass
class LearningHistory:
    """
    History tracking for learning operations.

    Maintains a buffer of recent predictions and outcomes for analysis.
    """
    max_history: int = 1000

    # Prediction history: (timestamp, features, prediction, probability, outcome)
    predictions: deque = field(default_factory=lambda: deque(maxlen=1000))

    # Outcome tracking for delayed learning
    pending_outcomes: Dict[str, dict] = field(default_factory=dict)

    def record_prediction(
        self,
        prediction_id: str,
        features: dict,
        prediction: int,
        probability: Optional[float] = None,
    ) -> None:
        """Record a prediction for later outcome matching."""
        self.predictions.append({
            'timestamp': datetime.now(timezone.utc),
            'prediction_id': prediction_id,
            'features': features.copy(),
            'prediction': prediction,
            'probability': probability,
            'outcome': None,
        })

        # Store for delayed outcome matching
        self.pending_outcomes[prediction_id] = {
            'features': features.copy(),
            'prediction': prediction,
            'probability': probability,
            'timestamp': datetime.now(timezone.utc),
        }

    def record_outcome(self, prediction_id: str, outcome: int) -> Optional[dict]:
        """
        Record outcome for a previous prediction.

        Args:
            prediction_id: ID of the prediction
            outcome: Actual outcome (0 or 1)

        Returns:
            The pending prediction dict if found, None otherwise
        """
        if prediction_id in self.pending_outcomes:
            pending = self.pending_outcomes.pop(prediction_id)
            pending['outcome'] = outcome
            return pending
        return None

    def get_recent_predictions(self, n: int = 100) -> List[dict]:
        """Get n most recent predictions."""
        return list(self.predictions)[-n:]

    def clear_stale_pending(self, max_age_seconds: int = 3600) -> int:
        """
        Clear pending outcomes older than max_age.

        Returns:
            Number of cleared entries
        """
        now = datetime.now(timezone.utc)
        stale_ids = []

        for pred_id, pending in self.pending_outcomes.items():
            age = (now - pending['timestamp']).total_seconds()
            if age > max_age_seconds:
                stale_ids.append(pred_id)

        for pred_id in stale_ids:
            del self.pending_outcomes[pred_id]

        return len(stale_ids)


class OnlineLearner:
    """
    Online learning system using River's predict_one/learn_one pattern.

    Implements continuous learning from trading outcomes with drift detection
    and comprehensive metric tracking. Designed for real-time adaptation
    where every trade outcome (especially losses) serves as learning feedback.

    CRITICAL PATTERNS:
    - Use learn_one() NOT fit_one()
    - Features must be Python dict, NOT numpy arrays
    - Losses are first-class feedback signals

    Usage:
        learner = OnlineLearner()

        # Make prediction
        prediction = learner.predict(features)  # features is dict

        # Later, when outcome is known
        learner.learn(features, outcome)  # outcome is 0 or 1

        # Or combined:
        prediction = learner.predict_and_learn(features, outcome)

        # Check for drift
        if learner.drift_detected:
            # Handle regime change
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        l2_regularization: float = 0.0,
        model_type: str = "logistic",
    ):
        """
        Initialize OnlineLearner.

        Args:
            learning_rate: Learning rate for gradient descent
            l2_regularization: L2 regularization strength (0 = none)
            model_type: Type of model ("logistic" or "perceptron")
        """
        self._config = get_config()
        self._logger = logging.getLogger(__name__)

        # Model configuration
        self._learning_rate = learning_rate
        self._l2_regularization = l2_regularization
        self._model_type = model_type

        # Create the River pipeline
        self._model = self._create_model()

        # Metrics tracking
        self._metrics = LearningMetrics()

        # Drift detection
        self._drift = DriftState()

        # History tracking
        self._history = LearningHistory()

        # State tracking
        self._is_warm = False
        self._warmup_samples = 50  # Minimum samples before model is "warm"
        self._samples_seen = 0
        self._last_prediction: Optional[int] = None
        self._last_probability: Optional[float] = None

        # Feature tracking for model introspection
        self._feature_names: List[str] = []

        # Callbacks for events
        self._on_drift_callbacks: List[Callable] = []

        self._logger.info(
            f"OnlineLearner initialized: model={model_type}, "
            f"lr={learning_rate}, l2={l2_regularization}"
        )

    def _create_model(self) -> compose.Pipeline:
        """
        Create the River model pipeline.

        Returns:
            River Pipeline with StandardScaler and classifier
        """
        if self._model_type == "logistic":
            classifier = linear_model.LogisticRegression(
                optimizer=optim.SGD(self._learning_rate),
                l2=self._l2_regularization,
            )
        elif self._model_type == "perceptron":
            classifier = linear_model.Perceptron(
                l2=self._l2_regularization,
            )
        else:
            # Default to logistic regression
            classifier = linear_model.LogisticRegression(
                optimizer=optim.SGD(self._learning_rate),
                l2=self._l2_regularization,
            )

        return compose.Pipeline(
            preprocessing.StandardScaler(),
            classifier
        )

    @property
    def is_warm(self) -> bool:
        """Check if model has seen enough samples to be reliable."""
        return self._samples_seen >= self._warmup_samples

    @property
    def samples_seen(self) -> int:
        """Get total number of training samples seen."""
        return self._samples_seen

    @property
    def drift_detected(self) -> bool:
        """Check if model is currently in drift state."""
        return self._drift.in_drift

    @property
    def metrics(self) -> LearningMetrics:
        """Get current metrics."""
        return self._metrics

    @property
    def last_prediction(self) -> Optional[int]:
        """Get the last prediction made."""
        return self._last_prediction

    @property
    def last_probability(self) -> Optional[float]:
        """Get probability of last prediction."""
        return self._last_probability

    def predict_one(self, x: dict) -> int:
        """
        Make a single prediction.

        CRITICAL: x must be a Python dict, NOT numpy array.

        Args:
            x: Feature dictionary {'feature1': value, 'feature2': value, ...}

        Returns:
            Predicted class (0 or 1)
        """
        if not isinstance(x, dict):
            raise TypeError(f"Features must be dict, got {type(x).__name__}")

        # Track feature names
        if not self._feature_names:
            self._feature_names = list(x.keys())

        # Get prediction
        y_pred = self._model.predict_one(x)

        # Default to 0 if model hasn't seen any data
        if y_pred is None:
            y_pred = 0

        self._last_prediction = int(y_pred)
        return self._last_prediction

    def predict_proba_one(self, x: dict) -> Dict[int, float]:
        """
        Get prediction probabilities.

        Args:
            x: Feature dictionary

        Returns:
            Dictionary of class probabilities {0: p0, 1: p1}
        """
        if not isinstance(x, dict):
            raise TypeError(f"Features must be dict, got {type(x).__name__}")

        proba = self._model.predict_proba_one(x)

        # Default probabilities if model hasn't seen data
        if proba is None or not proba:
            proba = {0: 0.5, 1: 0.5}

        # Store probability of predicted class
        self._last_probability = max(proba.values()) if proba else 0.5

        return proba

    def learn_one(self, x: dict, y: int) -> None:
        """
        Learn from a single sample.

        CRITICAL: Use learn_one() NOT fit_one().

        Args:
            x: Feature dictionary
            y: True label (0 or 1)
        """
        if not isinstance(x, dict):
            raise TypeError(f"Features must be dict, got {type(x).__name__}")

        if y not in (0, 1):
            raise ValueError(f"Label must be 0 or 1, got {y}")

        # Learn from this sample
        self._model.learn_one(x, y)
        self._samples_seen += 1

        # Check if model is now warm
        if not self._is_warm and self._samples_seen >= self._warmup_samples:
            self._is_warm = True
            self._logger.info(
                f"Model is now warm after {self._samples_seen} samples"
            )

    def predict_and_learn(
        self,
        x: dict,
        y_true: Optional[int] = None,
        prediction_id: Optional[str] = None,
    ) -> Tuple[int, Optional[float]]:
        """
        Make prediction and optionally learn from true outcome.

        This is the primary interface for online learning - predict first,
        then update the model when the true outcome is known.

        Args:
            x: Feature dictionary
            y_true: True label if known (0 or 1), None if not yet known
            prediction_id: Optional ID for tracking delayed outcomes

        Returns:
            Tuple of (prediction, probability)
        """
        # Get prediction and probability
        proba = self.predict_proba_one(x)
        y_pred = self.predict_one(x)

        # Get probability for positive class
        prob_positive = proba.get(1, 0.5)

        # Record prediction for history
        if prediction_id:
            self._history.record_prediction(
                prediction_id=prediction_id,
                features=x,
                prediction=y_pred,
                probability=prob_positive,
            )

        # Learn if outcome is known
        if y_true is not None:
            self._update_with_outcome(x, y_true, y_pred)

        return y_pred, prob_positive

    def learn_from_outcome(
        self,
        prediction_id: str,
        y_true: int,
    ) -> bool:
        """
        Learn from a delayed outcome.

        Use this when the true outcome becomes known after prediction.

        Args:
            prediction_id: ID of the original prediction
            y_true: Actual outcome (0 or 1)

        Returns:
            True if the prediction was found and learned from
        """
        pending = self._history.record_outcome(prediction_id, y_true)

        if pending is None:
            self._logger.warning(
                f"No pending prediction found for ID: {prediction_id}"
            )
            return False

        # Learn from the stored features
        self._update_with_outcome(
            x=pending['features'],
            y_true=y_true,
            y_pred=pending['prediction'],
        )

        return True

    def _update_with_outcome(
        self,
        x: dict,
        y_true: int,
        y_pred: int,
    ) -> None:
        """
        Internal method to update model and metrics with outcome.

        Args:
            x: Feature dictionary
            y_true: True label
            y_pred: Predicted label
        """
        # Learn from this sample - use learn_one NOT fit_one
        self.learn_one(x, y_true)

        # Update metrics
        self._metrics.update(y_true, y_pred)

        # Update drift detector (1 = error, 0 = correct)
        error = int(y_pred != y_true)
        drift_detected = self._drift.update(error)

        if drift_detected:
            self._logger.warning(
                f"Concept drift detected! Drift count: {self._drift.drift_count}"
            )
            self._handle_drift()

    def _handle_drift(self) -> None:
        """Handle detected concept drift."""
        # Call registered callbacks
        for callback in self._on_drift_callbacks:
            try:
                callback(self._drift.to_dict())
            except Exception as e:
                self._logger.error(f"Drift callback error: {e}")

        # Log drift event
        self._logger.info(
            f"Drift handled. Recent accuracy: {self._metrics.recent_accuracy:.3f}"
        )

    def register_drift_callback(self, callback: Callable[[dict], None]) -> None:
        """
        Register a callback for drift events.

        Args:
            callback: Function to call when drift is detected.
                     Receives drift state dict as argument.
        """
        self._on_drift_callbacks.append(callback)

    def reset_model(self, preserve_metrics: bool = False) -> None:
        """
        Reset the model to initial state.

        Useful after severe drift or regime change.

        Args:
            preserve_metrics: If True, keep historical metrics
        """
        self._model = self._create_model()
        self._samples_seen = 0
        self._is_warm = False
        self._feature_names = []

        if not preserve_metrics:
            self._metrics.reset()

        self._drift.reset()

        self._logger.info("Model reset to initial state")

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from model weights.

        Returns:
            Dictionary of feature names to importance scores
        """
        # Access the classifier (second step in pipeline)
        try:
            classifier = self._model.steps['LogisticRegression']
            weights = classifier.weights
            return dict(weights)
        except (KeyError, AttributeError):
            try:
                classifier = self._model.steps['Perceptron']
                weights = classifier.weights
                return dict(weights)
            except (KeyError, AttributeError):
                return {}

    def get_state(self) -> dict:
        """
        Get complete learner state for persistence or monitoring.

        Returns:
            Dictionary with all state information
        """
        return {
            'samples_seen': self._samples_seen,
            'is_warm': self._is_warm,
            'warmup_samples': self._warmup_samples,
            'learning_rate': self._learning_rate,
            'l2_regularization': self._l2_regularization,
            'model_type': self._model_type,
            'metrics': self._metrics.to_dict(),
            'drift': self._drift.to_dict(),
            'feature_names': self._feature_names,
            'last_prediction': self._last_prediction,
            'last_probability': self._last_probability,
            'pending_outcomes_count': len(self._history.pending_outcomes),
        }

    def get_summary(self) -> dict:
        """
        Get summary for dashboard display.

        Returns:
            Concise summary of learner state
        """
        return {
            'samples': self._samples_seen,
            'warm': self._is_warm,
            'accuracy': self._metrics.accuracy.get(),
            'recent_accuracy': self._metrics.recent_accuracy,
            'f1': self._metrics.f1.get(),
            'drift_count': self._drift.drift_count,
            'in_drift': self._drift.in_drift,
        }

    def clear_stale_history(self, max_age_seconds: int = 3600) -> int:
        """
        Clear old pending outcomes.

        Args:
            max_age_seconds: Maximum age in seconds

        Returns:
            Number of cleared entries
        """
        return self._history.clear_stale_pending(max_age_seconds)


class MultiContractLearner:
    """
    Manages multiple OnlineLearner instances, one per contract.

    Allows per-contract model adaptation while sharing common configuration.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        l2_regularization: float = 0.0,
        model_type: str = "logistic",
    ):
        """
        Initialize MultiContractLearner.

        Args:
            learning_rate: Learning rate for all learners
            l2_regularization: L2 regularization for all learners
            model_type: Model type for all learners
        """
        self._logger = logging.getLogger(__name__)

        self._learning_rate = learning_rate
        self._l2_regularization = l2_regularization
        self._model_type = model_type

        # Per-contract learners
        self._learners: Dict[str, OnlineLearner] = {}

        # Global metrics (aggregate across all contracts)
        self._global_metrics = LearningMetrics()

        self._logger.info("MultiContractLearner initialized")

    def get_learner(self, contract_id: str) -> OnlineLearner:
        """
        Get or create learner for a contract.

        Args:
            contract_id: Contract identifier

        Returns:
            OnlineLearner for this contract
        """
        if contract_id not in self._learners:
            self._learners[contract_id] = OnlineLearner(
                learning_rate=self._learning_rate,
                l2_regularization=self._l2_regularization,
                model_type=self._model_type,
            )
            self._logger.info(f"Created learner for contract: {contract_id}")

        return self._learners[contract_id]

    def predict_and_learn(
        self,
        contract_id: str,
        x: dict,
        y_true: Optional[int] = None,
    ) -> Tuple[int, Optional[float]]:
        """
        Predict and learn for a specific contract.

        Args:
            contract_id: Contract identifier
            x: Feature dictionary
            y_true: True outcome if known

        Returns:
            Tuple of (prediction, probability)
        """
        learner = self.get_learner(contract_id)
        y_pred, prob = learner.predict_and_learn(x, y_true)

        # Update global metrics if outcome is known
        if y_true is not None:
            self._global_metrics.update(y_true, y_pred)

        return y_pred, prob

    @property
    def contract_count(self) -> int:
        """Get number of tracked contracts."""
        return len(self._learners)

    @property
    def global_accuracy(self) -> float:
        """Get global accuracy across all contracts."""
        return float(self._global_metrics.accuracy.get())

    def get_all_summaries(self) -> Dict[str, dict]:
        """Get summaries for all contracts."""
        return {
            contract_id: learner.get_summary()
            for contract_id, learner in self._learners.items()
        }

    def get_summary(self) -> dict:
        """Get aggregate summary."""
        total_samples = sum(l.samples_seen for l in self._learners.values())
        warm_count = sum(1 for l in self._learners.values() if l.is_warm)
        drift_count = sum(l._drift.drift_count for l in self._learners.values())

        return {
            'contract_count': self.contract_count,
            'total_samples': total_samples,
            'warm_learners': warm_count,
            'global_accuracy': self.global_accuracy,
            'total_drift_events': drift_count,
            'global_metrics': self._global_metrics.to_dict(),
        }

    def reset_all(self, preserve_metrics: bool = False) -> None:
        """Reset all learners."""
        for learner in self._learners.values():
            learner.reset_model(preserve_metrics)

        if not preserve_metrics:
            self._global_metrics.reset()

        self._logger.info("All learners reset")
