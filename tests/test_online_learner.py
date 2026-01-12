"""
Unit tests for OnlineLearner - River-based continuous learning system.

Tests cover:
- Model initialization and configuration
- predict_one and learn_one operations
- Metrics tracking
- Drift detection
- History and delayed learning
- Multi-contract learning

NOTE: These tests require the 'river' package to be installed.
Tests will be skipped if river is not available.
"""

import pytest
import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

sys.path.insert(0, '.')

# Check if river is available
try:
    import river
    RIVER_AVAILABLE = True
except ImportError:
    RIVER_AVAILABLE = False

# Skip all tests in this module if river is not available
pytestmark = pytest.mark.skipif(
    not RIVER_AVAILABLE,
    reason="river package not installed"
)


@pytest.fixture
def mock_config():
    """Mock configuration for online learner."""
    config = MagicMock()
    return config


@pytest.fixture
def sample_features():
    """Sample feature dictionary for ML tests."""
    return {
        'spread': 0.25,
        'quote_imbalance': 0.2,
        'price_position': 0.5,
        'book_imbalance': 0.1,
        'liquidity_score': 0.8,
        'vwap_distance': 0.01,
        'momentum': 0.05,
        'volatility': 0.02,
        'z_score': 0.5,
    }


@pytest.mark.skipif(not RIVER_AVAILABLE, reason="river package not installed")
class TestOnlineLearnerInitialization:
    """Tests for OnlineLearner initialization."""

    def test_initialization_with_defaults(self, mock_config):
        """Test OnlineLearner initializes with default parameters."""
        with patch('src.learning.online_learner.get_config', return_value=mock_config):
            from src.learning.online_learner import OnlineLearner

            learner = OnlineLearner()

            assert learner._learning_rate == 0.1
            assert learner._l2_regularization == 0.0
            assert learner._model_type == "logistic"
            assert learner._samples_seen == 0
            assert learner._is_warm is False

    def test_initialization_with_custom_params(self, mock_config):
        """Test OnlineLearner initializes with custom parameters."""
        with patch('src.learning.online_learner.get_config', return_value=mock_config):
            from src.learning.online_learner import OnlineLearner

            learner = OnlineLearner(
                learning_rate=0.05,
                l2_regularization=0.01,
                model_type="perceptron"
            )

            assert learner._learning_rate == 0.05
            assert learner._l2_regularization == 0.01
            assert learner._model_type == "perceptron"

    def test_model_creation_logistic(self, mock_config):
        """Test logistic regression model is created correctly."""
        with patch('src.learning.online_learner.get_config', return_value=mock_config):
            from src.learning.online_learner import OnlineLearner

            learner = OnlineLearner(model_type="logistic")

            # Model should be a pipeline
            assert learner._model is not None
            assert hasattr(learner._model, 'predict_one')
            assert hasattr(learner._model, 'learn_one')

    def test_model_creation_perceptron(self, mock_config):
        """Test perceptron model is created correctly."""
        with patch('src.learning.online_learner.get_config', return_value=mock_config):
            from src.learning.online_learner import OnlineLearner

            learner = OnlineLearner(model_type="perceptron")

            assert learner._model is not None


@pytest.mark.skipif(not RIVER_AVAILABLE, reason="river package not installed")
class TestOnlineLearnerPrediction:
    """Tests for prediction methods."""

    def test_predict_one_returns_int(self, mock_config, sample_features):
        """Test predict_one returns integer prediction."""
        with patch('src.learning.online_learner.get_config', return_value=mock_config):
            from src.learning.online_learner import OnlineLearner

            learner = OnlineLearner()

            prediction = learner.predict_one(sample_features)

            assert isinstance(prediction, int)
            assert prediction in (0, 1)

    def test_predict_one_requires_dict(self, mock_config):
        """Test predict_one raises TypeError for non-dict input."""
        with patch('src.learning.online_learner.get_config', return_value=mock_config):
            from src.learning.online_learner import OnlineLearner

            learner = OnlineLearner()

            with pytest.raises(TypeError) as exc_info:
                learner.predict_one([1, 2, 3])  # List instead of dict

            assert "must be dict" in str(exc_info.value)

    def test_predict_proba_one_returns_dict(self, mock_config, sample_features):
        """Test predict_proba_one returns probability dictionary."""
        with patch('src.learning.online_learner.get_config', return_value=mock_config):
            from src.learning.online_learner import OnlineLearner

            learner = OnlineLearner()

            proba = learner.predict_proba_one(sample_features)

            assert isinstance(proba, dict)
            assert 0 in proba or 1 in proba or True in proba or False in proba
            # Probabilities should sum to ~1
            assert 0.99 <= sum(proba.values()) <= 1.01

    def test_predict_one_tracks_feature_names(self, mock_config, sample_features):
        """Test predict_one tracks feature names on first call."""
        with patch('src.learning.online_learner.get_config', return_value=mock_config):
            from src.learning.online_learner import OnlineLearner

            learner = OnlineLearner()
            assert learner._feature_names == []

            learner.predict_one(sample_features)

            assert learner._feature_names == list(sample_features.keys())


@pytest.mark.skipif(not RIVER_AVAILABLE, reason="river package not installed")
class TestOnlineLearnerLearning:
    """Tests for learning methods."""

    def test_learn_one_updates_model(self, mock_config, sample_features):
        """Test learn_one updates the model."""
        with patch('src.learning.online_learner.get_config', return_value=mock_config):
            from src.learning.online_learner import OnlineLearner

            learner = OnlineLearner()
            initial_samples = learner._samples_seen

            learner.learn_one(sample_features, 1)

            assert learner._samples_seen == initial_samples + 1

    def test_learn_one_requires_dict(self, mock_config):
        """Test learn_one raises TypeError for non-dict input."""
        with patch('src.learning.online_learner.get_config', return_value=mock_config):
            from src.learning.online_learner import OnlineLearner

            learner = OnlineLearner()

            with pytest.raises(TypeError) as exc_info:
                learner.learn_one([1, 2, 3], 1)

            assert "must be dict" in str(exc_info.value)

    def test_learn_one_requires_valid_label(self, mock_config, sample_features):
        """Test learn_one raises ValueError for invalid label."""
        with patch('src.learning.online_learner.get_config', return_value=mock_config):
            from src.learning.online_learner import OnlineLearner

            learner = OnlineLearner()

            with pytest.raises(ValueError) as exc_info:
                learner.learn_one(sample_features, 5)  # Invalid label

            assert "must be 0 or 1" in str(exc_info.value)

    def test_model_becomes_warm_after_enough_samples(self, mock_config, sample_features):
        """Test model becomes warm after warmup_samples."""
        with patch('src.learning.online_learner.get_config', return_value=mock_config):
            from src.learning.online_learner import OnlineLearner

            learner = OnlineLearner()
            learner._warmup_samples = 5  # Lower threshold for testing

            # Learn until warm
            for i in range(5):
                learner.learn_one(sample_features, i % 2)

            assert learner.is_warm is True


@pytest.mark.skipif(not RIVER_AVAILABLE, reason="river package not installed")
class TestOnlineLearnerPredictAndLearn:
    """Tests for combined predict and learn operation."""

    def test_predict_and_learn_with_outcome(self, mock_config, sample_features):
        """Test predict_and_learn updates model when outcome provided."""
        with patch('src.learning.online_learner.get_config', return_value=mock_config):
            from src.learning.online_learner import OnlineLearner

            learner = OnlineLearner()
            initial_samples = learner._samples_seen

            prediction, probability = learner.predict_and_learn(sample_features, y_true=1)

            assert isinstance(prediction, int)
            assert isinstance(probability, float)
            assert learner._samples_seen == initial_samples + 1

    def test_predict_and_learn_without_outcome(self, mock_config, sample_features):
        """Test predict_and_learn works without outcome."""
        with patch('src.learning.online_learner.get_config', return_value=mock_config):
            from src.learning.online_learner import OnlineLearner

            learner = OnlineLearner()
            initial_samples = learner._samples_seen

            prediction, probability = learner.predict_and_learn(sample_features, y_true=None)

            assert isinstance(prediction, int)
            assert isinstance(probability, float)
            assert learner._samples_seen == initial_samples  # Not incremented

    def test_predict_and_learn_with_prediction_id(self, mock_config, sample_features):
        """Test predict_and_learn records history with prediction_id."""
        with patch('src.learning.online_learner.get_config', return_value=mock_config):
            from src.learning.online_learner import OnlineLearner

            learner = OnlineLearner()

            learner.predict_and_learn(
                sample_features,
                y_true=None,
                prediction_id="test-123"
            )

            assert "test-123" in learner._history.pending_outcomes


@pytest.mark.skipif(not RIVER_AVAILABLE, reason="river package not installed")
class TestOnlineLearnerMetrics:
    """Tests for metrics tracking."""

    def test_metrics_update(self, mock_config, sample_features):
        """Test metrics are updated after learning."""
        with patch('src.learning.online_learner.get_config', return_value=mock_config):
            from src.learning.online_learner import OnlineLearner

            learner = OnlineLearner()

            # Make prediction and learn
            prediction, _ = learner.predict_and_learn(sample_features, y_true=1)

            # Metrics should be updated
            assert learner._metrics.total_predictions == 1
            if prediction == 1:
                assert learner._metrics.true_positives == 1
            else:
                assert learner._metrics.false_negatives == 1

    def test_metrics_to_dict(self, mock_config, sample_features):
        """Test metrics can be exported to dictionary."""
        with patch('src.learning.online_learner.get_config', return_value=mock_config):
            from src.learning.online_learner import OnlineLearner

            learner = OnlineLearner()

            # Learn some samples
            for i in range(10):
                learner.predict_and_learn(sample_features, y_true=i % 2)

            metrics_dict = learner._metrics.to_dict()

            assert 'accuracy' in metrics_dict
            assert 'precision' in metrics_dict
            assert 'recall' in metrics_dict
            assert 'f1' in metrics_dict
            assert 'total_predictions' in metrics_dict

    def test_recent_accuracy(self, mock_config, sample_features):
        """Test recent accuracy calculation."""
        with patch('src.learning.online_learner.get_config', return_value=mock_config):
            from src.learning.online_learner import OnlineLearner

            learner = OnlineLearner()

            # Add some correct predictions
            for _ in range(5):
                learner._metrics.recent_results.append(1)  # Correct
            for _ in range(5):
                learner._metrics.recent_results.append(0)  # Incorrect

            accuracy = learner._metrics.recent_accuracy

            assert accuracy == 0.5


@pytest.mark.skipif(not RIVER_AVAILABLE, reason="river package not installed")
class TestOnlineLearnerDriftDetection:
    """Tests for drift detection."""

    def test_drift_state_initialization(self, mock_config):
        """Test drift state initializes correctly."""
        with patch('src.learning.online_learner.get_config', return_value=mock_config):
            from src.learning.online_learner import OnlineLearner

            learner = OnlineLearner()

            assert learner._drift.drift_count == 0
            assert learner._drift.in_drift is False

    def test_drift_detected_property(self, mock_config):
        """Test drift_detected property."""
        with patch('src.learning.online_learner.get_config', return_value=mock_config):
            from src.learning.online_learner import OnlineLearner

            learner = OnlineLearner()

            assert learner.drift_detected is False

            learner._drift.in_drift = True

            assert learner.drift_detected is True

    def test_drift_callback_registration(self, mock_config):
        """Test drift callback can be registered."""
        with patch('src.learning.online_learner.get_config', return_value=mock_config):
            from src.learning.online_learner import OnlineLearner

            learner = OnlineLearner()
            callback = MagicMock()

            learner.register_drift_callback(callback)

            assert callback in learner._on_drift_callbacks


@pytest.mark.skipif(not RIVER_AVAILABLE, reason="river package not installed")
class TestOnlineLearnerHistory:
    """Tests for prediction history."""

    def test_learn_from_outcome_success(self, mock_config, sample_features):
        """Test learning from delayed outcome."""
        with patch('src.learning.online_learner.get_config', return_value=mock_config):
            from src.learning.online_learner import OnlineLearner

            learner = OnlineLearner()

            # Make prediction with ID
            learner.predict_and_learn(
                sample_features,
                y_true=None,
                prediction_id="test-456"
            )

            # Later, learn from outcome
            result = learner.learn_from_outcome("test-456", y_true=1)

            assert result is True
            assert "test-456" not in learner._history.pending_outcomes

    def test_learn_from_outcome_unknown_id(self, mock_config):
        """Test learning from unknown prediction ID returns False."""
        with patch('src.learning.online_learner.get_config', return_value=mock_config):
            from src.learning.online_learner import OnlineLearner

            learner = OnlineLearner()

            result = learner.learn_from_outcome("unknown-id", y_true=1)

            assert result is False

    def test_clear_stale_history(self, mock_config, sample_features):
        """Test clearing stale pending outcomes."""
        with patch('src.learning.online_learner.get_config', return_value=mock_config):
            from src.learning.online_learner import OnlineLearner
            from datetime import timedelta

            learner = OnlineLearner()

            # Add a prediction
            learner.predict_and_learn(
                sample_features,
                y_true=None,
                prediction_id="stale-123"
            )

            # Make it old by manipulating timestamp
            learner._history.pending_outcomes["stale-123"]['timestamp'] = (
                datetime.now(timezone.utc) - timedelta(hours=2)
            )

            # Clear stale (with 1 hour max age)
            cleared = learner.clear_stale_history(max_age_seconds=3600)

            assert cleared == 1
            assert "stale-123" not in learner._history.pending_outcomes


@pytest.mark.skipif(not RIVER_AVAILABLE, reason="river package not installed")
class TestOnlineLearnerState:
    """Tests for state export and reset."""

    def test_get_state(self, mock_config, sample_features):
        """Test getting complete learner state."""
        with patch('src.learning.online_learner.get_config', return_value=mock_config):
            from src.learning.online_learner import OnlineLearner

            learner = OnlineLearner()

            # Learn some samples
            for i in range(5):
                learner.learn_one(sample_features, i % 2)

            state = learner.get_state()

            assert 'samples_seen' in state
            assert state['samples_seen'] == 5
            assert 'is_warm' in state
            assert 'metrics' in state
            assert 'drift' in state

    def test_get_summary(self, mock_config, sample_features):
        """Test getting learner summary for dashboard."""
        with patch('src.learning.online_learner.get_config', return_value=mock_config):
            from src.learning.online_learner import OnlineLearner

            learner = OnlineLearner()

            summary = learner.get_summary()

            assert 'samples' in summary
            assert 'warm' in summary
            assert 'accuracy' in summary
            assert 'drift_count' in summary

    def test_reset_model(self, mock_config, sample_features):
        """Test resetting the model."""
        with patch('src.learning.online_learner.get_config', return_value=mock_config):
            from src.learning.online_learner import OnlineLearner

            learner = OnlineLearner()

            # Learn some samples
            for i in range(10):
                learner.learn_one(sample_features, i % 2)

            # Reset
            learner.reset_model()

            assert learner._samples_seen == 0
            assert learner._is_warm is False
            assert learner._feature_names == []

    def test_reset_model_preserve_metrics(self, mock_config, sample_features):
        """Test resetting model while preserving metrics."""
        with patch('src.learning.online_learner.get_config', return_value=mock_config):
            from src.learning.online_learner import OnlineLearner

            learner = OnlineLearner()

            # Learn and track metrics
            for i in range(10):
                learner.predict_and_learn(sample_features, y_true=i % 2)

            old_total = learner._metrics.total_predictions

            # Reset preserving metrics
            learner.reset_model(preserve_metrics=True)

            assert learner._samples_seen == 0
            assert learner._metrics.total_predictions == old_total


@pytest.mark.skipif(not RIVER_AVAILABLE, reason="river package not installed")
class TestOnlineLearnerProperties:
    """Tests for learner properties."""

    def test_is_warm_property(self, mock_config):
        """Test is_warm property."""
        with patch('src.learning.online_learner.get_config', return_value=mock_config):
            from src.learning.online_learner import OnlineLearner

            learner = OnlineLearner()

            assert learner.is_warm is False

            learner._samples_seen = learner._warmup_samples

            assert learner.is_warm is True

    def test_samples_seen_property(self, mock_config, sample_features):
        """Test samples_seen property."""
        with patch('src.learning.online_learner.get_config', return_value=mock_config):
            from src.learning.online_learner import OnlineLearner

            learner = OnlineLearner()

            assert learner.samples_seen == 0

            learner.learn_one(sample_features, 1)

            assert learner.samples_seen == 1

    def test_last_prediction_property(self, mock_config, sample_features):
        """Test last_prediction property."""
        with patch('src.learning.online_learner.get_config', return_value=mock_config):
            from src.learning.online_learner import OnlineLearner

            learner = OnlineLearner()

            assert learner.last_prediction is None

            learner.predict_one(sample_features)

            assert learner.last_prediction in (0, 1)


@pytest.mark.skipif(not RIVER_AVAILABLE, reason="river package not installed")
class TestMultiContractLearner:
    """Tests for MultiContractLearner."""

    def test_multi_contract_initialization(self, mock_config):
        """Test MultiContractLearner initialization."""
        with patch('src.learning.online_learner.get_config', return_value=mock_config):
            from src.learning.online_learner import MultiContractLearner

            multi = MultiContractLearner()

            assert multi.contract_count == 0
            assert len(multi._learners) == 0

    def test_get_learner_creates_new(self, mock_config):
        """Test get_learner creates new learner for unknown contract."""
        with patch('src.learning.online_learner.get_config', return_value=mock_config):
            from src.learning.online_learner import MultiContractLearner, OnlineLearner

            multi = MultiContractLearner()

            learner = multi.get_learner("ES")

            assert isinstance(learner, OnlineLearner)
            assert multi.contract_count == 1

    def test_get_learner_returns_existing(self, mock_config):
        """Test get_learner returns existing learner."""
        with patch('src.learning.online_learner.get_config', return_value=mock_config):
            from src.learning.online_learner import MultiContractLearner

            multi = MultiContractLearner()

            learner1 = multi.get_learner("ES")
            learner2 = multi.get_learner("ES")

            assert learner1 is learner2
            assert multi.contract_count == 1

    def test_predict_and_learn_per_contract(self, mock_config, sample_features):
        """Test predict and learn for specific contract."""
        with patch('src.learning.online_learner.get_config', return_value=mock_config):
            from src.learning.online_learner import MultiContractLearner

            multi = MultiContractLearner()

            prediction, prob = multi.predict_and_learn("ES", sample_features, y_true=1)

            assert isinstance(prediction, int)
            assert isinstance(prob, float)
            assert multi.get_learner("ES").samples_seen == 1

    def test_global_accuracy(self, mock_config, sample_features):
        """Test global accuracy across all contracts."""
        with patch('src.learning.online_learner.get_config', return_value=mock_config):
            from src.learning.online_learner import MultiContractLearner

            multi = MultiContractLearner()

            # Learn on multiple contracts
            multi.predict_and_learn("ES", sample_features, y_true=1)
            multi.predict_and_learn("NQ", sample_features, y_true=0)

            accuracy = multi.global_accuracy

            assert isinstance(accuracy, float)
            assert 0.0 <= accuracy <= 1.0

    def test_get_all_summaries(self, mock_config, sample_features):
        """Test getting summaries for all contracts."""
        with patch('src.learning.online_learner.get_config', return_value=mock_config):
            from src.learning.online_learner import MultiContractLearner

            multi = MultiContractLearner()

            multi.predict_and_learn("ES", sample_features, y_true=1)
            multi.predict_and_learn("NQ", sample_features, y_true=0)

            summaries = multi.get_all_summaries()

            assert "ES" in summaries
            assert "NQ" in summaries

    def test_reset_all(self, mock_config, sample_features):
        """Test resetting all learners."""
        with patch('src.learning.online_learner.get_config', return_value=mock_config):
            from src.learning.online_learner import MultiContractLearner

            multi = MultiContractLearner()

            # Learn on multiple contracts
            for _ in range(5):
                multi.predict_and_learn("ES", sample_features, y_true=1)
                multi.predict_and_learn("NQ", sample_features, y_true=0)

            multi.reset_all()

            assert multi.get_learner("ES").samples_seen == 0
            assert multi.get_learner("NQ").samples_seen == 0


@pytest.mark.skipif(not RIVER_AVAILABLE, reason="river package not installed")
class TestLearningMetrics:
    """Tests for LearningMetrics dataclass."""

    def test_metrics_initialization(self, mock_config):
        """Test LearningMetrics initialization."""
        from src.learning.online_learner import LearningMetrics

        metrics = LearningMetrics()

        assert metrics.total_predictions == 0
        assert metrics.total_correct == 0
        assert metrics.true_positives == 0
        assert metrics.false_positives == 0

    def test_metrics_update(self):
        """Test metrics update method."""
        from src.learning.online_learner import LearningMetrics

        metrics = LearningMetrics()

        # True positive
        metrics.update(y_true=1, y_pred=1)
        assert metrics.true_positives == 1
        assert metrics.total_correct == 1

        # False positive
        metrics.update(y_true=0, y_pred=1)
        assert metrics.false_positives == 1

        # True negative
        metrics.update(y_true=0, y_pred=0)
        assert metrics.true_negatives == 1

        # False negative
        metrics.update(y_true=1, y_pred=0)
        assert metrics.false_negatives == 1

    def test_metrics_reset(self):
        """Test metrics reset method."""
        from src.learning.online_learner import LearningMetrics

        metrics = LearningMetrics()
        metrics.update(y_true=1, y_pred=1)

        metrics.reset()

        assert metrics.total_predictions == 0
        assert metrics.true_positives == 0
