"""
Integration tests for Prediction Pipeline.

Tests the interaction between MarketData, FeatureEngine, and Predictor:
- Feature extraction from market data
- Prediction generation pipeline
- Confidence scoring integration
- Learning from outcomes
- Multi-contract prediction handling

These tests use mocked components to verify integration without
requiring a live TWS/Gateway connection.
"""

import pytest
import asyncio
import sys
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

sys.path.insert(0, '.')

# Check for river availability and proper numpy/pandas setup
# These tests require a working scientific Python environment
DEPENDENCIES_AVAILABLE = False
SKIP_REASON = "Required dependencies not available"

try:
    import river
    import numpy as np
    import pandas as pd
    # Try to import our modules to verify environment is properly set up
    from src.data.feature_engine import FeatureEngine
    from src.trading.predictor import Predictor
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    SKIP_REASON = f"Import error: {e}"
except Exception as e:
    SKIP_REASON = f"Setup error: {e}"

# Skip all tests if dependencies not available
pytestmark = pytest.mark.skipif(
    not DEPENDENCIES_AVAILABLE,
    reason=SKIP_REASON
)


@pytest.fixture
def mock_config():
    """Mock configuration for integration tests."""
    config = MagicMock()
    config.IB_HOST = '127.0.0.1'
    config.IB_PORT = 7497
    config.IB_CLIENT_ID = 1
    config.DATA_STALENESS_THRESHOLD_SECONDS = 30
    return config


@pytest.fixture
def sample_tick_data():
    """Create sample tick data for testing."""
    from src.data.market_data import TickData

    return TickData(
        contract_id=123456,
        symbol='ES',
        timestamp=datetime.now(timezone.utc),
        bid=5000.25,
        bid_size=100,
        ask=5000.50,
        ask_size=150,
        last=5000.25,
        last_size=10,
        volume=50000,
        high=5010.00,
        low=4990.00,
        open=4995.00,
        close=5002.00,
    )


@pytest.fixture
def sample_depth_data():
    """Create sample depth data for testing."""
    from src.data.market_data import DepthData, OrderBookLevel

    return DepthData(
        contract_id=123456,
        symbol='ES',
        timestamp=datetime.now(timezone.utc),
        bids=[
            OrderBookLevel(price=5000.25, size=100),
            OrderBookLevel(price=5000.00, size=200),
            OrderBookLevel(price=4999.75, size=150),
        ],
        asks=[
            OrderBookLevel(price=5000.50, size=120),
            OrderBookLevel(price=5000.75, size=180),
            OrderBookLevel(price=5001.00, size=250),
        ],
    )


@pytest.fixture
def stale_tick_data():
    """Create stale tick data for testing."""
    from src.data.market_data import TickData

    return TickData(
        contract_id=123456,
        symbol='ES',
        timestamp=datetime.now(timezone.utc) - timedelta(seconds=60),
        bid=5000.25,
        bid_size=100,
        ask=5000.50,
        ask_size=150,
        last=5000.25,
        last_size=10,
        volume=50000,
    )


class TestFeatureExtractionPipeline:
    """Tests for feature extraction from market data."""

    def test_feature_extraction_from_tick_data(self, mock_config, sample_tick_data):
        """Test feature extraction from L1 tick data."""
        with patch('src.data.feature_engine.get_config', return_value=mock_config):
            from src.data.feature_engine import FeatureEngine

            engine = FeatureEngine()

            # Extract features (symbol is derived from tick_data)
            features = engine.extract_features(
                tick_data=sample_tick_data,
                depth_data=None,
            )

            # Verify features extracted
            assert isinstance(features, dict)
            assert len(features) > 0

            # Check for expected L1 features
            assert 'spread' in features
            assert 'quote_imbalance' in features

    def test_feature_extraction_from_depth_data(self, mock_config, sample_depth_data):
        """Test feature extraction from L2 depth data."""
        with patch('src.data.feature_engine.get_config', return_value=mock_config):
            from src.data.feature_engine import FeatureEngine

            engine = FeatureEngine()

            # Extract features (symbol is derived from depth_data)
            features = engine.extract_features(
                tick_data=None,
                depth_data=sample_depth_data,
            )

            # Verify features extracted
            assert isinstance(features, dict)
            assert len(features) > 0

            # Check for expected L2 features
            assert 'book_imbalance' in features
            assert 'liquidity_score' in features

    def test_combined_feature_extraction(self, mock_config, sample_tick_data, sample_depth_data):
        """Test combined L1 + L2 feature extraction."""
        with patch('src.data.feature_engine.get_config', return_value=mock_config):
            from src.data.feature_engine import FeatureEngine

            engine = FeatureEngine()

            # Extract combined features
            features = engine.extract_features(
                tick_data=sample_tick_data,
                depth_data=sample_depth_data,
            )

            # Should have both L1 and L2 features
            assert 'spread' in features
            assert 'book_imbalance' in features

    def test_feature_dict_format_for_river(self, mock_config, sample_tick_data, sample_depth_data):
        """Test that features are dict format required by River."""
        with patch('src.data.feature_engine.get_config', return_value=mock_config):
            from src.data.feature_engine import FeatureEngine

            engine = FeatureEngine()

            features = engine.extract_features(
                tick_data=sample_tick_data,
                depth_data=sample_depth_data,
            )

            # CRITICAL: Must be dict for River compatibility
            assert isinstance(features, dict)

            # All values should be numeric (not numpy arrays)
            for key, value in features.items():
                assert isinstance(value, (int, float)), f"Feature {key} is not numeric: {type(value)}"


class TestPredictionGeneration:
    """Tests for prediction generation pipeline."""

    def test_prediction_from_market_data(self, mock_config, sample_tick_data, sample_depth_data):
        """Test full prediction pipeline from market data."""
        with patch('src.trading.predictor.get_config', return_value=mock_config):
            with patch('src.data.feature_engine.get_config', return_value=mock_config):
                with patch('src.learning.online_learner.get_config', return_value=mock_config):
                    from src.trading.predictor import Predictor, PredictionSignal

                    predictor = Predictor(use_multi_contract=True)

                    # Generate prediction
                    result = predictor.predict(
                        tick_data=sample_tick_data,
                        depth_data=sample_depth_data,
                    )

                    # Verify prediction result
                    assert result is not None
                    assert result.prediction_id is not None
                    assert result.contract_id == sample_tick_data.contract_id
                    assert result.symbol == sample_tick_data.symbol
                    assert result.signal in [PredictionSignal.BUY, PredictionSignal.SELL, PredictionSignal.NEUTRAL]
                    assert 0.0 <= result.confidence <= 1.0

    def test_prediction_result_structure(self, mock_config, sample_tick_data, sample_depth_data):
        """Test prediction result contains all required fields."""
        with patch('src.trading.predictor.get_config', return_value=mock_config):
            with patch('src.data.feature_engine.get_config', return_value=mock_config):
                with patch('src.learning.online_learner.get_config', return_value=mock_config):
                    from src.trading.predictor import Predictor

                    predictor = Predictor()

                    result = predictor.predict(
                        tick_data=sample_tick_data,
                        depth_data=sample_depth_data,
                    )

                    # Verify all required fields
                    assert hasattr(result, 'prediction_id')
                    assert hasattr(result, 'signal')
                    assert hasattr(result, 'direction')
                    assert hasattr(result, 'probability')
                    assert hasattr(result, 'confidence')
                    assert hasattr(result, 'model_warm')
                    assert hasattr(result, 'feature_count')
                    assert hasattr(result, 'is_valid')

    def test_prediction_rejection_for_stale_data(self, mock_config, stale_tick_data, sample_depth_data):
        """Test prediction is rejected for stale data."""
        with patch('src.trading.predictor.get_config', return_value=mock_config):
            with patch('src.data.feature_engine.get_config', return_value=mock_config):
                with patch('src.learning.online_learner.get_config', return_value=mock_config):
                    from src.trading.predictor import Predictor

                    predictor = Predictor()

                    result = predictor.predict(
                        tick_data=stale_tick_data,
                        depth_data=sample_depth_data,
                    )

                    # Prediction should be invalid due to stale data
                    assert result.is_valid is False
                    assert 'stale' in result.rejection_reason.lower()

    def test_prediction_rejection_for_no_data(self, mock_config):
        """Test prediction is rejected when no data provided."""
        with patch('src.trading.predictor.get_config', return_value=mock_config):
            with patch('src.data.feature_engine.get_config', return_value=mock_config):
                with patch('src.learning.online_learner.get_config', return_value=mock_config):
                    from src.trading.predictor import Predictor

                    predictor = Predictor()

                    result = predictor.predict(
                        tick_data=None,
                        depth_data=None,
                    )

                    assert result.is_valid is False
                    assert 'no_market_data' in result.rejection_reason


class TestPredictionConfidenceIntegration:
    """Tests for confidence scoring in prediction pipeline."""

    def test_confidence_calculation(self, mock_config, sample_tick_data, sample_depth_data):
        """Test confidence is calculated correctly."""
        with patch('src.trading.predictor.get_config', return_value=mock_config):
            with patch('src.data.feature_engine.get_config', return_value=mock_config):
                with patch('src.learning.online_learner.get_config', return_value=mock_config):
                    from src.trading.predictor import Predictor

                    predictor = Predictor()

                    result = predictor.predict(
                        tick_data=sample_tick_data,
                        depth_data=sample_depth_data,
                    )

                    # Confidence should be between 0 and 1
                    assert 0.0 <= result.confidence <= 1.0

    def test_low_confidence_neutral_signal(self, mock_config, sample_tick_data, sample_depth_data):
        """Test low confidence leads to neutral signal."""
        with patch('src.trading.predictor.get_config', return_value=mock_config):
            with patch('src.data.feature_engine.get_config', return_value=mock_config):
                with patch('src.learning.online_learner.get_config', return_value=mock_config):
                    from src.trading.predictor import Predictor, PredictionSignal

                    # Create predictor with high minimum confidence
                    predictor = Predictor(min_confidence=0.99)

                    result = predictor.predict(
                        tick_data=sample_tick_data,
                        depth_data=sample_depth_data,
                    )

                    # With high min_confidence, should be NEUTRAL
                    assert result.signal == PredictionSignal.NEUTRAL


class TestPredictionLearning:
    """Tests for learning from outcomes."""

    def test_predict_and_learn(self, mock_config, sample_tick_data, sample_depth_data):
        """Test combined prediction and learning."""
        with patch('src.trading.predictor.get_config', return_value=mock_config):
            with patch('src.data.feature_engine.get_config', return_value=mock_config):
                with patch('src.learning.online_learner.get_config', return_value=mock_config):
                    from src.trading.predictor import Predictor

                    predictor = Predictor()

                    # Predict and learn with known outcome
                    result = predictor.predict_and_learn(
                        tick_data=sample_tick_data,
                        depth_data=sample_depth_data,
                        y_true=1,  # Outcome: up
                    )

                    # Prediction should be made
                    assert result is not None

    def test_delayed_learning(self, mock_config, sample_tick_data, sample_depth_data):
        """Test learning from delayed outcome."""
        with patch('src.trading.predictor.get_config', return_value=mock_config):
            with patch('src.data.feature_engine.get_config', return_value=mock_config):
                with patch('src.learning.online_learner.get_config', return_value=mock_config):
                    from src.trading.predictor import Predictor

                    predictor = Predictor()

                    # Make prediction
                    result = predictor.predict(
                        tick_data=sample_tick_data,
                        depth_data=sample_depth_data,
                    )

                    # Store prediction ID
                    prediction_id = result.prediction_id

                    # Later, learn from outcome
                    success = predictor.learn_from_outcome(prediction_id, y_true=1)

                    assert success is True

    def test_pending_predictions_tracking(self, mock_config, sample_tick_data, sample_depth_data):
        """Test tracking of pending predictions."""
        with patch('src.trading.predictor.get_config', return_value=mock_config):
            with patch('src.data.feature_engine.get_config', return_value=mock_config):
                with patch('src.learning.online_learner.get_config', return_value=mock_config):
                    from src.trading.predictor import Predictor

                    predictor = Predictor()

                    # Make prediction
                    result = predictor.predict(
                        tick_data=sample_tick_data,
                        depth_data=sample_depth_data,
                    )

                    # Should have pending prediction
                    assert predictor.get_pending_count() == 1

                    # Learn from outcome
                    predictor.learn_from_outcome(result.prediction_id, y_true=1)

                    # Pending should be cleared
                    assert predictor.get_pending_count() == 0


class TestMultiContractPrediction:
    """Tests for multi-contract prediction handling."""

    def test_multi_contract_predictor(self, mock_config, sample_tick_data, sample_depth_data):
        """Test predictor handles multiple contracts."""
        with patch('src.trading.predictor.get_config', return_value=mock_config):
            with patch('src.data.feature_engine.get_config', return_value=mock_config):
                with patch('src.learning.online_learner.get_config', return_value=mock_config):
                    from src.trading.predictor import Predictor
                    from src.data.market_data import TickData, DepthData, OrderBookLevel

                    predictor = Predictor(use_multi_contract=True)

                    # Predict for ES
                    result_es = predictor.predict(
                        tick_data=sample_tick_data,
                        depth_data=sample_depth_data,
                    )

                    # Create NQ data
                    nq_tick = TickData(
                        contract_id=789012,
                        symbol='NQ',
                        timestamp=datetime.now(timezone.utc),
                        bid=18000.25,
                        bid_size=50,
                        ask=18000.50,
                        ask_size=75,
                        last=18000.25,
                        last_size=5,
                        volume=25000,
                    )

                    nq_depth = DepthData(
                        contract_id=789012,
                        symbol='NQ',
                        timestamp=datetime.now(timezone.utc),
                        bids=[OrderBookLevel(price=18000.25, size=50)],
                        asks=[OrderBookLevel(price=18000.50, size=75)],
                    )

                    # Predict for NQ
                    result_nq = predictor.predict(
                        tick_data=nq_tick,
                        depth_data=nq_depth,
                    )

                    # Both should have valid predictions
                    assert result_es.symbol == 'ES'
                    assert result_nq.symbol == 'NQ'

    def test_per_contract_statistics(self, mock_config, sample_tick_data, sample_depth_data):
        """Test per-contract statistics are tracked."""
        with patch('src.trading.predictor.get_config', return_value=mock_config):
            with patch('src.data.feature_engine.get_config', return_value=mock_config):
                with patch('src.learning.online_learner.get_config', return_value=mock_config):
                    from src.trading.predictor import Predictor

                    predictor = Predictor(use_multi_contract=True)

                    # Make multiple predictions
                    for _ in range(5):
                        predictor.predict(
                            tick_data=sample_tick_data,
                            depth_data=sample_depth_data,
                        )

                    # Get contract summary
                    summary = predictor.get_contract_summary(sample_tick_data.contract_id)

                    assert summary is not None
                    assert summary['stats']['total_predictions'] == 5


class TestPredictionStatistics:
    """Tests for prediction statistics tracking."""

    def test_prediction_stats_recording(self, mock_config, sample_tick_data, sample_depth_data):
        """Test prediction statistics are recorded."""
        with patch('src.trading.predictor.get_config', return_value=mock_config):
            with patch('src.data.feature_engine.get_config', return_value=mock_config):
                with patch('src.learning.online_learner.get_config', return_value=mock_config):
                    from src.trading.predictor import Predictor

                    predictor = Predictor()

                    # Make predictions
                    for _ in range(10):
                        predictor.predict(
                            tick_data=sample_tick_data,
                            depth_data=sample_depth_data,
                        )

                    # Check stats
                    assert predictor.stats.total_predictions == 10
                    assert predictor.stats.valid_predictions <= 10

    def test_predictor_summary(self, mock_config, sample_tick_data, sample_depth_data):
        """Test predictor summary for dashboard."""
        with patch('src.trading.predictor.get_config', return_value=mock_config):
            with patch('src.data.feature_engine.get_config', return_value=mock_config):
                with patch('src.learning.online_learner.get_config', return_value=mock_config):
                    from src.trading.predictor import Predictor

                    predictor = Predictor()

                    # Make predictions
                    predictor.predict(
                        tick_data=sample_tick_data,
                        depth_data=sample_depth_data,
                    )

                    # Get summary
                    summary = predictor.get_summary()

                    # Check required fields
                    assert 'total_predictions' in summary
                    assert 'valid_predictions' in summary
                    assert 'average_confidence' in summary
                    assert 'samples_learned' in summary

    def test_predictor_reset(self, mock_config, sample_tick_data, sample_depth_data):
        """Test predictor reset functionality."""
        with patch('src.trading.predictor.get_config', return_value=mock_config):
            with patch('src.data.feature_engine.get_config', return_value=mock_config):
                with patch('src.learning.online_learner.get_config', return_value=mock_config):
                    from src.trading.predictor import Predictor

                    predictor = Predictor()

                    # Make predictions
                    for _ in range(5):
                        predictor.predict(
                            tick_data=sample_tick_data,
                            depth_data=sample_depth_data,
                        )

                    # Reset
                    predictor.reset()

                    # Stats should be cleared
                    assert predictor.stats.total_predictions == 0


class TestPredictionCallbacks:
    """Tests for prediction callback handling."""

    def test_prediction_callback_invocation(self, mock_config, sample_tick_data, sample_depth_data):
        """Test prediction callbacks are invoked."""
        with patch('src.trading.predictor.get_config', return_value=mock_config):
            with patch('src.data.feature_engine.get_config', return_value=mock_config):
                with patch('src.learning.online_learner.get_config', return_value=mock_config):
                    from src.trading.predictor import Predictor

                    predictor = Predictor()

                    # Track callback invocations
                    callback_results = []

                    def on_prediction(result):
                        callback_results.append(result)

                    predictor.register_prediction_callback(on_prediction)

                    # Make prediction
                    predictor.predict(
                        tick_data=sample_tick_data,
                        depth_data=sample_depth_data,
                    )

                    # Callback should be invoked
                    assert len(callback_results) == 1

    def test_drift_callback_registration(self, mock_config, sample_tick_data, sample_depth_data):
        """Test drift callback can be registered."""
        with patch('src.trading.predictor.get_config', return_value=mock_config):
            with patch('src.data.feature_engine.get_config', return_value=mock_config):
                with patch('src.learning.online_learner.get_config', return_value=mock_config):
                    from src.trading.predictor import Predictor

                    predictor = Predictor()

                    # Track drift callbacks
                    drift_events = []

                    def on_drift(contract_id, drift_info):
                        drift_events.append((contract_id, drift_info))

                    predictor.register_drift_callback(on_drift)

                    # Callback should be registered
                    assert len(predictor._on_drift_callbacks) == 1
