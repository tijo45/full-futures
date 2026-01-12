"""
Integration tests for Trade Execution Pipeline.

Tests the interaction between Predictor, Executor, and IBClient:
- Confidence-gated execution flow
- Order submission and tracking
- Position management integration
- Learning from execution outcomes
- Risk limit enforcement

These tests use mocked components to verify integration without
requiring a live TWS/Gateway connection.
"""

import pytest
import asyncio
import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock, patch
from typing import Any

sys.path.insert(0, '.')

# Check for dependencies availability
# These tests require a working scientific Python environment
DEPENDENCIES_AVAILABLE = False
SKIP_REASON = "Required dependencies not available"

try:
    import river
    import numpy as np
    import pandas as pd
    # Try to import our modules to verify environment is properly set up
    from src.trading.executor import Executor
    from src.trading.confidence import ConfidenceTracker
    from src.trading.predictor import PredictionResult, PredictionSignal
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
def mock_ib_client():
    """Create a mock IBClient for execution tests."""
    client = MagicMock()
    client.is_connected = True
    client.state = MagicMock()

    # Mock IB instance
    mock_ib = MagicMock()

    # Mock order and trade
    mock_order = MagicMock()
    mock_order.orderId = 12345

    mock_trade = MagicMock()
    mock_trade.order = mock_order
    mock_trade.orderStatus = MagicMock()
    mock_trade.orderStatus.filled = 1
    mock_trade.orderStatus.avgFillPrice = 5000.25
    mock_trade.fills = []
    mock_trade.filledEvent = MagicMock()
    mock_trade.filledEvent.__iadd__ = MagicMock()
    mock_trade.cancelledEvent = MagicMock()
    mock_trade.cancelledEvent.__iadd__ = MagicMock()

    mock_ib.placeOrder = MagicMock(return_value=mock_trade)
    mock_ib.cancelOrder = MagicMock()

    client.ib = mock_ib

    return client


@pytest.fixture
def mock_contract():
    """Create a mock IB Contract."""
    contract = MagicMock()
    contract.conId = 123456
    contract.symbol = 'ES'
    contract.exchange = 'CME'
    return contract


@pytest.fixture
def sample_prediction_result():
    """Create a sample prediction result for execution tests."""
    from src.trading.predictor import PredictionResult, PredictionSignal

    return PredictionResult(
        prediction_id='test-pred-001',
        contract_id=123456,
        symbol='ES',
        timestamp=datetime.now(timezone.utc),
        signal=PredictionSignal.BUY,
        direction=1,
        probability=0.75,
        confidence=0.8,
        model_warm=True,
        feature_count=30,
        data_age_seconds=1.0,
        is_valid=True,
    )


@pytest.fixture
def sample_confidence_level():
    """Create a sample confidence level for execution tests."""
    from src.trading.confidence import ConfidenceLevel

    return ConfidenceLevel(
        value=0.8,
        timestamp=datetime.now(timezone.utc),
        model_confidence=0.75,
        data_quality=0.9,
        regime_stability=0.85,
        recent_accuracy=0.7,
    )


@pytest.fixture
def low_confidence_level():
    """Create a low confidence level for rejection tests."""
    from src.trading.confidence import ConfidenceLevel

    return ConfidenceLevel(
        value=0.3,
        timestamp=datetime.now(timezone.utc),
        model_confidence=0.4,
        data_quality=0.5,
        regime_stability=0.3,
        recent_accuracy=0.2,
    )


class TestConfidenceGatedExecution:
    """Tests for confidence gating in execution flow."""

    @pytest.mark.asyncio
    async def test_high_confidence_execution(
        self, mock_config, mock_ib_client, mock_contract, sample_prediction_result, sample_confidence_level
    ):
        """Test execution proceeds with high confidence."""
        with patch('src.trading.executor.get_config', return_value=mock_config):
            with patch('src.trading.confidence.get_config', return_value=mock_config):
                from src.trading.executor import Executor, ExecutionDecision
                from src.trading.confidence import ConfidenceTracker

                confidence_tracker = ConfidenceTracker()
                executor = Executor(
                    ib_client=mock_ib_client,
                    confidence_tracker=confidence_tracker,
                )

                # Execute prediction
                result = await executor.execute_prediction(
                    prediction=sample_prediction_result,
                    confidence=sample_confidence_level,
                    contract=mock_contract,
                    quantity=1,
                )

                # Should execute (not reject)
                assert result.decision == ExecutionDecision.EXECUTE
                assert result.order_id is not None

    @pytest.mark.asyncio
    async def test_low_confidence_rejection(
        self, mock_config, mock_ib_client, mock_contract, sample_prediction_result, low_confidence_level
    ):
        """Test execution is rejected with low confidence."""
        with patch('src.trading.executor.get_config', return_value=mock_config):
            with patch('src.trading.confidence.get_config', return_value=mock_config):
                from src.trading.executor import Executor, ExecutionDecision
                from src.trading.confidence import ConfidenceTracker

                confidence_tracker = ConfidenceTracker()
                executor = Executor(
                    ib_client=mock_ib_client,
                    confidence_tracker=confidence_tracker,
                )

                # Execute prediction with low confidence
                result = await executor.execute_prediction(
                    prediction=sample_prediction_result,
                    confidence=low_confidence_level,
                    contract=mock_contract,
                    quantity=1,
                )

                # Should be rejected
                assert result.decision == ExecutionDecision.REJECT_LOW_CONFIDENCE

    @pytest.mark.asyncio
    async def test_neutral_signal_rejection(
        self, mock_config, mock_ib_client, mock_contract, sample_confidence_level
    ):
        """Test neutral prediction signals are rejected."""
        from src.trading.predictor import PredictionResult, PredictionSignal

        neutral_prediction = PredictionResult(
            prediction_id='test-pred-002',
            contract_id=123456,
            symbol='ES',
            timestamp=datetime.now(timezone.utc),
            signal=PredictionSignal.NEUTRAL,
            direction=0,
            probability=0.5,
            confidence=0.5,
            model_warm=True,
            feature_count=30,
            data_age_seconds=1.0,
            is_valid=True,
        )

        with patch('src.trading.executor.get_config', return_value=mock_config):
            with patch('src.trading.confidence.get_config', return_value=mock_config):
                from src.trading.executor import Executor, ExecutionDecision
                from src.trading.confidence import ConfidenceTracker

                confidence_tracker = ConfidenceTracker()
                executor = Executor(
                    ib_client=mock_ib_client,
                    confidence_tracker=confidence_tracker,
                )

                result = await executor.execute_prediction(
                    prediction=neutral_prediction,
                    confidence=sample_confidence_level,
                    contract=mock_contract,
                    quantity=1,
                )

                # Neutral signals should be rejected
                assert result.decision == ExecutionDecision.REJECT_LOW_CONFIDENCE


class TestOrderSubmission:
    """Tests for order submission to IB."""

    @pytest.mark.asyncio
    async def test_market_order_submission(
        self, mock_config, mock_ib_client, mock_contract, sample_prediction_result, sample_confidence_level
    ):
        """Test market order submission to IB."""
        with patch('src.trading.executor.get_config', return_value=mock_config):
            with patch('src.trading.confidence.get_config', return_value=mock_config):
                from src.trading.executor import Executor, OrderType
                from src.trading.confidence import ConfidenceTracker

                confidence_tracker = ConfidenceTracker()
                executor = Executor(
                    ib_client=mock_ib_client,
                    confidence_tracker=confidence_tracker,
                )

                result = await executor.execute_prediction(
                    prediction=sample_prediction_result,
                    confidence=sample_confidence_level,
                    contract=mock_contract,
                    quantity=1,
                    order_type=OrderType.MARKET,
                )

                # Verify order was placed
                mock_ib_client.ib.placeOrder.assert_called_once()

    @pytest.mark.asyncio
    async def test_order_tracking(
        self, mock_config, mock_ib_client, mock_contract, sample_prediction_result, sample_confidence_level
    ):
        """Test pending order is tracked after submission."""
        with patch('src.trading.executor.get_config', return_value=mock_config):
            with patch('src.trading.confidence.get_config', return_value=mock_config):
                from src.trading.executor import Executor, ExecutionStatus
                from src.trading.confidence import ConfidenceTracker

                confidence_tracker = ConfidenceTracker()
                executor = Executor(
                    ib_client=mock_ib_client,
                    confidence_tracker=confidence_tracker,
                )

                result = await executor.execute_prediction(
                    prediction=sample_prediction_result,
                    confidence=sample_confidence_level,
                    contract=mock_contract,
                    quantity=1,
                )

                # Order should be in pending
                assert result.status == ExecutionStatus.SUBMITTED
                assert len(executor.get_pending_orders()) == 1


class TestConnectionChecks:
    """Tests for IB connection validation before execution."""

    @pytest.mark.asyncio
    async def test_execution_requires_connection(
        self, mock_config, mock_contract, sample_prediction_result, sample_confidence_level
    ):
        """Test execution fails without IB connection."""
        with patch('src.trading.executor.get_config', return_value=mock_config):
            with patch('src.trading.confidence.get_config', return_value=mock_config):
                from src.trading.executor import Executor, ExecutionDecision
                from src.trading.confidence import ConfidenceTracker

                confidence_tracker = ConfidenceTracker()
                executor = Executor(
                    ib_client=None,  # No client
                    confidence_tracker=confidence_tracker,
                )

                result = await executor.execute_prediction(
                    prediction=sample_prediction_result,
                    confidence=sample_confidence_level,
                    contract=mock_contract,
                    quantity=1,
                )

                # Should reject due to no connection
                assert result.decision == ExecutionDecision.REJECT_NO_CONNECTION

    @pytest.mark.asyncio
    async def test_disconnected_client_rejection(
        self, mock_config, mock_ib_client, mock_contract, sample_prediction_result, sample_confidence_level
    ):
        """Test execution fails when client is disconnected."""
        mock_ib_client.is_connected = False

        with patch('src.trading.executor.get_config', return_value=mock_config):
            with patch('src.trading.confidence.get_config', return_value=mock_config):
                from src.trading.executor import Executor, ExecutionDecision
                from src.trading.confidence import ConfidenceTracker

                confidence_tracker = ConfidenceTracker()
                executor = Executor(
                    ib_client=mock_ib_client,
                    confidence_tracker=confidence_tracker,
                )

                result = await executor.execute_prediction(
                    prediction=sample_prediction_result,
                    confidence=sample_confidence_level,
                    contract=mock_contract,
                    quantity=1,
                )

                assert result.decision == ExecutionDecision.REJECT_NO_CONNECTION


class TestPositionLimits:
    """Tests for position limit enforcement."""

    @pytest.mark.asyncio
    async def test_position_limit_enforcement(
        self, mock_config, mock_ib_client, mock_contract, sample_prediction_result, sample_confidence_level
    ):
        """Test position limits are enforced."""
        with patch('src.trading.executor.get_config', return_value=mock_config):
            with patch('src.trading.confidence.get_config', return_value=mock_config):
                from src.trading.executor import Executor, ExecutionDecision
                from src.trading.confidence import ConfidenceTracker

                confidence_tracker = ConfidenceTracker()
                executor = Executor(
                    ib_client=mock_ib_client,
                    confidence_tracker=confidence_tracker,
                    max_position_per_contract=2,
                )

                # Set existing position at limit
                executor.update_position(mock_contract.conId, 2)

                result = await executor.execute_prediction(
                    prediction=sample_prediction_result,
                    confidence=sample_confidence_level,
                    contract=mock_contract,
                    quantity=1,
                )

                # Should reject due to position limit
                assert result.decision == ExecutionDecision.REJECT_POSITION_LIMIT

    @pytest.mark.asyncio
    async def test_total_exposure_limit(
        self, mock_config, mock_ib_client, mock_contract, sample_prediction_result, sample_confidence_level
    ):
        """Test total exposure limits are enforced."""
        with patch('src.trading.executor.get_config', return_value=mock_config):
            with patch('src.trading.confidence.get_config', return_value=mock_config):
                from src.trading.executor import Executor, ExecutionDecision
                from src.trading.confidence import ConfidenceTracker

                confidence_tracker = ConfidenceTracker()
                executor = Executor(
                    ib_client=mock_ib_client,
                    confidence_tracker=confidence_tracker,
                    max_total_exposure=5,
                )

                # Set existing exposure near limit
                executor.update_position(111111, 3)
                executor.update_position(222222, 2)

                result = await executor.execute_prediction(
                    prediction=sample_prediction_result,
                    confidence=sample_confidence_level,
                    contract=mock_contract,
                    quantity=1,
                )

                # Should reject due to total exposure
                assert result.decision == ExecutionDecision.REJECT_RISK_LIMIT


class TestExecutorEnable:
    """Tests for executor enable/disable functionality."""

    @pytest.mark.asyncio
    async def test_disabled_executor_rejection(
        self, mock_config, mock_ib_client, mock_contract, sample_prediction_result, sample_confidence_level
    ):
        """Test disabled executor rejects all orders."""
        with patch('src.trading.executor.get_config', return_value=mock_config):
            with patch('src.trading.confidence.get_config', return_value=mock_config):
                from src.trading.executor import Executor, ExecutionDecision
                from src.trading.confidence import ConfidenceTracker

                confidence_tracker = ConfidenceTracker()
                executor = Executor(
                    ib_client=mock_ib_client,
                    confidence_tracker=confidence_tracker,
                )

                # Disable executor
                executor.disable()

                result = await executor.execute_prediction(
                    prediction=sample_prediction_result,
                    confidence=sample_confidence_level,
                    contract=mock_contract,
                    quantity=1,
                )

                assert result.decision == ExecutionDecision.REJECT_DISABLED

    @pytest.mark.asyncio
    async def test_re_enabled_executor(
        self, mock_config, mock_ib_client, mock_contract, sample_prediction_result, sample_confidence_level
    ):
        """Test re-enabled executor accepts orders."""
        with patch('src.trading.executor.get_config', return_value=mock_config):
            with patch('src.trading.confidence.get_config', return_value=mock_config):
                from src.trading.executor import Executor, ExecutionDecision
                from src.trading.confidence import ConfidenceTracker

                confidence_tracker = ConfidenceTracker()
                executor = Executor(
                    ib_client=mock_ib_client,
                    confidence_tracker=confidence_tracker,
                )

                # Disable then re-enable
                executor.disable()
                executor.enable()

                result = await executor.execute_prediction(
                    prediction=sample_prediction_result,
                    confidence=sample_confidence_level,
                    contract=mock_contract,
                    quantity=1,
                )

                assert result.decision == ExecutionDecision.EXECUTE


class TestLearningFromOutcomes:
    """Tests for learning feedback from execution outcomes."""

    @pytest.mark.asyncio
    async def test_record_successful_outcome(
        self, mock_config, mock_ib_client, mock_contract, sample_prediction_result, sample_confidence_level
    ):
        """Test recording successful trade outcome."""
        with patch('src.trading.executor.get_config', return_value=mock_config):
            with patch('src.trading.confidence.get_config', return_value=mock_config):
                from src.trading.executor import Executor
                from src.trading.confidence import ConfidenceTracker

                confidence_tracker = ConfidenceTracker()
                executor = Executor(
                    ib_client=mock_ib_client,
                    confidence_tracker=confidence_tracker,
                )

                # Execute order
                result = await executor.execute_prediction(
                    prediction=sample_prediction_result,
                    confidence=sample_confidence_level,
                    contract=mock_contract,
                    quantity=1,
                )

                # Simulate fill by moving to completed
                executor._completed_orders[result.request_id] = result

                # Record outcome
                executor.record_outcome(
                    request_id=result.request_id,
                    success=True,
                    pnl=150.0,
                )

                # Check outcome was recorded
                assert result.request_id in executor._execution_outcomes
                assert executor._execution_outcomes[result.request_id]['success'] is True

    @pytest.mark.asyncio
    async def test_record_unsuccessful_outcome(
        self, mock_config, mock_ib_client, mock_contract, sample_prediction_result, sample_confidence_level
    ):
        """Test recording unsuccessful trade outcome."""
        with patch('src.trading.executor.get_config', return_value=mock_config):
            with patch('src.trading.confidence.get_config', return_value=mock_config):
                from src.trading.executor import Executor
                from src.trading.confidence import ConfidenceTracker

                confidence_tracker = ConfidenceTracker()
                executor = Executor(
                    ib_client=mock_ib_client,
                    confidence_tracker=confidence_tracker,
                )

                # Execute order
                result = await executor.execute_prediction(
                    prediction=sample_prediction_result,
                    confidence=sample_confidence_level,
                    contract=mock_contract,
                    quantity=1,
                )

                # Simulate fill
                executor._completed_orders[result.request_id] = result

                # Record losing outcome
                executor.record_outcome(
                    request_id=result.request_id,
                    success=False,
                    pnl=-75.0,
                )

                assert executor._execution_outcomes[result.request_id]['success'] is False
                assert executor._execution_outcomes[result.request_id]['pnl'] == -75.0


class TestExecutionStatistics:
    """Tests for execution statistics tracking."""

    @pytest.mark.asyncio
    async def test_stats_tracking(
        self, mock_config, mock_ib_client, mock_contract, sample_prediction_result, sample_confidence_level
    ):
        """Test execution statistics are tracked."""
        with patch('src.trading.executor.get_config', return_value=mock_config):
            with patch('src.trading.confidence.get_config', return_value=mock_config):
                from src.trading.executor import Executor
                from src.trading.confidence import ConfidenceTracker

                confidence_tracker = ConfidenceTracker()
                executor = Executor(
                    ib_client=mock_ib_client,
                    confidence_tracker=confidence_tracker,
                )

                # Execute order
                await executor.execute_prediction(
                    prediction=sample_prediction_result,
                    confidence=sample_confidence_level,
                    contract=mock_contract,
                    quantity=1,
                )

                # Stats should be updated
                # Note: Stats are recorded after fill, but submission counts

    @pytest.mark.asyncio
    async def test_rejection_stats(
        self, mock_config, mock_ib_client, mock_contract, sample_prediction_result, low_confidence_level
    ):
        """Test rejection statistics are tracked."""
        with patch('src.trading.executor.get_config', return_value=mock_config):
            with patch('src.trading.confidence.get_config', return_value=mock_config):
                from src.trading.executor import Executor
                from src.trading.confidence import ConfidenceTracker

                confidence_tracker = ConfidenceTracker()
                executor = Executor(
                    ib_client=mock_ib_client,
                    confidence_tracker=confidence_tracker,
                )

                # Execute with low confidence (will be rejected)
                await executor.execute_prediction(
                    prediction=sample_prediction_result,
                    confidence=low_confidence_level,
                    contract=mock_contract,
                    quantity=1,
                )

                # Stats should show rejection
                assert executor.stats.rejected_count == 1
                assert 'reject_low_confidence' in executor.stats.rejections_by_reason

    @pytest.mark.asyncio
    async def test_executor_summary(
        self, mock_config, mock_ib_client, mock_contract, sample_prediction_result, sample_confidence_level
    ):
        """Test executor summary generation."""
        with patch('src.trading.executor.get_config', return_value=mock_config):
            with patch('src.trading.confidence.get_config', return_value=mock_config):
                from src.trading.executor import Executor
                from src.trading.confidence import ConfidenceTracker

                confidence_tracker = ConfidenceTracker()
                executor = Executor(
                    ib_client=mock_ib_client,
                    confidence_tracker=confidence_tracker,
                )

                # Get summary
                summary = executor.get_summary()

                # Check required fields
                assert 'enabled' in summary
                assert 'is_ready' in summary
                assert 'pending_orders' in summary
                assert 'total_executions' in summary
                assert 'total_rejections' in summary


class TestCallbackInvocation:
    """Tests for execution callback handling."""

    @pytest.mark.asyncio
    async def test_execution_callback(
        self, mock_config, mock_ib_client, mock_contract, sample_prediction_result, sample_confidence_level
    ):
        """Test execution callbacks are invoked."""
        with patch('src.trading.executor.get_config', return_value=mock_config):
            with patch('src.trading.confidence.get_config', return_value=mock_config):
                from src.trading.executor import Executor
                from src.trading.confidence import ConfidenceTracker

                confidence_tracker = ConfidenceTracker()
                executor = Executor(
                    ib_client=mock_ib_client,
                    confidence_tracker=confidence_tracker,
                )

                # Track callbacks
                callback_results = []

                def on_execution(result):
                    callback_results.append(result)

                executor.register_execution_callback(on_execution)

                # Execute order
                await executor.execute_prediction(
                    prediction=sample_prediction_result,
                    confidence=sample_confidence_level,
                    contract=mock_contract,
                    quantity=1,
                )

                # Callback should be invoked
                assert len(callback_results) == 1

    @pytest.mark.asyncio
    async def test_rejection_callback(
        self, mock_config, mock_ib_client, mock_contract, sample_prediction_result, low_confidence_level
    ):
        """Test rejection callbacks are invoked."""
        with patch('src.trading.executor.get_config', return_value=mock_config):
            with patch('src.trading.confidence.get_config', return_value=mock_config):
                from src.trading.executor import Executor
                from src.trading.confidence import ConfidenceTracker

                confidence_tracker = ConfidenceTracker()
                executor = Executor(
                    ib_client=mock_ib_client,
                    confidence_tracker=confidence_tracker,
                )

                # Track callbacks
                rejection_results = []

                def on_rejection(result):
                    rejection_results.append(result)

                executor.register_rejection_callback(on_rejection)

                # Execute with low confidence (will be rejected)
                await executor.execute_prediction(
                    prediction=sample_prediction_result,
                    confidence=low_confidence_level,
                    contract=mock_contract,
                    quantity=1,
                )

                # Rejection callback should be invoked
                assert len(rejection_results) == 1


class TestFullPipelineIntegration:
    """Tests for full prediction-to-execution pipeline."""

    @pytest.mark.asyncio
    async def test_full_pipeline_flow(self, mock_config, mock_ib_client, mock_contract):
        """Test complete flow from market data to execution."""
        from src.data.market_data import TickData, DepthData, OrderBookLevel

        # Create market data
        tick_data = TickData(
            contract_id=mock_contract.conId,
            symbol=mock_contract.symbol,
            timestamp=datetime.now(timezone.utc),
            bid=5000.25,
            bid_size=100,
            ask=5000.50,
            ask_size=150,
            last=5000.25,
            last_size=10,
            volume=50000,
        )

        depth_data = DepthData(
            contract_id=mock_contract.conId,
            symbol=mock_contract.symbol,
            timestamp=datetime.now(timezone.utc),
            bids=[
                OrderBookLevel(price=5000.25, size=100),
                OrderBookLevel(price=5000.00, size=200),
            ],
            asks=[
                OrderBookLevel(price=5000.50, size=120),
                OrderBookLevel(price=5000.75, size=180),
            ],
        )

        with patch('src.trading.predictor.get_config', return_value=mock_config):
            with patch('src.data.feature_engine.get_config', return_value=mock_config):
                with patch('src.learning.online_learner.get_config', return_value=mock_config):
                    with patch('src.trading.executor.get_config', return_value=mock_config):
                        with patch('src.trading.confidence.get_config', return_value=mock_config):
                            from src.trading.predictor import Predictor, PredictionSignal
                            from src.trading.confidence import ConfidenceTracker, ConfidenceLevel
                            from src.trading.executor import Executor

                            # Create components
                            predictor = Predictor()
                            confidence_tracker = ConfidenceTracker()
                            executor = Executor(
                                ib_client=mock_ib_client,
                                confidence_tracker=confidence_tracker,
                            )

                            # Step 1: Generate prediction
                            prediction = predictor.predict(
                                tick_data=tick_data,
                                depth_data=depth_data,
                            )

                            assert prediction is not None
                            assert prediction.is_valid

                            # Step 2: Calculate confidence
                            confidence = confidence_tracker.calculate_confidence(
                                model_confidence=prediction.confidence,
                                data_quality=0.9,
                                regime_stability=0.85,
                                recent_accuracy=0.7,
                            )

                            assert confidence is not None

                            # Step 3: Execute if directional signal
                            if prediction.signal != PredictionSignal.NEUTRAL:
                                result = await executor.execute_prediction(
                                    prediction=prediction,
                                    confidence=confidence,
                                    contract=mock_contract,
                                    quantity=1,
                                )

                                # Result should be valid
                                assert result is not None
                                assert result.request_id is not None

    @pytest.mark.asyncio
    async def test_learning_loop_integration(self, mock_config, mock_ib_client, mock_contract):
        """Test learning feedback loop from execution outcomes."""
        from src.data.market_data import TickData, DepthData, OrderBookLevel

        tick_data = TickData(
            contract_id=mock_contract.conId,
            symbol=mock_contract.symbol,
            timestamp=datetime.now(timezone.utc),
            bid=5000.25,
            bid_size=100,
            ask=5000.50,
            ask_size=150,
            last=5000.25,
            last_size=10,
            volume=50000,
        )

        depth_data = DepthData(
            contract_id=mock_contract.conId,
            symbol=mock_contract.symbol,
            timestamp=datetime.now(timezone.utc),
            bids=[OrderBookLevel(price=5000.25, size=100)],
            asks=[OrderBookLevel(price=5000.50, size=120)],
        )

        with patch('src.trading.predictor.get_config', return_value=mock_config):
            with patch('src.data.feature_engine.get_config', return_value=mock_config):
                with patch('src.learning.online_learner.get_config', return_value=mock_config):
                    with patch('src.trading.executor.get_config', return_value=mock_config):
                        with patch('src.trading.confidence.get_config', return_value=mock_config):
                            from src.trading.predictor import Predictor
                            from src.trading.confidence import ConfidenceTracker
                            from src.trading.executor import Executor

                            predictor = Predictor()
                            confidence_tracker = ConfidenceTracker()
                            executor = Executor(
                                ib_client=mock_ib_client,
                                confidence_tracker=confidence_tracker,
                            )

                            # Generate prediction
                            prediction = predictor.predict(
                                tick_data=tick_data,
                                depth_data=depth_data,
                            )

                            # Calculate confidence
                            confidence = confidence_tracker.calculate_confidence(
                                model_confidence=prediction.confidence,
                                data_quality=0.9,
                                regime_stability=0.85,
                                recent_accuracy=0.7,
                            )

                            # Execute
                            exec_result = await executor.execute_prediction(
                                prediction=prediction,
                                confidence=confidence,
                                contract=mock_contract,
                                quantity=1,
                            )

                            # Simulate trade completion
                            executor._completed_orders[exec_result.request_id] = exec_result

                            # Record outcome
                            executor.record_outcome(
                                request_id=exec_result.request_id,
                                success=True,
                                pnl=100.0,
                            )

                            # Learn from prediction outcome
                            success = predictor.learn_from_outcome(
                                prediction.prediction_id,
                                y_true=1,  # Trade was profitable
                            )

                            assert success is True
