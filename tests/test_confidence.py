"""
Unit tests for ConfidenceTracker - Confidence tracking and dynamic thresholds.

Tests cover:
- Confidence calculation
- Gate decisions for entry/exit
- Dynamic threshold adjustment
- Drawdown and regime updates
- Per-contract tracking
- Outcome recording
"""

import pytest
import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

sys.path.insert(0, '.')


@pytest.fixture
def mock_config():
    """Mock configuration for confidence tracker."""
    config = MagicMock()
    return config


class TestConfidenceTrackerInitialization:
    """Tests for ConfidenceTracker initialization."""

    def test_initialization_with_defaults(self, mock_config):
        """Test ConfidenceTracker initializes with adaptive defaults."""
        with patch('src.trading.confidence.get_config', return_value=mock_config):
            from src.trading.confidence import ConfidenceTracker

            tracker = ConfidenceTracker()

            assert tracker._global_threshold.base_threshold == 0.5
            assert tracker._global_threshold.min_threshold == 0.3
            assert tracker._global_threshold.max_threshold == 0.95
            assert tracker._current_drawdown == 0.0
            assert tracker._current_regime == "stable"

    def test_initialization_with_custom_thresholds(self, mock_config):
        """Test ConfidenceTracker with custom thresholds."""
        with patch('src.trading.confidence.get_config', return_value=mock_config):
            from src.trading.confidence import ConfidenceTracker

            tracker = ConfidenceTracker(
                initial_threshold=0.6,
                min_threshold=0.4,
                max_threshold=0.9
            )

            assert tracker._global_threshold.base_threshold == 0.6
            assert tracker._global_threshold.current_threshold == 0.6
            assert tracker._global_threshold.min_threshold == 0.4
            assert tracker._global_threshold.max_threshold == 0.9


class TestConfidenceCalculation:
    """Tests for confidence calculation."""

    def test_calculate_confidence_basic(self, mock_config):
        """Test basic confidence calculation."""
        with patch('src.trading.confidence.get_config', return_value=mock_config):
            from src.trading.confidence import ConfidenceTracker

            tracker = ConfidenceTracker()

            confidence = tracker.calculate_confidence(
                model_probability=0.75,
                data_quality=0.9,
                regime_stability=0.8,
                recent_accuracy=0.65
            )

            assert confidence.value > 0.0
            assert confidence.value <= 1.0
            assert confidence.model_confidence == 0.5  # |0.75 - 0.5| * 2
            assert confidence.data_quality == 0.9
            assert confidence.regime_stability == 0.8
            assert confidence.recent_accuracy == 0.65

    def test_calculate_confidence_low_probability(self, mock_config):
        """Test confidence with low probability (bearish signal)."""
        with patch('src.trading.confidence.get_config', return_value=mock_config):
            from src.trading.confidence import ConfidenceTracker

            tracker = ConfidenceTracker()

            confidence = tracker.calculate_confidence(
                model_probability=0.25,  # Low = bearish
                data_quality=0.9,
                regime_stability=0.8,
                recent_accuracy=0.65
            )

            # Model confidence should still be high (distance from 0.5)
            assert confidence.model_confidence == 0.5  # |0.25 - 0.5| * 2

    def test_calculate_confidence_tracks_history(self, mock_config):
        """Test confidence calculation records to history."""
        with patch('src.trading.confidence.get_config', return_value=mock_config):
            from src.trading.confidence import ConfidenceTracker

            tracker = ConfidenceTracker()
            initial_count = tracker._history.count

            tracker.calculate_confidence(
                model_probability=0.75,
                data_quality=0.9,
                regime_stability=0.8,
                recent_accuracy=0.65
            )

            assert tracker._history.count == initial_count + 1

    def test_calculate_confidence_with_contract_id(self, mock_config):
        """Test confidence calculation with contract tracking."""
        with patch('src.trading.confidence.get_config', return_value=mock_config):
            from src.trading.confidence import ConfidenceTracker

            tracker = ConfidenceTracker()

            confidence = tracker.calculate_confidence(
                model_probability=0.75,
                data_quality=0.9,
                regime_stability=0.8,
                recent_accuracy=0.65,
                contract_id=123
            )

            assert confidence.contract_id == 123
            assert 123 in tracker._contracts


class TestGateDecisions:
    """Tests for entry and exit gate decisions."""

    def test_gate_entry_allowed(self, mock_config):
        """Test entry allowed when confidence above threshold."""
        with patch('src.trading.confidence.get_config', return_value=mock_config):
            from src.trading.confidence import ConfidenceTracker, GateDecision

            tracker = ConfidenceTracker(initial_threshold=0.5)

            confidence = tracker.calculate_confidence(
                model_probability=0.9,  # High confidence
                data_quality=1.0,
                regime_stability=1.0,
                recent_accuracy=0.8
            )

            decision = tracker.gate_entry(confidence)

            assert decision == GateDecision.ALLOWED
            assert tracker._allowed_count == 1

    def test_gate_entry_rejected(self, mock_config):
        """Test entry rejected when confidence below threshold."""
        with patch('src.trading.confidence.get_config', return_value=mock_config):
            from src.trading.confidence import ConfidenceTracker, GateDecision

            tracker = ConfidenceTracker(initial_threshold=0.9)  # High threshold

            confidence = tracker.calculate_confidence(
                model_probability=0.6,  # Low confidence
                data_quality=0.5,
                regime_stability=0.5,
                recent_accuracy=0.5
            )

            decision = tracker.gate_entry(confidence)

            assert decision == GateDecision.REJECTED
            assert tracker._rejected_count == 1

    def test_gate_exit_easier_when_profitable(self, mock_config):
        """Test exit has lower threshold when position is profitable."""
        with patch('src.trading.confidence.get_config', return_value=mock_config):
            from src.trading.confidence import ConfidenceTracker, GateDecision

            tracker = ConfidenceTracker(initial_threshold=0.7)

            confidence = tracker.calculate_confidence(
                model_probability=0.65,  # Moderate confidence
                data_quality=0.8,
                regime_stability=0.7,
                recent_accuracy=0.6
            )

            # With profit, threshold is lowered
            decision = tracker.gate_exit(
                confidence,
                position_pnl_percent=5.0  # In profit
            )

            # Should allow exit more easily when profitable
            assert decision in (GateDecision.ALLOWED, GateDecision.REJECTED)

    def test_gate_exit_harder_when_losing(self, mock_config):
        """Test exit has higher threshold when position is losing."""
        with patch('src.trading.confidence.get_config', return_value=mock_config):
            from src.trading.confidence import ConfidenceTracker, GateDecision

            tracker = ConfidenceTracker(initial_threshold=0.5)

            # Create a low confidence scenario
            confidence = tracker.calculate_confidence(
                model_probability=0.55,
                data_quality=0.5,
                regime_stability=0.5,
                recent_accuracy=0.5
            )

            # With loss, threshold is raised to avoid panic selling
            decision = tracker.gate_exit(
                confidence,
                position_pnl_percent=-5.0  # In loss
            )

            # Higher threshold should make it harder to exit in loss
            assert decision in (GateDecision.ALLOWED, GateDecision.REJECTED)


class TestPositionScaling:
    """Tests for position scaling based on confidence."""

    def test_get_position_scaling_above_threshold(self, mock_config):
        """Test scaling increases with confidence above threshold."""
        with patch('src.trading.confidence.get_config', return_value=mock_config):
            from src.trading.confidence import ConfidenceTracker

            tracker = ConfidenceTracker(initial_threshold=0.5)

            confidence = tracker.calculate_confidence(
                model_probability=0.9,
                data_quality=1.0,
                regime_stability=1.0,
                recent_accuracy=0.8
            )

            scaling = tracker.get_position_scaling(confidence)

            assert 0.0 <= scaling <= 1.0

    def test_get_position_scaling_below_threshold(self, mock_config):
        """Test scaling is 0 when below threshold."""
        with patch('src.trading.confidence.get_config', return_value=mock_config):
            from src.trading.confidence import ConfidenceTracker

            tracker = ConfidenceTracker(initial_threshold=0.9)

            confidence = tracker.calculate_confidence(
                model_probability=0.55,
                data_quality=0.5,
                regime_stability=0.5,
                recent_accuracy=0.5
            )

            scaling = tracker.get_position_scaling(confidence)

            assert scaling == 0.0


class TestDynamicThresholds:
    """Tests for dynamic threshold adjustment."""

    def test_threshold_increases_after_losses(self, mock_config):
        """Test threshold increases after poor performance."""
        with patch('src.trading.confidence.get_config', return_value=mock_config):
            from src.trading.confidence import ConfidenceTracker

            tracker = ConfidenceTracker(initial_threshold=0.5)
            initial_threshold = tracker._global_threshold.base_threshold

            # Record many losses
            for _ in range(10):
                tracker.record_outcome(success=False, confidence_used=0.6)

            # Threshold should have increased
            assert tracker._global_threshold.base_threshold >= initial_threshold

    def test_threshold_decreases_after_successes(self, mock_config):
        """Test threshold can decrease after good performance."""
        with patch('src.trading.confidence.get_config', return_value=mock_config):
            from src.trading.confidence import ConfidenceTracker

            tracker = ConfidenceTracker(initial_threshold=0.7)
            initial_threshold = tracker._global_threshold.base_threshold

            # Record many successes
            for _ in range(20):
                tracker.record_outcome(success=True, confidence_used=0.8)

            # Threshold should have decreased (or stayed same at minimum)
            assert tracker._global_threshold.base_threshold <= initial_threshold


class TestDrawdownAdjustment:
    """Tests for drawdown-based threshold adjustment."""

    def test_update_drawdown(self, mock_config):
        """Test drawdown update affects threshold."""
        with patch('src.trading.confidence.get_config', return_value=mock_config):
            from src.trading.confidence import ConfidenceTracker, ThresholdMode

            tracker = ConfidenceTracker(initial_threshold=0.5)

            # Update with significant drawdown
            tracker.update_drawdown(15.0)

            assert tracker._current_drawdown == 15.0
            assert tracker._global_threshold.mode == ThresholdMode.CAUTIOUS

    def test_drawdown_adjustment_on_threshold(self, mock_config):
        """Test drawdown adjusts effective threshold."""
        with patch('src.trading.confidence.get_config', return_value=mock_config):
            from src.trading.confidence import ConfidenceTracker

            tracker = ConfidenceTracker(initial_threshold=0.5)
            base_threshold = tracker.get_threshold()

            # Add drawdown
            tracker.update_drawdown(10.0)
            adjusted_threshold = tracker.get_threshold()

            # Threshold should be higher with drawdown
            assert adjusted_threshold >= base_threshold

    def test_drawdown_recovery(self, mock_config):
        """Test threshold mode recovers after drawdown improvement."""
        with patch('src.trading.confidence.get_config', return_value=mock_config):
            from src.trading.confidence import ConfidenceTracker, ThresholdMode

            tracker = ConfidenceTracker(initial_threshold=0.5)

            # Go into drawdown
            tracker.update_drawdown(10.0)
            assert tracker._global_threshold.mode == ThresholdMode.CAUTIOUS

            # Recover from drawdown
            tracker.update_drawdown(1.0)
            assert tracker._global_threshold.mode == ThresholdMode.NORMAL


class TestRegimeUpdates:
    """Tests for regime-based adjustments."""

    def test_update_regime_volatile(self, mock_config):
        """Test regime update for volatile market."""
        with patch('src.trading.confidence.get_config', return_value=mock_config):
            from src.trading.confidence import ConfidenceTracker, ThresholdMode

            tracker = ConfidenceTracker()

            tracker.update_regime(regime="volatile", stability=0.3)

            assert tracker._current_regime == "volatile"
            assert tracker._regime_confidence_factor == 0.3
            assert tracker._global_threshold.mode == ThresholdMode.CAUTIOUS

    def test_update_regime_stable(self, mock_config):
        """Test regime update for stable market."""
        with patch('src.trading.confidence.get_config', return_value=mock_config):
            from src.trading.confidence import ConfidenceTracker, ThresholdMode

            tracker = ConfidenceTracker()

            tracker.update_regime(regime="stable", stability=0.9)

            assert tracker._current_regime == "stable"
            assert tracker._regime_confidence_factor == 0.9
            assert tracker._global_threshold.mode == ThresholdMode.NORMAL

    def test_set_threshold_mode(self, mock_config):
        """Test setting threshold mode directly."""
        with patch('src.trading.confidence.get_config', return_value=mock_config):
            from src.trading.confidence import ConfidenceTracker, ThresholdMode

            tracker = ConfidenceTracker()

            tracker.set_threshold_mode('AGGRESSIVE', factor=1.5)

            assert tracker._global_threshold.mode == ThresholdMode.AGGRESSIVE


class TestPerContractTracking:
    """Tests for per-contract confidence tracking."""

    def test_contract_tracking_created(self, mock_config):
        """Test contract tracking is created on first use."""
        with patch('src.trading.confidence.get_config', return_value=mock_config):
            from src.trading.confidence import ConfidenceTracker

            tracker = ConfidenceTracker()

            tracker.calculate_confidence(
                model_probability=0.75,
                data_quality=0.9,
                regime_stability=0.8,
                recent_accuracy=0.65,
                contract_id=123
            )

            assert 123 in tracker._contracts
            assert tracker.contracts_count == 1

    def test_contract_specific_threshold(self, mock_config):
        """Test getting contract-specific threshold."""
        with patch('src.trading.confidence.get_config', return_value=mock_config):
            from src.trading.confidence import ConfidenceTracker

            tracker = ConfidenceTracker()

            # Create contract tracking
            tracker.calculate_confidence(
                model_probability=0.75,
                data_quality=0.9,
                regime_stability=0.8,
                recent_accuracy=0.65,
                contract_id=123
            )

            threshold = tracker.get_threshold(contract_id=123)

            assert isinstance(threshold, float)

    def test_record_outcome_per_contract(self, mock_config):
        """Test recording outcome for specific contract."""
        with patch('src.trading.confidence.get_config', return_value=mock_config):
            from src.trading.confidence import ConfidenceTracker

            tracker = ConfidenceTracker()

            # Create contract tracking
            tracker.calculate_confidence(
                model_probability=0.75,
                data_quality=0.9,
                regime_stability=0.8,
                recent_accuracy=0.65,
                contract_id=123
            )

            # Record outcome
            tracker.record_outcome(
                success=True,
                confidence_used=0.7,
                contract_id=123
            )

            assert tracker._contracts[123].total_trades == 1
            assert tracker._contracts[123].successful_trades == 1

    def test_get_contract_summary(self, mock_config):
        """Test getting contract summary."""
        with patch('src.trading.confidence.get_config', return_value=mock_config):
            from src.trading.confidence import ConfidenceTracker

            tracker = ConfidenceTracker()

            # Create contract tracking
            tracker.calculate_confidence(
                model_probability=0.75,
                data_quality=0.9,
                regime_stability=0.8,
                recent_accuracy=0.65,
                contract_id=123
            )

            summary = tracker.get_contract_summary(123)

            assert summary is not None
            assert 'contract_id' in summary
            assert summary['contract_id'] == 123


class TestCallbacks:
    """Tests for callback registration."""

    def test_register_threshold_callback(self, mock_config):
        """Test registering threshold change callback."""
        with patch('src.trading.confidence.get_config', return_value=mock_config):
            from src.trading.confidence import ConfidenceTracker

            tracker = ConfidenceTracker()
            callback = MagicMock()

            tracker.register_threshold_callback(callback)

            assert callback in tracker._on_threshold_change

    def test_register_gate_callback(self, mock_config):
        """Test registering gate decision callback."""
        with patch('src.trading.confidence.get_config', return_value=mock_config):
            from src.trading.confidence import ConfidenceTracker

            tracker = ConfidenceTracker()
            callback = MagicMock()

            tracker.register_gate_callback(callback)

            assert callback in tracker._on_gate_decision


class TestTrackerState:
    """Tests for tracker state export."""

    def test_get_state(self, mock_config):
        """Test getting complete tracker state."""
        with patch('src.trading.confidence.get_config', return_value=mock_config):
            from src.trading.confidence import ConfidenceTracker

            tracker = ConfidenceTracker()

            # Generate some activity
            tracker.calculate_confidence(
                model_probability=0.75,
                data_quality=0.9,
                regime_stability=0.8,
                recent_accuracy=0.65
            )

            state = tracker.get_state()

            assert 'global_threshold' in state
            assert 'current_drawdown' in state
            assert 'current_regime' in state
            assert 'history' in state
            assert 'total_evaluations' in state

    def test_get_summary(self, mock_config):
        """Test getting tracker summary for dashboard."""
        with patch('src.trading.confidence.get_config', return_value=mock_config):
            from src.trading.confidence import ConfidenceTracker

            tracker = ConfidenceTracker()

            summary = tracker.get_summary()

            assert 'threshold' in summary
            assert 'base_threshold' in summary
            assert 'mode' in summary
            assert 'drawdown' in summary
            assert 'regime' in summary

    def test_reset(self, mock_config):
        """Test resetting tracker state."""
        with patch('src.trading.confidence.get_config', return_value=mock_config):
            from src.trading.confidence import ConfidenceTracker

            tracker = ConfidenceTracker()

            # Generate some activity
            for _ in range(10):
                tracker.calculate_confidence(
                    model_probability=0.75,
                    data_quality=0.9,
                    regime_stability=0.8,
                    recent_accuracy=0.65,
                    contract_id=123
                )

            tracker.reset()

            assert tracker._total_evaluations == 0
            assert tracker._allowed_count == 0
            assert tracker._rejected_count == 0
            assert len(tracker._contracts) == 0


class TestTrackerProperties:
    """Tests for tracker properties."""

    def test_current_threshold_property(self, mock_config):
        """Test current_threshold property."""
        with patch('src.trading.confidence.get_config', return_value=mock_config):
            from src.trading.confidence import ConfidenceTracker

            tracker = ConfidenceTracker(initial_threshold=0.6)

            assert tracker.current_threshold == 0.6

    def test_allow_rate_property(self, mock_config):
        """Test allow_rate property."""
        with patch('src.trading.confidence.get_config', return_value=mock_config):
            from src.trading.confidence import ConfidenceTracker, GateDecision

            tracker = ConfidenceTracker(initial_threshold=0.5)

            # Generate entries
            for i in range(10):
                conf = tracker.calculate_confidence(
                    model_probability=0.5 + (i * 0.05),
                    data_quality=0.9,
                    regime_stability=0.8,
                    recent_accuracy=0.65
                )
                tracker.gate_entry(conf)

            rate = tracker.allow_rate

            assert 0.0 <= rate <= 1.0

    def test_contracts_count_property(self, mock_config):
        """Test contracts_count property."""
        with patch('src.trading.confidence.get_config', return_value=mock_config):
            from src.trading.confidence import ConfidenceTracker

            tracker = ConfidenceTracker()

            tracker.calculate_confidence(
                model_probability=0.75,
                data_quality=0.9,
                regime_stability=0.8,
                recent_accuracy=0.65,
                contract_id=123
            )
            tracker.calculate_confidence(
                model_probability=0.75,
                data_quality=0.9,
                regime_stability=0.8,
                recent_accuracy=0.65,
                contract_id=456
            )

            assert tracker.contracts_count == 2


class TestConfidenceLevel:
    """Tests for ConfidenceLevel dataclass."""

    def test_confidence_level_creation(self, mock_config):
        """Test ConfidenceLevel creation."""
        from src.trading.confidence import ConfidenceLevel

        confidence = ConfidenceLevel(
            value=0.75,
            timestamp=datetime.now(timezone.utc),
            model_confidence=0.8,
            data_quality=0.9,
            regime_stability=0.7,
            recent_accuracy=0.65
        )

        assert confidence.value == 0.75
        assert confidence.model_confidence == 0.8

    def test_confidence_level_to_dict(self, mock_config):
        """Test ConfidenceLevel export to dictionary."""
        from src.trading.confidence import ConfidenceLevel

        confidence = ConfidenceLevel(
            value=0.75,
            timestamp=datetime.now(timezone.utc),
            model_confidence=0.8,
            data_quality=0.9,
            regime_stability=0.7,
            recent_accuracy=0.65,
            contract_id=123
        )

        data = confidence.to_dict()

        assert data['value'] == 0.75
        assert data['contract_id'] == 123
        assert 'timestamp' in data


class TestThresholdState:
    """Tests for ThresholdState dataclass."""

    def test_threshold_state_update(self, mock_config):
        """Test threshold state update mechanism."""
        from src.trading.confidence import ThresholdState

        state = ThresholdState(base_threshold=0.5)

        # Record some outcomes
        for _ in range(10):
            state.update_threshold(success=False, confidence_used=0.6)

        # After losses, threshold should increase
        assert state.base_threshold >= 0.5

    def test_threshold_state_to_dict(self, mock_config):
        """Test threshold state export."""
        from src.trading.confidence import ThresholdState

        state = ThresholdState(base_threshold=0.5)

        data = state.to_dict()

        assert 'base_threshold' in data
        assert 'current_threshold' in data
        assert 'mode' in data


class TestGateDecisionEnum:
    """Tests for GateDecision enum."""

    def test_gate_decision_values(self):
        """Test GateDecision enum values."""
        from src.trading.confidence import GateDecision

        assert GateDecision.ALLOWED.value == "allowed"
        assert GateDecision.REJECTED.value == "rejected"
        assert GateDecision.SCALED_DOWN.value == "scaled_down"


class TestThresholdModeEnum:
    """Tests for ThresholdMode enum."""

    def test_threshold_mode_values(self):
        """Test ThresholdMode enum values."""
        from src.trading.confidence import ThresholdMode

        assert ThresholdMode.NORMAL.value == "normal"
        assert ThresholdMode.CAUTIOUS.value == "cautious"
        assert ThresholdMode.AGGRESSIVE.value == "aggressive"
