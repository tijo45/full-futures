"""
Unit tests for PositionManager - Authoritative order/position tracking.

Tests cover:
- Order creation and lifecycle
- Fill recording
- Position tracking and P&L
- Exposure calculations
- Reconciliation with IB
- Audit trail
"""

import pytest
import asyncio
import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock, patch

sys.path.insert(0, '.')


@pytest.fixture
def mock_config():
    """Mock configuration for position manager."""
    config = MagicMock()
    return config


@pytest.fixture
def mock_ib_client():
    """Create mock IB client for reconciliation tests."""
    client = MagicMock()
    client.request_positions = AsyncMock(return_value=[])
    return client


@pytest.fixture
def mock_ib_position():
    """Create a mock IB position."""
    position = MagicMock()
    position.contract = MagicMock()
    position.contract.conId = 123456
    position.contract.symbol = 'ES'
    position.contract.exchange = 'CME'
    position.contract.multiplier = '50'
    position.position = 5
    position.avgCost = 225000.0  # 4500 * 50
    return position


class TestPositionManagerInitialization:
    """Tests for PositionManager initialization."""

    def test_initialization_with_defaults(self, mock_config):
        """Test PositionManager initializes with default limits."""
        with patch('src.trading.position_manager.get_config', return_value=mock_config):
            from src.trading.position_manager import PositionManager

            manager = PositionManager()

            assert manager._max_exposure == 50
            assert manager._max_position == 10
            assert len(manager._orders) == 0
            assert len(manager._positions) == 0

    def test_initialization_with_custom_limits(self, mock_config):
        """Test PositionManager with custom limits."""
        with patch('src.trading.position_manager.get_config', return_value=mock_config):
            from src.trading.position_manager import PositionManager

            manager = PositionManager(
                max_exposure_contracts=100,
                max_position_per_contract=20
            )

            assert manager._max_exposure == 100
            assert manager._max_position == 20


class TestOrderCreation:
    """Tests for order creation."""

    def test_create_order(self, mock_config):
        """Test creating a new order."""
        with patch('src.trading.position_manager.get_config', return_value=mock_config):
            from src.trading.position_manager import PositionManager, OrderState

            manager = PositionManager()

            order = manager.create_order(
                contract_id=123456,
                symbol='ES',
                side='BUY',
                quantity=1,
                exchange='CME',
                reason='Test order'
            )

            assert order.contract_id == 123456
            assert order.symbol == 'ES'
            assert order.side == 'BUY'
            assert order.quantity == 1
            assert order.state == OrderState.PENDING
            assert order.order_id in manager._orders
            assert order.order_id in manager._active_orders

    def test_create_order_with_limit_price(self, mock_config):
        """Test creating limit order."""
        with patch('src.trading.position_manager.get_config', return_value=mock_config):
            from src.trading.position_manager import PositionManager

            manager = PositionManager()

            order = manager.create_order(
                contract_id=123456,
                symbol='ES',
                side='BUY',
                quantity=1,
                order_type='limit',
                limit_price=4500.0
            )

            assert order.order_type == 'limit'
            assert order.limit_price == 4500.0

    def test_create_order_increments_counter(self, mock_config):
        """Test order creation increments total orders."""
        with patch('src.trading.position_manager.get_config', return_value=mock_config):
            from src.trading.position_manager import PositionManager

            manager = PositionManager()

            manager.create_order(
                contract_id=123456,
                symbol='ES',
                side='BUY',
                quantity=1
            )

            assert manager._total_orders == 1


class TestOrderSubmission:
    """Tests for order submission tracking."""

    def test_update_order_submitted(self, mock_config):
        """Test updating order to submitted state."""
        with patch('src.trading.position_manager.get_config', return_value=mock_config):
            from src.trading.position_manager import PositionManager, OrderState

            manager = PositionManager()

            order = manager.create_order(
                contract_id=123456,
                symbol='ES',
                side='BUY',
                quantity=1
            )

            updated = manager.update_order_submitted(
                order_id=order.order_id,
                ib_order_id=12345
            )

            assert updated.state == OrderState.SUBMITTED
            assert updated.ib_order_id == 12345
            assert 12345 in manager._ib_order_map

    def test_update_order_submitted_unknown(self, mock_config):
        """Test updating unknown order returns None."""
        with patch('src.trading.position_manager.get_config', return_value=mock_config):
            from src.trading.position_manager import PositionManager

            manager = PositionManager()

            result = manager.update_order_submitted(
                order_id='unknown-id',
                ib_order_id=12345
            )

            assert result is None


class TestFillRecording:
    """Tests for fill recording."""

    def test_record_fill(self, mock_config):
        """Test recording a fill updates order and position."""
        with patch('src.trading.position_manager.get_config', return_value=mock_config):
            from src.trading.position_manager import PositionManager, OrderState

            manager = PositionManager()

            order = manager.create_order(
                contract_id=123456,
                symbol='ES',
                side='BUY',
                quantity=1
            )

            realized_pnl = manager.record_fill(
                order_id=order.order_id,
                quantity=1,
                price=4500.0,
                commission=2.50
            )

            # Order should be filled
            assert order.state == OrderState.FILLED
            assert order.filled_quantity == 1

            # Position should be created
            position = manager.get_position(123456)
            assert position is not None
            assert position.quantity == 1
            assert position.average_entry_price == 4500.0

    def test_record_fill_partial(self, mock_config):
        """Test recording partial fill."""
        with patch('src.trading.position_manager.get_config', return_value=mock_config):
            from src.trading.position_manager import PositionManager, OrderState

            manager = PositionManager()

            order = manager.create_order(
                contract_id=123456,
                symbol='ES',
                side='BUY',
                quantity=5
            )

            manager.record_fill(
                order_id=order.order_id,
                quantity=2,
                price=4500.0
            )

            assert order.state == OrderState.PARTIAL_FILL
            assert order.filled_quantity == 2
            assert order.remaining_quantity == 3

    def test_record_fill_by_ib_order_id(self, mock_config):
        """Test recording fill by IB order ID."""
        with patch('src.trading.position_manager.get_config', return_value=mock_config):
            from src.trading.position_manager import PositionManager

            manager = PositionManager()

            order = manager.create_order(
                contract_id=123456,
                symbol='ES',
                side='BUY',
                quantity=1
            )
            manager.update_order_submitted(order.order_id, ib_order_id=12345)

            manager.record_fill(
                ib_order_id=12345,
                quantity=1,
                price=4500.0
            )

            assert order.filled_quantity == 1


class TestPositionTracking:
    """Tests for position tracking."""

    def test_get_position(self, mock_config):
        """Test getting a position."""
        with patch('src.trading.position_manager.get_config', return_value=mock_config):
            from src.trading.position_manager import PositionManager

            manager = PositionManager()

            order = manager.create_order(
                contract_id=123456,
                symbol='ES',
                side='BUY',
                quantity=1
            )
            manager.record_fill(
                order_id=order.order_id,
                quantity=1,
                price=4500.0
            )

            position = manager.get_position(123456)

            assert position is not None
            assert position.quantity == 1

    def test_get_position_quantity(self, mock_config):
        """Test getting position quantity."""
        with patch('src.trading.position_manager.get_config', return_value=mock_config):
            from src.trading.position_manager import PositionManager

            manager = PositionManager()

            # No position yet
            assert manager.get_position_quantity(123456) == 0

            # Create position
            order = manager.create_order(
                contract_id=123456,
                symbol='ES',
                side='BUY',
                quantity=1
            )
            manager.record_fill(
                order_id=order.order_id,
                quantity=1,
                price=4500.0
            )

            assert manager.get_position_quantity(123456) == 1

    def test_get_open_positions(self, mock_config):
        """Test getting open positions."""
        with patch('src.trading.position_manager.get_config', return_value=mock_config):
            from src.trading.position_manager import PositionManager

            manager = PositionManager()

            # Create two positions
            order1 = manager.create_order(
                contract_id=123456,
                symbol='ES',
                side='BUY',
                quantity=1
            )
            manager.record_fill(order_id=order1.order_id, quantity=1, price=4500.0)

            order2 = manager.create_order(
                contract_id=789012,
                symbol='NQ',
                side='SELL',
                quantity=1
            )
            manager.record_fill(order_id=order2.order_id, quantity=1, price=16000.0)

            positions = manager.get_open_positions()

            assert len(positions) == 2


class TestPnLCalculation:
    """Tests for P&L calculation."""

    def test_realized_pnl_on_close(self, mock_config):
        """Test realized P&L calculated when closing position."""
        with patch('src.trading.position_manager.get_config', return_value=mock_config):
            from src.trading.position_manager import PositionManager

            manager = PositionManager()

            # Open long position
            buy_order = manager.create_order(
                contract_id=123456,
                symbol='ES',
                side='BUY',
                quantity=1
            )
            manager.record_fill(
                order_id=buy_order.order_id,
                quantity=1,
                price=4500.0
            )

            # Set multiplier
            manager.set_contract_multiplier(123456, 50.0)

            # Close position at higher price
            sell_order = manager.create_order(
                contract_id=123456,
                symbol='ES',
                side='SELL',
                quantity=1
            )
            realized_pnl = manager.record_fill(
                order_id=sell_order.order_id,
                quantity=1,
                price=4510.0
            )

            # P&L should be (4510 - 4500) * 1 * 50 = 500
            assert realized_pnl == 500.0

    def test_unrealized_pnl(self, mock_config):
        """Test unrealized P&L calculation."""
        with patch('src.trading.position_manager.get_config', return_value=mock_config):
            from src.trading.position_manager import PositionManager

            manager = PositionManager()

            order = manager.create_order(
                contract_id=123456,
                symbol='ES',
                side='BUY',
                quantity=1
            )
            manager.record_fill(
                order_id=order.order_id,
                quantity=1,
                price=4500.0
            )
            manager.set_contract_multiplier(123456, 50.0)

            # Update with current price
            manager.update_market_prices({123456: 4520.0})

            position = manager.get_position(123456)

            # Unrealized P&L should be (4520 - 4500) * 1 * 50 = 1000
            assert position.unrealized_pnl == 1000.0


class TestExposureCalculations:
    """Tests for exposure calculations."""

    def test_get_total_exposure(self, mock_config):
        """Test total exposure calculation."""
        with patch('src.trading.position_manager.get_config', return_value=mock_config):
            from src.trading.position_manager import PositionManager

            manager = PositionManager()

            # Create long and short positions
            order1 = manager.create_order(
                contract_id=123456,
                symbol='ES',
                side='BUY',
                quantity=3
            )
            manager.record_fill(order_id=order1.order_id, quantity=3, price=4500.0)

            order2 = manager.create_order(
                contract_id=789012,
                symbol='NQ',
                side='SELL',
                quantity=2
            )
            manager.record_fill(order_id=order2.order_id, quantity=2, price=16000.0)

            exposure = manager.get_total_exposure()

            # Should be abs(3) + abs(-2) = 5
            assert exposure == 5

    def test_get_net_exposure(self, mock_config):
        """Test net exposure calculation."""
        with patch('src.trading.position_manager.get_config', return_value=mock_config):
            from src.trading.position_manager import PositionManager

            manager = PositionManager()

            # Create long and short positions
            order1 = manager.create_order(
                contract_id=123456,
                symbol='ES',
                side='BUY',
                quantity=3
            )
            manager.record_fill(order_id=order1.order_id, quantity=3, price=4500.0)

            order2 = manager.create_order(
                contract_id=789012,
                symbol='NQ',
                side='SELL',
                quantity=2
            )
            manager.record_fill(order_id=order2.order_id, quantity=2, price=16000.0)

            exposure = manager.get_net_exposure()

            # Should be 3 + (-2) = 1
            assert exposure == 1

    def test_get_exposure_by_side(self, mock_config):
        """Test exposure breakdown by side."""
        with patch('src.trading.position_manager.get_config', return_value=mock_config):
            from src.trading.position_manager import PositionManager

            manager = PositionManager()

            order1 = manager.create_order(
                contract_id=123456,
                symbol='ES',
                side='BUY',
                quantity=3
            )
            manager.record_fill(order_id=order1.order_id, quantity=3, price=4500.0)

            order2 = manager.create_order(
                contract_id=789012,
                symbol='NQ',
                side='SELL',
                quantity=2
            )
            manager.record_fill(order_id=order2.order_id, quantity=2, price=16000.0)

            exposure = manager.get_exposure_by_side()

            assert exposure['long'] == 3
            assert exposure['short'] == 2
            assert exposure['net'] == 1
            assert exposure['gross'] == 5


class TestLimitChecks:
    """Tests for position and exposure limit checks."""

    def test_check_position_limit_within(self, mock_config):
        """Test position limit check passes within limit."""
        with patch('src.trading.position_manager.get_config', return_value=mock_config):
            from src.trading.position_manager import PositionManager

            manager = PositionManager(max_position_per_contract=10)

            result = manager.check_position_limit(123456, 5)

            assert result is True

    def test_check_position_limit_exceeded(self, mock_config):
        """Test position limit check fails when exceeded."""
        with patch('src.trading.position_manager.get_config', return_value=mock_config):
            from src.trading.position_manager import PositionManager

            manager = PositionManager(max_position_per_contract=10)

            # Create existing position
            order = manager.create_order(
                contract_id=123456,
                symbol='ES',
                side='BUY',
                quantity=8
            )
            manager.record_fill(order_id=order.order_id, quantity=8, price=4500.0)

            # Check if adding more would exceed
            result = manager.check_position_limit(123456, 5)

            assert result is False

    def test_check_exposure_limit_within(self, mock_config):
        """Test exposure limit check passes within limit."""
        with patch('src.trading.position_manager.get_config', return_value=mock_config):
            from src.trading.position_manager import PositionManager

            manager = PositionManager(max_exposure_contracts=50)

            result = manager.check_exposure_limit(10)

            assert result is True

    def test_check_exposure_limit_exceeded(self, mock_config):
        """Test exposure limit check fails when exceeded."""
        with patch('src.trading.position_manager.get_config', return_value=mock_config):
            from src.trading.position_manager import PositionManager

            manager = PositionManager(max_exposure_contracts=10)

            # Create existing exposure
            order = manager.create_order(
                contract_id=123456,
                symbol='ES',
                side='BUY',
                quantity=8
            )
            manager.record_fill(order_id=order.order_id, quantity=8, price=4500.0)

            # Check if adding more would exceed
            result = manager.check_exposure_limit(5)

            assert result is False


class TestOrderCancellation:
    """Tests for order cancellation."""

    def test_record_cancellation(self, mock_config):
        """Test recording order cancellation."""
        with patch('src.trading.position_manager.get_config', return_value=mock_config):
            from src.trading.position_manager import PositionManager, OrderState

            manager = PositionManager()

            order = manager.create_order(
                contract_id=123456,
                symbol='ES',
                side='BUY',
                quantity=1
            )

            result = manager.record_cancellation(order_id=order.order_id)

            assert result.state == OrderState.CANCELLED
            assert order.order_id not in manager._active_orders


class TestReconciliation:
    """Tests for IB reconciliation."""

    @pytest.mark.asyncio
    async def test_reconcile_match(self, mock_config, mock_ib_client, mock_ib_position):
        """Test reconciliation with matching position."""
        mock_ib_client.request_positions = AsyncMock(return_value=[mock_ib_position])

        with patch('src.trading.position_manager.get_config', return_value=mock_config):
            from src.trading.position_manager import PositionManager

            manager = PositionManager()

            # Create matching local position
            order = manager.create_order(
                contract_id=123456,
                symbol='ES',
                side='BUY',
                quantity=5
            )
            manager.record_fill(order_id=order.order_id, quantity=5, price=4500.0)

            report = await manager.reconcile_with_ib(mock_ib_client)

            assert report.is_clean is True
            assert report.matches == 1
            assert report.mismatches == 0

    @pytest.mark.asyncio
    async def test_reconcile_mismatch(self, mock_config, mock_ib_client, mock_ib_position):
        """Test reconciliation with mismatched position."""
        mock_ib_client.request_positions = AsyncMock(return_value=[mock_ib_position])

        with patch('src.trading.position_manager.get_config', return_value=mock_config):
            from src.trading.position_manager import PositionManager

            manager = PositionManager()

            # Create different local position
            order = manager.create_order(
                contract_id=123456,
                symbol='ES',
                side='BUY',
                quantity=3  # Different from IB's 5
            )
            manager.record_fill(order_id=order.order_id, quantity=3, price=4500.0)

            report = await manager.reconcile_with_ib(mock_ib_client, auto_correct=True)

            assert report.is_clean is False
            assert report.mismatches == 1
            assert report.corrections_applied == 1

    @pytest.mark.asyncio
    async def test_reconcile_ib_only(self, mock_config, mock_ib_client, mock_ib_position):
        """Test reconciliation with IB-only position."""
        mock_ib_client.request_positions = AsyncMock(return_value=[mock_ib_position])

        with patch('src.trading.position_manager.get_config', return_value=mock_config):
            from src.trading.position_manager import PositionManager

            manager = PositionManager()

            # No local positions

            report = await manager.reconcile_with_ib(mock_ib_client, auto_correct=True)

            assert report.is_clean is False
            assert report.ib_only == 1
            assert report.corrections_applied == 1


class TestAuditLog:
    """Tests for audit log."""

    def test_audit_log_recorded(self, mock_config):
        """Test that actions are recorded to audit log."""
        with patch('src.trading.position_manager.get_config', return_value=mock_config):
            from src.trading.position_manager import PositionManager

            manager = PositionManager()

            order = manager.create_order(
                contract_id=123456,
                symbol='ES',
                side='BUY',
                quantity=1
            )
            manager.record_fill(order_id=order.order_id, quantity=1, price=4500.0)

            audit_log = manager.get_audit_log()

            assert len(audit_log) >= 2  # At least order created and fill recorded

    def test_get_audit_log_limit(self, mock_config):
        """Test audit log respects count limit."""
        with patch('src.trading.position_manager.get_config', return_value=mock_config):
            from src.trading.position_manager import PositionManager

            manager = PositionManager()

            # Create multiple events
            for i in range(5):
                order = manager.create_order(
                    contract_id=123456 + i,
                    symbol='ES',
                    side='BUY',
                    quantity=1
                )

            audit_log = manager.get_audit_log(count=3)

            assert len(audit_log) <= 3


class TestManagerState:
    """Tests for manager state export."""

    def test_get_state(self, mock_config):
        """Test getting complete manager state."""
        with patch('src.trading.position_manager.get_config', return_value=mock_config):
            from src.trading.position_manager import PositionManager

            manager = PositionManager()

            order = manager.create_order(
                contract_id=123456,
                symbol='ES',
                side='BUY',
                quantity=1
            )
            manager.record_fill(order_id=order.order_id, quantity=1, price=4500.0)

            state = manager.get_state()

            assert 'total_orders' in state
            assert 'active_orders' in state
            assert 'positions_count' in state
            assert 'exposure' in state

    def test_get_summary(self, mock_config):
        """Test getting manager summary for dashboard."""
        with patch('src.trading.position_manager.get_config', return_value=mock_config):
            from src.trading.position_manager import PositionManager

            manager = PositionManager()

            summary = manager.get_summary()

            assert 'active_orders' in summary
            assert 'open_positions' in summary
            assert 'gross_exposure' in summary
            assert 'net_exposure' in summary
            assert 'total_pnl' in summary

    def test_reset(self, mock_config):
        """Test resetting manager state."""
        with patch('src.trading.position_manager.get_config', return_value=mock_config):
            from src.trading.position_manager import PositionManager

            manager = PositionManager()

            order = manager.create_order(
                contract_id=123456,
                symbol='ES',
                side='BUY',
                quantity=1
            )
            manager.record_fill(order_id=order.order_id, quantity=1, price=4500.0)

            manager.reset()

            assert len(manager._orders) == 0
            assert len(manager._positions) == 0
            assert manager._total_orders == 0


class TestCallbacks:
    """Tests for callback registration."""

    def test_register_position_callback(self, mock_config):
        """Test registering position change callback."""
        with patch('src.trading.position_manager.get_config', return_value=mock_config):
            from src.trading.position_manager import PositionManager

            manager = PositionManager()
            callback = MagicMock()

            manager.register_position_callback(callback)

            assert callback in manager._on_position_change

    def test_position_callback_triggered(self, mock_config):
        """Test position callback is triggered on fill."""
        with patch('src.trading.position_manager.get_config', return_value=mock_config):
            from src.trading.position_manager import PositionManager

            manager = PositionManager()
            callback = MagicMock()
            manager.register_position_callback(callback)

            order = manager.create_order(
                contract_id=123456,
                symbol='ES',
                side='BUY',
                quantity=1
            )
            manager.record_fill(order_id=order.order_id, quantity=1, price=4500.0)

            callback.assert_called_once()


class TestOrderRecord:
    """Tests for OrderRecord dataclass."""

    def test_order_record_to_dict(self, mock_config):
        """Test OrderRecord export to dictionary."""
        from src.trading.position_manager import OrderRecord, OrderState

        order = OrderRecord(
            order_id='test-123',
            contract_id=123456,
            symbol='ES',
            side='BUY',
            quantity=1,
            state=OrderState.PENDING
        )

        data = order.to_dict()

        assert data['order_id'] == 'test-123'
        assert data['symbol'] == 'ES'
        assert data['state'] == 'pending'

    def test_order_is_complete(self, mock_config):
        """Test order is_complete property."""
        from src.trading.position_manager import OrderRecord, OrderState

        order = OrderRecord(
            order_id='test-123',
            contract_id=123456,
            symbol='ES',
            side='BUY',
            quantity=1
        )

        assert order.is_complete is False

        order.state = OrderState.FILLED
        assert order.is_complete is True

    def test_order_is_active(self, mock_config):
        """Test order is_active property."""
        from src.trading.position_manager import OrderRecord, OrderState

        order = OrderRecord(
            order_id='test-123',
            contract_id=123456,
            symbol='ES',
            side='BUY',
            quantity=1,
            state=OrderState.SUBMITTED
        )

        assert order.is_active is True

        order.state = OrderState.FILLED
        assert order.is_active is False


class TestPositionRecord:
    """Tests for PositionRecord dataclass."""

    def test_position_record_side(self, mock_config):
        """Test position side property."""
        from src.trading.position_manager import PositionRecord, PositionSide

        position = PositionRecord(contract_id=123456)

        assert position.side == PositionSide.FLAT

        position.quantity = 5
        assert position.side == PositionSide.LONG

        position.quantity = -5
        assert position.side == PositionSide.SHORT

    def test_position_update(self, mock_config):
        """Test position update method."""
        from src.trading.position_manager import PositionRecord

        position = PositionRecord(contract_id=123456, multiplier=50.0)

        # Open long
        position.update_position(quantity_change=1, price=4500.0)
        assert position.quantity == 1
        assert position.average_entry_price == 4500.0

        # Add to long
        position.update_position(quantity_change=1, price=4510.0)
        assert position.quantity == 2
        assert position.average_entry_price == 4505.0  # Average

    def test_position_pnl(self, mock_config):
        """Test position P&L calculations."""
        from src.trading.position_manager import PositionRecord

        position = PositionRecord(contract_id=123456, multiplier=50.0)

        position.update_position(quantity_change=1, price=4500.0)
        position.update_unrealized_pnl(4520.0)

        assert position.unrealized_pnl == 1000.0  # (4520-4500)*1*50


class TestEnums:
    """Tests for enums."""

    def test_order_state_values(self):
        """Test OrderState enum values."""
        from src.trading.position_manager import OrderState

        assert OrderState.PENDING.value == 'pending'
        assert OrderState.SUBMITTED.value == 'submitted'
        assert OrderState.FILLED.value == 'filled'
        assert OrderState.CANCELLED.value == 'cancelled'

    def test_position_side_values(self):
        """Test PositionSide enum values."""
        from src.trading.position_manager import PositionSide

        assert PositionSide.LONG.value == 'long'
        assert PositionSide.SHORT.value == 'short'
        assert PositionSide.FLAT.value == 'flat'
