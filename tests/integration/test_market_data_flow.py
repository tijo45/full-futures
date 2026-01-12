"""
Integration tests for Market Data Flow.

Tests the interaction between IBClient and MarketDataHandler:
- Data subscription flow
- Tick data extraction and freshness tracking
- Depth data extraction and order book handling
- Staleness detection and callback handling
- Data update propagation

These tests use mocked IB components to verify integration without
requiring a live TWS/Gateway connection.
"""

import pytest
import asyncio
import sys
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, AsyncMock, patch

sys.path.insert(0, '.')


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
    """Create a mock IBClient with all required methods."""
    client = MagicMock()
    client.is_connected = True
    client.subscribe_market_data = AsyncMock()
    client.subscribe_depth = AsyncMock()
    client.cancel_market_data = MagicMock()
    client.cancel_depth = MagicMock()
    return client


@pytest.fixture
def mock_ticker():
    """Create a mock IB Ticker with realistic data."""
    ticker = MagicMock()
    ticker.bid = 5000.25
    ticker.bidSize = 100
    ticker.ask = 5000.50
    ticker.askSize = 150
    ticker.last = 5000.25
    ticker.lastSize = 10
    ticker.volume = 50000
    ticker.high = 5010.00
    ticker.low = 4990.00
    ticker.open = 4995.00
    ticker.close = 5002.00
    ticker.domBids = []
    ticker.domAsks = []
    ticker.updateEvent = MagicMock()
    ticker.updateEvent.__iadd__ = MagicMock()
    return ticker


@pytest.fixture
def mock_depth_ticker():
    """Create a mock IB Ticker with depth data."""
    ticker = MagicMock()

    # Create mock DOMLevel objects
    class MockDOMLevel:
        def __init__(self, price, size, mm=""):
            self.price = price
            self.size = size
            self.marketMaker = mm

    ticker.domBids = [
        MockDOMLevel(5000.25, 100),
        MockDOMLevel(5000.00, 200),
        MockDOMLevel(4999.75, 150),
    ]
    ticker.domAsks = [
        MockDOMLevel(5000.50, 120),
        MockDOMLevel(5000.75, 180),
        MockDOMLevel(5001.00, 250),
    ]
    ticker.updateEvent = MagicMock()
    ticker.updateEvent.__iadd__ = MagicMock()
    return ticker


@pytest.fixture
def mock_contract():
    """Create a mock IB Contract."""
    contract = MagicMock()
    contract.conId = 123456
    contract.symbol = 'ES'
    contract.exchange = 'CME'
    contract.lastTradeDateOrContractMonth = '20241220'
    return contract


class TestMarketDataSubscriptionFlow:
    """Tests for market data subscription integration."""

    @pytest.mark.asyncio
    async def test_level_1_subscription_flow(self, mock_config, mock_ib_client, mock_contract, mock_ticker):
        """Test Level 1 market data subscription flow."""
        mock_ib_client.subscribe_market_data.return_value = mock_ticker

        with patch('src.data.market_data.get_config', return_value=mock_config):
            from src.data.market_data import MarketDataHandler

            handler = MarketDataHandler(mock_ib_client)

            # Subscribe to Level 1 data
            tick_data = await handler.subscribe_level_1(mock_contract)

            # Verify subscription was made
            mock_ib_client.subscribe_market_data.assert_called_once()

            # Verify tick data was extracted
            assert tick_data is not None
            assert tick_data.contract_id == mock_contract.conId
            assert tick_data.symbol == mock_contract.symbol
            assert tick_data.bid == mock_ticker.bid
            assert tick_data.ask == mock_ticker.ask

    @pytest.mark.asyncio
    async def test_level_2_subscription_flow(self, mock_config, mock_ib_client, mock_contract, mock_depth_ticker):
        """Test Level 2 depth subscription flow."""
        mock_ib_client.subscribe_depth.return_value = mock_depth_ticker

        with patch('src.data.market_data.get_config', return_value=mock_config):
            from src.data.market_data import MarketDataHandler

            handler = MarketDataHandler(mock_ib_client)

            # Subscribe to Level 2 data
            depth_data = await handler.subscribe_level_2(mock_contract, num_rows=5)

            # Verify subscription was made
            mock_ib_client.subscribe_depth.assert_called_once()

            # Verify depth data was extracted
            assert depth_data is not None
            assert depth_data.contract_id == mock_contract.conId
            assert len(depth_data.bids) == 3
            assert len(depth_data.asks) == 3

    @pytest.mark.asyncio
    async def test_full_subscription_flow(self, mock_config, mock_ib_client, mock_contract, mock_ticker, mock_depth_ticker):
        """Test subscribing to both Level 1 and Level 2."""
        mock_ib_client.subscribe_market_data.return_value = mock_ticker
        mock_ib_client.subscribe_depth.return_value = mock_depth_ticker

        with patch('src.data.market_data.get_config', return_value=mock_config):
            from src.data.market_data import MarketDataHandler

            handler = MarketDataHandler(mock_ib_client)

            # Subscribe to all
            tick_data, depth_data = await handler.subscribe_all(mock_contract, num_rows=5)

            # Verify both subscriptions
            assert tick_data is not None
            assert depth_data is not None
            assert handler.level_1_count == 1
            assert handler.level_2_count == 1


class TestTickDataExtraction:
    """Tests for tick data extraction from IB tickers."""

    @pytest.mark.asyncio
    async def test_tick_data_extraction(self, mock_config, mock_ib_client, mock_contract, mock_ticker):
        """Test correct extraction of tick data from IB ticker."""
        mock_ib_client.subscribe_market_data.return_value = mock_ticker

        with patch('src.data.market_data.get_config', return_value=mock_config):
            from src.data.market_data import MarketDataHandler

            handler = MarketDataHandler(mock_ib_client)
            tick_data = await handler.subscribe_level_1(mock_contract)

            # Verify all fields extracted correctly
            assert tick_data.bid == 5000.25
            assert tick_data.bid_size == 100
            assert tick_data.ask == 5000.50
            assert tick_data.ask_size == 150
            assert tick_data.last == 5000.25
            assert tick_data.volume == 50000

            # Verify computed fields
            assert tick_data.spread == pytest.approx(0.25, rel=0.01)
            assert tick_data.mid == pytest.approx(5000.375, rel=0.001)

    @pytest.mark.asyncio
    async def test_tick_data_to_dict(self, mock_config, mock_ib_client, mock_contract, mock_ticker):
        """Test tick data dictionary conversion for feature engine."""
        mock_ib_client.subscribe_market_data.return_value = mock_ticker

        with patch('src.data.market_data.get_config', return_value=mock_config):
            from src.data.market_data import MarketDataHandler

            handler = MarketDataHandler(mock_ib_client)
            tick_data = await handler.subscribe_level_1(mock_contract)

            # Convert to dict
            tick_dict = tick_data.to_dict()

            # Verify dict format
            assert isinstance(tick_dict, dict)
            assert 'bid' in tick_dict
            assert 'ask' in tick_dict
            assert 'spread' in tick_dict
            assert 'mid' in tick_dict


class TestDepthDataExtraction:
    """Tests for depth data extraction from IB tickers."""

    @pytest.mark.asyncio
    async def test_depth_data_extraction(self, mock_config, mock_ib_client, mock_contract, mock_depth_ticker):
        """Test correct extraction of depth data from IB ticker."""
        mock_ib_client.subscribe_depth.return_value = mock_depth_ticker

        with patch('src.data.market_data.get_config', return_value=mock_config):
            from src.data.market_data import MarketDataHandler

            handler = MarketDataHandler(mock_ib_client)
            depth_data = await handler.subscribe_level_2(mock_contract)

            # Verify bids extracted
            assert len(depth_data.bids) == 3
            assert depth_data.bids[0].price == 5000.25
            assert depth_data.bids[0].size == 100

            # Verify asks extracted
            assert len(depth_data.asks) == 3
            assert depth_data.asks[0].price == 5000.50
            assert depth_data.asks[0].size == 120

    @pytest.mark.asyncio
    async def test_depth_data_computed_properties(self, mock_config, mock_ib_client, mock_contract, mock_depth_ticker):
        """Test depth data computed properties."""
        mock_ib_client.subscribe_depth.return_value = mock_depth_ticker

        with patch('src.data.market_data.get_config', return_value=mock_config):
            from src.data.market_data import MarketDataHandler

            handler = MarketDataHandler(mock_ib_client)
            depth_data = await handler.subscribe_level_2(mock_contract)

            # Verify best bid/ask
            assert depth_data.best_bid == 5000.25
            assert depth_data.best_ask == 5000.50

            # Verify totals
            assert depth_data.total_bid_size == 450  # 100 + 200 + 150
            assert depth_data.total_ask_size == 550  # 120 + 180 + 250

            # Verify imbalance (positive means more on bid side)
            expected_imbalance = (450 - 550) / (450 + 550)
            assert depth_data.bid_ask_imbalance == pytest.approx(expected_imbalance, rel=0.01)


class TestDataFreshnessTracking:
    """Tests for data freshness and staleness tracking."""

    @pytest.mark.asyncio
    async def test_data_freshness_check(self, mock_config, mock_ib_client, mock_contract, mock_ticker):
        """Test that fresh data is detected correctly."""
        mock_ib_client.subscribe_market_data.return_value = mock_ticker

        with patch('src.data.market_data.get_config', return_value=mock_config):
            from src.data.market_data import MarketDataHandler, DataType

            handler = MarketDataHandler(mock_ib_client)
            tick_data = await handler.subscribe_level_1(mock_contract)

            # Ensure initial tick_data was returned
            assert tick_data is not None

            # Simulate ticker update to populate subscription.tick_data
            # (subscription.tick_data is populated via update events, not initial subscription)
            handler._handle_ticker_update(mock_contract.conId, mock_ticker)

            # Fresh data should be detected
            assert handler.is_data_fresh(mock_contract, DataType.LEVEL_1) is True

    @pytest.mark.asyncio
    async def test_stale_data_detection(self, mock_config, mock_ib_client, mock_contract, mock_ticker):
        """Test stale data detection when timestamp is old."""
        mock_ib_client.subscribe_market_data.return_value = mock_ticker

        with patch('src.data.market_data.get_config', return_value=mock_config):
            from src.data.market_data import MarketDataHandler, DataType

            handler = MarketDataHandler(mock_ib_client)
            await handler.subscribe_level_1(mock_contract)

            # Simulate ticker update to populate tick_data
            handler._handle_ticker_update(mock_contract.conId, mock_ticker)

            # Manually age the data
            subscription = handler._subscriptions[mock_contract.conId]
            subscription.tick_data.timestamp = datetime.now(timezone.utc) - timedelta(seconds=60)

            # Stale data should be detected
            assert handler.is_data_fresh(mock_contract, DataType.LEVEL_1) is False

    @pytest.mark.asyncio
    async def test_get_stale_contracts(self, mock_config, mock_ib_client, mock_contract, mock_ticker):
        """Test retrieving list of contracts with stale data."""
        mock_ib_client.subscribe_market_data.return_value = mock_ticker

        with patch('src.data.market_data.get_config', return_value=mock_config):
            from src.data.market_data import MarketDataHandler

            handler = MarketDataHandler(mock_ib_client)
            await handler.subscribe_level_1(mock_contract)

            # Simulate ticker update to populate tick_data
            handler._handle_ticker_update(mock_contract.conId, mock_ticker)

            # Initially no stale contracts
            assert len(handler.get_stale_contracts()) == 0

            # Age the data
            subscription = handler._subscriptions[mock_contract.conId]
            subscription.tick_data.timestamp = datetime.now(timezone.utc) - timedelta(seconds=60)

            # Now should have stale contract
            stale = handler.get_stale_contracts()
            assert len(stale) == 1
            assert stale[0][0] == 'ES'


class TestDataUpdatePropagation:
    """Tests for data update callback propagation."""

    @pytest.mark.asyncio
    async def test_tick_callback_invocation(self, mock_config, mock_ib_client, mock_contract, mock_ticker):
        """Test that tick callbacks are invoked on data updates."""
        mock_ib_client.subscribe_market_data.return_value = mock_ticker

        with patch('src.data.market_data.get_config', return_value=mock_config):
            from src.data.market_data import MarketDataHandler

            handler = MarketDataHandler(mock_ib_client)

            # Set up callback tracking
            callback_received = []

            async def on_tick(tick_data):
                callback_received.append(tick_data)

            handler.set_on_tick(on_tick)

            await handler.subscribe_level_1(mock_contract)

            # Simulate a ticker update
            handler._handle_ticker_update(mock_contract.conId, mock_ticker)

            # Allow callback to execute
            await asyncio.sleep(0.1)

            # Verify callback was invoked
            assert len(callback_received) == 1
            assert callback_received[0].symbol == 'ES'

    @pytest.mark.asyncio
    async def test_depth_callback_invocation(self, mock_config, mock_ib_client, mock_contract, mock_depth_ticker):
        """Test that depth callbacks are invoked on data updates."""
        mock_ib_client.subscribe_depth.return_value = mock_depth_ticker

        with patch('src.data.market_data.get_config', return_value=mock_config):
            from src.data.market_data import MarketDataHandler

            handler = MarketDataHandler(mock_ib_client)

            # Set up callback tracking
            callback_received = []

            async def on_depth(depth_data):
                callback_received.append(depth_data)

            handler.set_on_depth(on_depth)

            await handler.subscribe_level_2(mock_contract)

            # Simulate a depth update
            handler._handle_depth_update(mock_contract.conId, mock_depth_ticker)

            # Allow callback to execute
            await asyncio.sleep(0.1)

            # Verify callback was invoked
            assert len(callback_received) == 1
            assert len(callback_received[0].bids) == 3


class TestDataUnsubscription:
    """Tests for data unsubscription flow."""

    @pytest.mark.asyncio
    async def test_level_1_unsubscription(self, mock_config, mock_ib_client, mock_contract, mock_ticker):
        """Test Level 1 unsubscription flow."""
        mock_ib_client.subscribe_market_data.return_value = mock_ticker

        with patch('src.data.market_data.get_config', return_value=mock_config):
            from src.data.market_data import MarketDataHandler

            handler = MarketDataHandler(mock_ib_client)

            # Subscribe
            await handler.subscribe_level_1(mock_contract)
            assert handler.level_1_count == 1

            # Unsubscribe
            await handler.unsubscribe_level_1(mock_contract)

            # Verify cancellation was called
            mock_ib_client.cancel_market_data.assert_called_once()

    @pytest.mark.asyncio
    async def test_full_unsubscription(self, mock_config, mock_ib_client, mock_contract, mock_ticker, mock_depth_ticker):
        """Test full unsubscription cleans up subscription."""
        mock_ib_client.subscribe_market_data.return_value = mock_ticker
        mock_ib_client.subscribe_depth.return_value = mock_depth_ticker

        with patch('src.data.market_data.get_config', return_value=mock_config):
            from src.data.market_data import MarketDataHandler

            handler = MarketDataHandler(mock_ib_client)

            # Subscribe to all
            await handler.subscribe_all(mock_contract)
            assert len(handler._subscriptions) == 1

            # Unsubscribe from all
            await handler.unsubscribe_all(mock_contract)

            # Subscription should be cleaned up
            assert len(handler._subscriptions) == 0


class TestIBClientIntegration:
    """Tests for direct IBClient integration scenarios."""

    @pytest.mark.asyncio
    async def test_subscription_failure_handling(self, mock_config, mock_ib_client, mock_contract):
        """Test handling of subscription failures."""
        mock_ib_client.subscribe_market_data.side_effect = Exception("Subscription failed")

        with patch('src.data.market_data.get_config', return_value=mock_config):
            from src.data.market_data import MarketDataHandler

            handler = MarketDataHandler(mock_ib_client)

            # Subscription should return None on failure
            tick_data = await handler.subscribe_level_1(mock_contract)
            assert tick_data is None

            # No subscription should be tracked
            assert handler.level_1_count == 0

    @pytest.mark.asyncio
    async def test_duplicate_subscription_handling(self, mock_config, mock_ib_client, mock_contract, mock_ticker):
        """Test handling of duplicate subscription requests."""
        mock_ib_client.subscribe_market_data.return_value = mock_ticker

        with patch('src.data.market_data.get_config', return_value=mock_config):
            from src.data.market_data import MarketDataHandler

            handler = MarketDataHandler(mock_ib_client)

            # First subscription
            await handler.subscribe_level_1(mock_contract)

            # Second subscription should not call IB again
            await handler.subscribe_level_1(mock_contract)

            # Should only have called subscribe once
            assert mock_ib_client.subscribe_market_data.call_count == 1

    @pytest.mark.asyncio
    async def test_get_data_retrieval(self, mock_config, mock_ib_client, mock_contract, mock_ticker, mock_depth_ticker):
        """Test data retrieval after subscription."""
        mock_ib_client.subscribe_market_data.return_value = mock_ticker
        mock_ib_client.subscribe_depth.return_value = mock_depth_ticker

        with patch('src.data.market_data.get_config', return_value=mock_config):
            from src.data.market_data import MarketDataHandler

            handler = MarketDataHandler(mock_ib_client)

            # Subscribe
            await handler.subscribe_all(mock_contract)

            # Simulate ticker updates to populate subscription data
            handler._handle_ticker_update(mock_contract.conId, mock_ticker)
            handler._handle_depth_update(mock_contract.conId, mock_depth_ticker)

            # Retrieve data
            tick_data = handler.get_tick_data(mock_contract)
            depth_data = handler.get_depth_data(mock_contract)

            assert tick_data is not None
            assert depth_data is not None
            assert tick_data.symbol == 'ES'
            assert depth_data.symbol == 'ES'

    @pytest.mark.asyncio
    async def test_summary_generation(self, mock_config, mock_ib_client, mock_contract, mock_ticker, mock_depth_ticker):
        """Test summary generation for monitoring."""
        mock_ib_client.subscribe_market_data.return_value = mock_ticker
        mock_ib_client.subscribe_depth.return_value = mock_depth_ticker

        with patch('src.data.market_data.get_config', return_value=mock_config):
            from src.data.market_data import MarketDataHandler

            handler = MarketDataHandler(mock_ib_client)

            # Subscribe
            await handler.subscribe_all(mock_contract)

            # Get summary
            summary = handler.get_summary()

            assert summary['total_subscriptions'] == 1
            assert summary['level_1_count'] == 1
            assert summary['level_2_count'] == 1
            assert summary['stale_count'] == 0
