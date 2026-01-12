"""
Pytest configuration and shared fixtures for unit tests.

Provides common fixtures for testing trading bot components.
"""

import pytest
import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock, patch

# Add project root to path
sys.path.insert(0, '.')


@pytest.fixture
def mock_config():
    """Mock configuration object for tests."""
    config = MagicMock()
    config.IB_HOST = '127.0.0.1'
    config.IB_PORT = 7497
    config.IB_CLIENT_ID = 1
    config.RECONNECT_MAX_ATTEMPTS = 5
    config.RECONNECT_BASE_DELAY_SECONDS = 0.1
    config.SESSION_CLOSE_BUFFER_MINUTES = 5
    config.DATA_STALENESS_THRESHOLD_SECONDS = 30
    return config


@pytest.fixture
def mock_ib():
    """Mock IB connection object."""
    ib = MagicMock()
    ib.isConnected.return_value = True
    ib.connectAsync = AsyncMock()
    ib.disconnect = MagicMock()
    ib.reqMktData = MagicMock()
    ib.reqMktDepth = MagicMock()
    ib.cancelMktData = MagicMock()
    ib.cancelMktDepth = MagicMock()
    ib.reqMarketDataType = MagicMock()
    ib.positions = MagicMock(return_value=[])
    ib.accountSummary = MagicMock(return_value=[])
    ib.qualifyContractsAsync = AsyncMock(return_value=[])
    ib.reqContractDetailsAsync = AsyncMock(return_value=[])
    ib.connectedEvent = MagicMock()
    ib.disconnectedEvent = MagicMock()
    ib.errorEvent = MagicMock()
    return ib


@pytest.fixture
def mock_contract():
    """Mock IB contract object."""
    contract = MagicMock()
    contract.conId = 123456
    contract.symbol = 'ES'
    contract.exchange = 'CME'
    contract.lastTradeDateOrContractMonth = '20241220'
    contract.localSymbol = 'ESZ4'
    contract.multiplier = '50'
    contract.currency = 'USD'
    return contract


@pytest.fixture
def mock_ticker():
    """Mock IB ticker object for market data."""
    ticker = MagicMock()
    ticker.bid = 5000.25
    ticker.ask = 5000.50
    ticker.last = 5000.25
    ticker.bidSize = 100
    ticker.askSize = 150
    ticker.volume = 50000
    ticker.domBids = []
    ticker.domAsks = []
    return ticker


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


@pytest.fixture
def sample_tick_data():
    """Sample tick data dictionary."""
    return {
        'contract_id': 123456,
        'symbol': 'ES',
        'bid': 5000.25,
        'ask': 5000.50,
        'last': 5000.25,
        'bid_size': 100,
        'ask_size': 150,
        'volume': 50000,
        'timestamp': datetime.now(timezone.utc),
    }
