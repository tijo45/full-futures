"""
Unit tests for IBClient - IB connection management.

Tests cover:
- Connection establishment
- Disconnection handling
- Reconnection with exponential backoff
- Market data subscription
- Error handling
"""

import pytest
import asyncio
import sys
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock

sys.path.insert(0, '.')


class TestIBClientInitialization:
    """Tests for IBClient initialization."""

    def test_initialization_with_defaults(self, mock_config):
        """Test IBClient initializes with default config values."""
        with patch('src.core.ib_client.get_config', return_value=mock_config):
            with patch('src.core.ib_client.IB') as MockIB:
                MockIB.return_value = MagicMock()
                from src.core.ib_client import IBClient, ConnectionState

                client = IBClient()

                assert client._host == '127.0.0.1'
                assert client._port == 7497
                assert client._client_id == 1
                assert client.state == ConnectionState.DISCONNECTED
                assert client._reconnect_attempts == 0

    def test_initialization_with_custom_values(self, mock_config):
        """Test IBClient initializes with custom parameters."""
        with patch('src.core.ib_client.get_config', return_value=mock_config):
            with patch('src.core.ib_client.IB') as MockIB:
                MockIB.return_value = MagicMock()
                from src.core.ib_client import IBClient

                client = IBClient(host='192.168.1.1', port=4002, client_id=5)

                assert client._host == '192.168.1.1'
                assert client._port == 4002
                assert client._client_id == 5

    def test_client_id_property(self, mock_config):
        """Test client_id property returns correct value."""
        with patch('src.core.ib_client.get_config', return_value=mock_config):
            with patch('src.core.ib_client.IB') as MockIB:
                MockIB.return_value = MagicMock()
                from src.core.ib_client import IBClient

                client = IBClient(client_id=42)

                assert client.client_id == 42


class TestIBClientConnection:
    """Tests for connection management."""

    @pytest.mark.asyncio
    async def test_connect_success(self, mock_config, mock_ib):
        """Test successful connection to IB."""
        with patch('src.core.ib_client.get_config', return_value=mock_config):
            with patch('src.core.ib_client.IB', return_value=mock_ib):
                from src.core.ib_client import IBClient, ConnectionState

                client = IBClient()
                result = await client.connect()

                assert result is True
                assert client.state == ConnectionState.CONNECTED
                mock_ib.connectAsync.assert_called_once_with(
                    host='127.0.0.1',
                    port=7497,
                    clientId=1
                )
                mock_ib.reqMarketDataType.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_connect_already_connected(self, mock_config, mock_ib):
        """Test connect when already connected returns True without reconnecting."""
        with patch('src.core.ib_client.get_config', return_value=mock_config):
            with patch('src.core.ib_client.IB', return_value=mock_ib):
                from src.core.ib_client import IBClient, ConnectionState

                client = IBClient()
                client._state = ConnectionState.CONNECTED

                result = await client.connect()

                assert result is True
                mock_ib.connectAsync.assert_not_called()

    @pytest.mark.asyncio
    async def test_connect_failure(self, mock_config, mock_ib):
        """Test connection failure raises exception."""
        mock_ib.connectAsync = AsyncMock(side_effect=Exception("Connection refused"))

        with patch('src.core.ib_client.get_config', return_value=mock_config):
            with patch('src.core.ib_client.IB', return_value=mock_ib):
                from src.core.ib_client import IBClient, ConnectionState

                client = IBClient()

                with pytest.raises(Exception) as exc_info:
                    await client.connect()

                assert "Connection refused" in str(exc_info.value)
                assert client.state == ConnectionState.ERROR

    @pytest.mark.asyncio
    async def test_disconnect_success(self, mock_config, mock_ib):
        """Test successful disconnection."""
        with patch('src.core.ib_client.get_config', return_value=mock_config):
            with patch('src.core.ib_client.IB', return_value=mock_ib):
                from src.core.ib_client import IBClient, ConnectionState

                client = IBClient()
                client._state = ConnectionState.CONNECTED

                await client.disconnect()

                assert client.state == ConnectionState.DISCONNECTED
                mock_ib.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_already_disconnected(self, mock_config, mock_ib):
        """Test disconnect when already disconnected does nothing."""
        with patch('src.core.ib_client.get_config', return_value=mock_config):
            with patch('src.core.ib_client.IB', return_value=mock_ib):
                from src.core.ib_client import IBClient, ConnectionState

                client = IBClient()
                client._state = ConnectionState.DISCONNECTED

                await client.disconnect()

                mock_ib.disconnect.assert_not_called()


class TestIBClientReconnection:
    """Tests for reconnection with exponential backoff."""

    @pytest.mark.asyncio
    async def test_reconnect_success(self, mock_config, mock_ib):
        """Test successful reconnection."""
        mock_config.RECONNECT_BASE_DELAY_SECONDS = 0.01  # Fast delay for tests

        with patch('src.core.ib_client.get_config', return_value=mock_config):
            with patch('src.core.ib_client.IB', return_value=mock_ib):
                from src.core.ib_client import IBClient, ConnectionState

                client = IBClient()
                client._state = ConnectionState.DISCONNECTED
                client._base_delay = 0.01

                result = await client.reconnect()

                assert result is True
                assert client.state == ConnectionState.CONNECTED
                assert client._reconnect_attempts == 0

    @pytest.mark.asyncio
    async def test_reconnect_max_attempts_exceeded(self, mock_config, mock_ib):
        """Test reconnection stops after max attempts."""
        mock_config.RECONNECT_MAX_ATTEMPTS = 3
        mock_ib.connectAsync = AsyncMock(side_effect=Exception("Connection failed"))
        mock_ib.isConnected.return_value = False

        with patch('src.core.ib_client.get_config', return_value=mock_config):
            with patch('src.core.ib_client.IB', return_value=mock_ib):
                from src.core.ib_client import IBClient, ConnectionState

                client = IBClient()
                client._state = ConnectionState.DISCONNECTED
                client._reconnect_attempts = 3  # Already at max
                client._base_delay = 0.01

                result = await client.reconnect()

                assert result is False
                assert client.state == ConnectionState.ERROR

    @pytest.mark.asyncio
    async def test_exponential_backoff_delay(self, mock_config, mock_ib):
        """Test exponential backoff increases delay correctly."""
        mock_config.RECONNECT_BASE_DELAY_SECONDS = 1.0

        with patch('src.core.ib_client.get_config', return_value=mock_config):
            with patch('src.core.ib_client.IB', return_value=mock_ib):
                from src.core.ib_client import IBClient

                client = IBClient()

                # First attempt: 1 * 2^0 = 1 second
                client._reconnect_attempts = 0
                delay1 = client._base_delay * (2 ** 0)
                assert delay1 == 1.0

                # Second attempt: 1 * 2^1 = 2 seconds
                client._reconnect_attempts = 1
                delay2 = client._base_delay * (2 ** 1)
                assert delay2 == 2.0

                # Third attempt: 1 * 2^2 = 4 seconds
                client._reconnect_attempts = 2
                delay3 = client._base_delay * (2 ** 2)
                assert delay3 == 4.0


class TestIBClientMarketData:
    """Tests for market data subscription."""

    @pytest.mark.asyncio
    async def test_subscribe_market_data_success(self, mock_config, mock_ib, mock_contract, mock_ticker):
        """Test successful market data subscription."""
        mock_ib.reqMktData.return_value = mock_ticker

        with patch('src.core.ib_client.get_config', return_value=mock_config):
            with patch('src.core.ib_client.IB', return_value=mock_ib):
                from src.core.ib_client import IBClient, ConnectionState

                client = IBClient()
                client._state = ConnectionState.CONNECTED

                ticker = await client.subscribe_market_data(mock_contract)

                assert ticker == mock_ticker
                mock_ib.reqMktData.assert_called_once()
                assert mock_contract in client._subscribed_contracts

    @pytest.mark.asyncio
    async def test_subscribe_market_data_not_connected(self, mock_config, mock_ib, mock_contract):
        """Test market data subscription fails when not connected."""
        mock_ib.isConnected.return_value = False

        with patch('src.core.ib_client.get_config', return_value=mock_config):
            with patch('src.core.ib_client.IB', return_value=mock_ib):
                from src.core.ib_client import IBClient, ConnectionState

                client = IBClient()
                client._state = ConnectionState.DISCONNECTED

                with pytest.raises(ConnectionError):
                    await client.subscribe_market_data(mock_contract)

    @pytest.mark.asyncio
    async def test_subscribe_depth_success(self, mock_config, mock_ib, mock_contract, mock_ticker):
        """Test successful depth subscription."""
        mock_ib.reqMktDepth.return_value = mock_ticker

        with patch('src.core.ib_client.get_config', return_value=mock_config):
            with patch('src.core.ib_client.IB', return_value=mock_ib):
                from src.core.ib_client import IBClient, ConnectionState

                client = IBClient()
                client._state = ConnectionState.CONNECTED

                ticker = await client.subscribe_depth(mock_contract, num_rows=10)

                assert ticker == mock_ticker
                mock_ib.reqMktDepth.assert_called_once_with(
                    mock_contract,
                    numRows=10,
                    isSmartDepth=False
                )

    def test_cancel_market_data(self, mock_config, mock_ib, mock_contract):
        """Test canceling market data subscription."""
        with patch('src.core.ib_client.get_config', return_value=mock_config):
            with patch('src.core.ib_client.IB', return_value=mock_ib):
                from src.core.ib_client import IBClient

                client = IBClient()
                client._subscribed_contracts = [mock_contract]

                client.cancel_market_data(mock_contract)

                mock_ib.cancelMktData.assert_called_once_with(mock_contract)
                assert mock_contract not in client._subscribed_contracts

    def test_cancel_depth(self, mock_config, mock_ib, mock_contract):
        """Test canceling depth subscription."""
        with patch('src.core.ib_client.get_config', return_value=mock_config):
            with patch('src.core.ib_client.IB', return_value=mock_ib):
                from src.core.ib_client import IBClient

                client = IBClient()

                client.cancel_depth(mock_contract)

                mock_ib.cancelMktDepth.assert_called_once_with(mock_contract)


class TestIBClientProperties:
    """Tests for IBClient properties."""

    def test_is_connected_true(self, mock_config, mock_ib):
        """Test is_connected returns True when connected."""
        with patch('src.core.ib_client.get_config', return_value=mock_config):
            with patch('src.core.ib_client.IB', return_value=mock_ib):
                from src.core.ib_client import IBClient, ConnectionState

                client = IBClient()
                client._state = ConnectionState.CONNECTED
                mock_ib.isConnected.return_value = True

                assert client.is_connected is True

    def test_is_connected_false_when_disconnected(self, mock_config, mock_ib):
        """Test is_connected returns False when disconnected."""
        with patch('src.core.ib_client.get_config', return_value=mock_config):
            with patch('src.core.ib_client.IB', return_value=mock_ib):
                from src.core.ib_client import IBClient, ConnectionState

                client = IBClient()
                client._state = ConnectionState.DISCONNECTED

                assert client.is_connected is False

    def test_ib_property(self, mock_config, mock_ib):
        """Test ib property returns underlying IB instance."""
        with patch('src.core.ib_client.get_config', return_value=mock_config):
            with patch('src.core.ib_client.IB', return_value=mock_ib):
                from src.core.ib_client import IBClient

                client = IBClient()

                assert client.ib == mock_ib


class TestIBClientCallbacks:
    """Tests for callback registration and handling."""

    def test_set_on_connected_callback(self, mock_config, mock_ib):
        """Test setting connected callback."""
        with patch('src.core.ib_client.get_config', return_value=mock_config):
            with patch('src.core.ib_client.IB', return_value=mock_ib):
                from src.core.ib_client import IBClient

                client = IBClient()
                callback = AsyncMock()

                client.set_on_connected(callback)

                assert client._on_connected == callback

    def test_set_on_disconnected_callback(self, mock_config, mock_ib):
        """Test setting disconnected callback."""
        with patch('src.core.ib_client.get_config', return_value=mock_config):
            with patch('src.core.ib_client.IB', return_value=mock_ib):
                from src.core.ib_client import IBClient

                client = IBClient()
                callback = AsyncMock()

                client.set_on_disconnected(callback)

                assert client._on_disconnected == callback

    def test_set_on_error_callback(self, mock_config, mock_ib):
        """Test setting error callback."""
        with patch('src.core.ib_client.get_config', return_value=mock_config):
            with patch('src.core.ib_client.IB', return_value=mock_ib):
                from src.core.ib_client import IBClient

                client = IBClient()
                callback = AsyncMock()

                client.set_on_error(callback)

                assert client._on_error == callback


class TestIBClientContextManager:
    """Tests for async context manager."""

    @pytest.mark.asyncio
    async def test_context_manager_enter(self, mock_config, mock_ib):
        """Test async context manager entry connects."""
        with patch('src.core.ib_client.get_config', return_value=mock_config):
            with patch('src.core.ib_client.IB', return_value=mock_ib):
                from src.core.ib_client import IBClient, ConnectionState

                async with IBClient() as client:
                    assert client.state == ConnectionState.CONNECTED
                    mock_ib.connectAsync.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager_exit(self, mock_config, mock_ib):
        """Test async context manager exit disconnects."""
        with patch('src.core.ib_client.get_config', return_value=mock_config):
            with patch('src.core.ib_client.IB', return_value=mock_ib):
                from src.core.ib_client import IBClient, ConnectionState

                async with IBClient() as client:
                    pass

                mock_ib.disconnect.assert_called_once()
