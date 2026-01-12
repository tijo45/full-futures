"""
IB Client - Async connection manager for Interactive Brokers API.

Uses ib_async library for all IB interactions with proper async patterns,
connection management, and automatic reconnection with exponential backoff.

Supports both ib_async (Python 3.10+) and ib_insync (Python 3.9) for
development compatibility.
"""

import asyncio
import logging
from typing import Optional, Callable, Awaitable, List, Any
from enum import Enum

# Support both ib_async (preferred, Python 3.10+) and ib_insync (fallback)
try:
    from ib_async import IB, Future, Contract, Ticker
    IB_LIBRARY = "ib_async"
except ImportError:
    try:
        from ib_insync import IB, Future, Contract, Ticker
        IB_LIBRARY = "ib_insync"
    except ImportError:
        # Allow import for type checking even without IB library installed
        IB = None  # type: ignore
        Future = None  # type: ignore
        Contract = Any  # type: ignore
        Ticker = Any  # type: ignore
        IB_LIBRARY = None

from config import get_config


class ConnectionState(Enum):
    """IB connection state enumeration."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


class IBClient:
    """
    Async IB connection manager using ib_async.

    Provides connection lifecycle management, automatic reconnection with
    exponential backoff, and market data subscription capabilities.

    Key Features:
    - Async connection management with connectAsync()
    - Automatic reconnection on disconnect
    - Exponential backoff for reconnection attempts
    - Market data subscription (Level 1 and Level 2)
    - Connection state tracking with callbacks

    Usage:
        client = IBClient()
        await client.connect()
        ticker = await client.subscribe_market_data(contract)
        await client.disconnect()
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        client_id: Optional[int] = None,
    ):
        """
        Initialize IBClient with connection parameters.

        Args:
            host: IB Gateway/TWS host address (default from config)
            port: IB Gateway/TWS port (default from config)
            client_id: Unique client ID for this connection (default from config)

        Raises:
            ImportError: If neither ib_async nor ib_insync is installed
        """
        if IB is None:
            raise ImportError(
                "IB library not found. Install ib_async (Python 3.10+) or "
                "ib_insync (Python 3.9+): pip install ib-async or pip install ib-insync"
            )

        config = get_config()

        self._host = host or config.IB_HOST
        self._port = port or config.IB_PORT
        self._client_id = client_id or config.IB_CLIENT_ID

        # IB connection instance
        self._ib = IB()

        # Connection state
        self._state = ConnectionState.DISCONNECTED
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = config.RECONNECT_MAX_ATTEMPTS
        self._base_delay = config.RECONNECT_BASE_DELAY_SECONDS

        # Callbacks
        self._on_connected: Optional[Callable[[], Awaitable[None]]] = None
        self._on_disconnected: Optional[Callable[[], Awaitable[None]]] = None
        self._on_error: Optional[Callable[[Exception], Awaitable[None]]] = None

        # Subscribed tickers for reconnection
        self._subscribed_contracts: List[Contract] = []

        # Setup logging
        self._logger = logging.getLogger(__name__)

        # Register IB event handlers
        self._setup_event_handlers()

    def _setup_event_handlers(self) -> None:
        """Setup IB event handlers for connection lifecycle."""
        self._ib.connectedEvent += self._handle_connected
        self._ib.disconnectedEvent += self._handle_disconnected
        self._ib.errorEvent += self._handle_error

    @property
    def ib(self) -> IB:
        """Get the underlying IB instance for direct access when needed."""
        return self._ib

    @property
    def state(self) -> ConnectionState:
        """Get current connection state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Check if client is currently connected."""
        return self._ib.isConnected() and self._state == ConnectionState.CONNECTED

    @property
    def client_id(self) -> int:
        """Get the client ID for this connection."""
        return self._client_id

    def set_on_connected(self, callback: Callable[[], Awaitable[None]]) -> None:
        """Set callback for connection established event."""
        self._on_connected = callback

    def set_on_disconnected(self, callback: Callable[[], Awaitable[None]]) -> None:
        """Set callback for disconnection event."""
        self._on_disconnected = callback

    def set_on_error(self, callback: Callable[[Exception], Awaitable[None]]) -> None:
        """Set callback for error events."""
        self._on_error = callback

    async def connect(self) -> bool:
        """
        Connect to IB Gateway/TWS asynchronously.

        Returns:
            True if connection successful, False otherwise.
        """
        if self._state == ConnectionState.CONNECTED:
            self._logger.warning("Already connected to IB")
            return True

        self._state = ConnectionState.CONNECTING
        self._logger.info(
            f"Connecting to IB at {self._host}:{self._port} "
            f"with client ID {self._client_id}"
        )

        try:
            await self._ib.connectAsync(
                host=self._host,
                port=self._port,
                clientId=self._client_id,
            )

            # Request real-time market data type (requires subscription)
            self._ib.reqMarketDataType(1)

            self._state = ConnectionState.CONNECTED
            self._reconnect_attempts = 0
            self._logger.info("Successfully connected to IB")

            return True

        except Exception as e:
            self._state = ConnectionState.ERROR
            self._logger.error(f"Failed to connect to IB: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from IB Gateway/TWS."""
        if self._state == ConnectionState.DISCONNECTED:
            self._logger.warning("Already disconnected from IB")
            return

        self._logger.info("Disconnecting from IB")

        try:
            self._ib.disconnect()
            self._state = ConnectionState.DISCONNECTED
            self._logger.info("Successfully disconnected from IB")
        except Exception as e:
            self._logger.error(f"Error during disconnect: {e}")
            raise

    async def reconnect(self) -> bool:
        """
        Attempt to reconnect with exponential backoff.

        Returns:
            True if reconnection successful, False if max attempts exceeded.
        """
        if self._reconnect_attempts >= self._max_reconnect_attempts:
            self._logger.error(
                f"Max reconnection attempts ({self._max_reconnect_attempts}) exceeded"
            )
            self._state = ConnectionState.ERROR
            return False

        self._state = ConnectionState.RECONNECTING
        self._reconnect_attempts += 1

        # Calculate delay with exponential backoff
        delay = self._base_delay * (2 ** (self._reconnect_attempts - 1))
        self._logger.info(
            f"Reconnection attempt {self._reconnect_attempts}/{self._max_reconnect_attempts} "
            f"in {delay:.1f} seconds"
        )

        await asyncio.sleep(delay)

        try:
            # Disconnect first if there's a stale connection
            if self._ib.isConnected():
                self._ib.disconnect()

            await self._ib.connectAsync(
                host=self._host,
                port=self._port,
                clientId=self._client_id,
            )

            # Request real-time market data type
            self._ib.reqMarketDataType(1)

            self._state = ConnectionState.CONNECTED
            self._reconnect_attempts = 0
            self._logger.info("Reconnection successful")

            # Re-subscribe to previously subscribed contracts
            await self._resubscribe_contracts()

            return True

        except Exception as e:
            self._logger.warning(f"Reconnection attempt failed: {e}")
            return await self.reconnect()

    async def _resubscribe_contracts(self) -> None:
        """Re-subscribe to all previously subscribed contracts after reconnection."""
        if not self._subscribed_contracts:
            return

        self._logger.info(
            f"Re-subscribing to {len(self._subscribed_contracts)} contracts"
        )

        for contract in self._subscribed_contracts:
            try:
                self._ib.reqMktData(contract)
                self._logger.debug(f"Re-subscribed to {contract}")
            except Exception as e:
                self._logger.error(f"Failed to re-subscribe to {contract}: {e}")

    async def subscribe_market_data(
        self,
        contract: Contract,
        generic_tick_list: str = "",
        snapshot: bool = False,
        regulatory_snapshot: bool = False,
    ) -> Ticker:
        """
        Subscribe to Level 1 market data for a contract.

        Args:
            contract: The contract to subscribe to
            generic_tick_list: Comma-separated list of generic tick types
            snapshot: Request snapshot instead of streaming data
            regulatory_snapshot: Request regulatory snapshot

        Returns:
            Ticker object with live market data
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to IB")

        self._logger.debug(f"Subscribing to market data for {contract}")

        ticker = self._ib.reqMktData(
            contract,
            genericTickList=generic_tick_list,
            snapshot=snapshot,
            regulatorySnapshot=regulatory_snapshot,
        )

        # Track subscription for reconnection
        if contract not in self._subscribed_contracts and not snapshot:
            self._subscribed_contracts.append(contract)

        return ticker

    async def subscribe_depth(
        self,
        contract: Contract,
        num_rows: int = 5,
        is_smart_depth: bool = False,
    ) -> Ticker:
        """
        Subscribe to Level 2 market depth data for a contract.

        Args:
            contract: The contract to subscribe to
            num_rows: Number of rows to request (default 5)
            is_smart_depth: Use SMART depth aggregation

        Returns:
            Ticker object with domBids and domAsks for order book
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to IB")

        self._logger.debug(f"Subscribing to market depth for {contract}")

        ticker = self._ib.reqMktDepth(
            contract,
            numRows=num_rows,
            isSmartDepth=is_smart_depth,
        )

        return ticker

    def cancel_market_data(self, contract: Contract) -> None:
        """
        Cancel market data subscription for a contract.

        Args:
            contract: The contract to unsubscribe from
        """
        if contract in self._subscribed_contracts:
            self._subscribed_contracts.remove(contract)

        self._ib.cancelMktData(contract)
        self._logger.debug(f"Cancelled market data for {contract}")

    def cancel_depth(self, contract: Contract) -> None:
        """
        Cancel market depth subscription for a contract.

        Args:
            contract: The contract to unsubscribe from
        """
        self._ib.cancelMktDepth(contract)
        self._logger.debug(f"Cancelled market depth for {contract}")

    def _handle_connected(self) -> None:
        """Handle IB connected event."""
        self._logger.info("IB connection established")
        if self._on_connected:
            asyncio.create_task(self._on_connected())

    def _handle_disconnected(self) -> None:
        """Handle IB disconnected event."""
        self._logger.warning("IB connection lost")
        previous_state = self._state
        self._state = ConnectionState.DISCONNECTED

        if self._on_disconnected:
            asyncio.create_task(self._on_disconnected())

        # Attempt automatic reconnection if we were previously connected
        if previous_state == ConnectionState.CONNECTED:
            self._logger.info("Attempting automatic reconnection")
            asyncio.create_task(self.reconnect())

    def _handle_error(self, req_id: int, error_code: int, error_string: str, contract: Contract) -> None:
        """Handle IB error events."""
        # Common non-critical error codes
        non_critical_codes = {
            2104,  # Market data farm connection is OK
            2106,  # HMDS data farm connection is OK
            2158,  # Sec-def data farm connection is OK
        }

        if error_code in non_critical_codes:
            self._logger.debug(f"IB info [{error_code}]: {error_string}")
            return

        self._logger.error(
            f"IB error [{error_code}] for request {req_id}: {error_string}"
        )

        if self._on_error:
            exception = Exception(f"IB Error {error_code}: {error_string}")
            asyncio.create_task(self._on_error(exception))

    async def request_positions(self) -> list:
        """
        Request all current positions.

        Returns:
            List of Position objects
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to IB")

        return self._ib.positions()

    async def request_account_summary(self) -> list:
        """
        Request account summary.

        Returns:
            List of AccountValue objects
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to IB")

        return self._ib.accountSummary()

    async def qualify_contracts(self, *contracts: Contract) -> List[Contract]:
        """
        Qualify contracts with IB to get full contract details.

        Args:
            contracts: Contracts to qualify

        Returns:
            List of qualified contracts
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to IB")

        return await self._ib.qualifyContractsAsync(*contracts)

    async def __aenter__(self) -> "IBClient":
        """Async context manager entry - connect to IB."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit - disconnect from IB."""
        await self.disconnect()
