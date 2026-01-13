"""
Market Data Handler - Level 1 and Level 2 data subscriptions.

Manages market data subscriptions for futures contracts with real-time
tick freshness tracking for staleness detection. Supports both Level 1
(quotes) and Level 2 (order book depth) data streams.

Key Features:
- Level 1 data (bid, ask, last, volume) via reqMktData
- Level 2 data (order book) via reqMktDepth with domBids/domAsks
- Real-time tick freshness tracking for staleness detection
- Callbacks for data update notifications
- Automatic subscription management per contract
"""

import asyncio
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Dict, Callable, Awaitable, Any, List
from enum import Enum

# Support both ib_async (preferred, Python 3.10+) and ib_insync (fallback)
try:
    from ib_async import Contract, Ticker
except ImportError:
    try:
        from ib_insync import Contract, Ticker
    except ImportError:
        Contract = Any
        Ticker = Any

from config import get_config


class DataType(Enum):
    """Market data type enumeration."""
    LEVEL_1 = "level_1"
    LEVEL_2 = "level_2"


@dataclass
class TickData:
    """
    Level 1 tick data snapshot.

    Contains current quote and trade data for a contract.
    """
    contract_id: int
    symbol: str
    timestamp: datetime

    # Quote data
    bid: Optional[float] = None
    bid_size: Optional[int] = None
    ask: Optional[float] = None
    ask_size: Optional[int] = None

    # Trade data
    last: Optional[float] = None
    last_size: Optional[int] = None
    volume: Optional[int] = None

    # High/Low/Open/Close
    high: Optional[float] = None
    low: Optional[float] = None
    open: Optional[float] = None
    close: Optional[float] = None

    # Computed fields
    mid: Optional[float] = None
    spread: Optional[float] = None

    def __post_init__(self):
        """Compute derived fields."""
        if self.bid is not None and self.ask is not None:
            self.mid = (self.bid + self.ask) / 2
            self.spread = self.ask - self.bid

    @property
    def age_seconds(self) -> float:
        """Get age of this tick data in seconds."""
        return (datetime.now(timezone.utc) - self.timestamp).total_seconds()

    def is_stale(self, threshold_seconds: int) -> bool:
        """Check if this tick data is stale."""
        return self.age_seconds > threshold_seconds

    def to_dict(self) -> dict:
        """Convert to dictionary for feature extraction."""
        return {
            'contract_id': self.contract_id,
            'symbol': self.symbol,
            'bid': self.bid,
            'bid_size': self.bid_size,
            'ask': self.ask,
            'ask_size': self.ask_size,
            'last': self.last,
            'last_size': self.last_size,
            'volume': self.volume,
            'high': self.high,
            'low': self.low,
            'open': self.open,
            'close': self.close,
            'mid': self.mid,
            'spread': self.spread,
            'age_seconds': self.age_seconds,
        }


@dataclass
class OrderBookLevel:
    """Single level in the order book."""
    price: float
    size: int
    market_maker: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'price': self.price,
            'size': self.size,
            'market_maker': self.market_maker,
        }


@dataclass
class DepthData:
    """
    Level 2 order book data snapshot.

    Contains order book depth with bid and ask levels.
    """
    contract_id: int
    symbol: str
    timestamp: datetime

    # Order book
    bids: List[OrderBookLevel] = field(default_factory=list)
    asks: List[OrderBookLevel] = field(default_factory=list)

    @property
    def age_seconds(self) -> float:
        """Get age of this depth data in seconds."""
        return (datetime.now(timezone.utc) - self.timestamp).total_seconds()

    def is_stale(self, threshold_seconds: int) -> bool:
        """Check if this depth data is stale."""
        return self.age_seconds > threshold_seconds

    @property
    def best_bid(self) -> Optional[float]:
        """Get best bid price."""
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        """Get best ask price."""
        return self.asks[0].price if self.asks else None

    @property
    def total_bid_size(self) -> int:
        """Get total size on bid side."""
        return sum(level.size for level in self.bids)

    @property
    def total_ask_size(self) -> int:
        """Get total size on ask side."""
        return sum(level.size for level in self.asks)

    @property
    def bid_ask_imbalance(self) -> float:
        """Calculate bid/ask size imbalance (-1 to 1 scale)."""
        total_bid = self.total_bid_size
        total_ask = self.total_ask_size
        total = total_bid + total_ask
        if total == 0:
            return 0.0
        return (total_bid - total_ask) / total

    @property
    def book_depth(self) -> int:
        """Get total number of levels."""
        return len(self.bids) + len(self.asks)

    def to_dict(self) -> dict:
        """Convert to dictionary for feature extraction."""
        return {
            'contract_id': self.contract_id,
            'symbol': self.symbol,
            'best_bid': self.best_bid,
            'best_ask': self.best_ask,
            'total_bid_size': self.total_bid_size,
            'total_ask_size': self.total_ask_size,
            'bid_ask_imbalance': self.bid_ask_imbalance,
            'book_depth': self.book_depth,
            'bid_levels': len(self.bids),
            'ask_levels': len(self.asks),
            'age_seconds': self.age_seconds,
        }


@dataclass
class ContractSubscription:
    """Tracks subscription state for a contract."""
    contract: Contract
    symbol: str

    # Tickers from IB
    level_1_ticker: Optional[Ticker] = None
    level_2_ticker: Optional[Ticker] = None

    # Latest data snapshots
    tick_data: Optional[TickData] = None
    depth_data: Optional[DepthData] = None

    # Subscription status
    level_1_subscribed: bool = False
    level_2_subscribed: bool = False

    # Timestamps
    level_1_last_update: Optional[datetime] = None
    level_2_last_update: Optional[datetime] = None


class MarketDataHandler:
    """
    Market data subscription manager for Level 1 and Level 2 data.

    Manages subscriptions to market data feeds for futures contracts,
    tracks tick freshness for staleness detection, and provides
    callbacks for data update notifications.

    Usage:
        handler = MarketDataHandler(ib_client)
        await handler.subscribe_level_1(contract)
        await handler.subscribe_level_2(contract)
        tick = handler.get_tick_data(contract)
        depth = handler.get_depth_data(contract)
    """

    def __init__(self, ib_client: Any):
        """
        Initialize MarketDataHandler.

        Args:
            ib_client: IBClient instance for IB API access
        """
        self._ib_client = ib_client
        self._config = get_config()
        self._logger = logging.getLogger(__name__)

        # Subscriptions keyed by contract ID
        self._subscriptions: Dict[int, ContractSubscription] = {}

        # Callbacks for data updates
        self._on_tick: Optional[Callable[[TickData], Awaitable[None]]] = None
        self._on_depth: Optional[Callable[[DepthData], Awaitable[None]]] = None
        self._on_stale: Optional[Callable[[str, DataType], Awaitable[None]]] = None

        # Staleness threshold from config
        self._staleness_threshold = self._config.DATA_STALENESS_THRESHOLD_SECONDS

        # Running flag
        self._running = False
        self._update_task: Optional[asyncio.Task] = None

    @property
    def subscriptions(self) -> Dict[int, ContractSubscription]:
        """Get all current subscriptions."""
        return self._subscriptions

    @property
    def subscribed_contracts(self) -> List[Contract]:
        """Get list of all subscribed contracts."""
        return [sub.contract for sub in self._subscriptions.values()]

    @property
    def level_1_count(self) -> int:
        """Count of Level 1 subscriptions."""
        return sum(1 for sub in self._subscriptions.values() if sub.level_1_subscribed)

    @property
    def level_2_count(self) -> int:
        """Count of Level 2 subscriptions."""
        return sum(1 for sub in self._subscriptions.values() if sub.level_2_subscribed)

    def set_on_tick(self, callback: Callable[[TickData], Awaitable[None]]) -> None:
        """Set callback for Level 1 tick updates."""
        self._on_tick = callback

    def set_on_depth(self, callback: Callable[[DepthData], Awaitable[None]]) -> None:
        """Set callback for Level 2 depth updates."""
        self._on_depth = callback

    def set_on_stale(self, callback: Callable[[str, DataType], Awaitable[None]]) -> None:
        """Set callback for stale data detection."""
        self._on_stale = callback

    async def subscribe_level_1(
        self,
        contract: Contract,
        generic_tick_list: str = "",
    ) -> Optional[TickData]:
        """
        Subscribe to Level 1 market data for a contract.

        Args:
            contract: The contract to subscribe to
            generic_tick_list: Optional additional tick types

        Returns:
            Initial TickData snapshot or None if subscription failed
        """
        contract_id = contract.conId
        symbol = contract.symbol or str(contract_id)

        self._logger.info(f"Subscribing to Level 1 data for {symbol}")

        # Get or create subscription
        if contract_id not in self._subscriptions:
            self._subscriptions[contract_id] = ContractSubscription(
                contract=contract,
                symbol=symbol,
            )

        sub = self._subscriptions[contract_id]

        # Skip if already subscribed
        if sub.level_1_subscribed:
            self._logger.warning(f"Already subscribed to Level 1 for {symbol}")
            return sub.tick_data

        try:
            # Request market data from IB
            ticker = await self._ib_client.subscribe_market_data(
                contract,
                generic_tick_list=generic_tick_list,
            )

            sub.level_1_ticker = ticker
            sub.level_1_subscribed = True
            sub.level_1_last_update = datetime.now(timezone.utc)

            # Set up event handler for ticker updates
            ticker.updateEvent += lambda t: self._handle_ticker_update(contract_id, t)

            self._logger.info(f"Successfully subscribed to Level 1 for {symbol}")

            # Return initial snapshot
            return self._extract_tick_data(contract_id, ticker)

        except Exception as e:
            self._logger.error(f"Failed to subscribe to Level 1 for {symbol}: {e}")
            return None

    async def subscribe_level_2(
        self,
        contract: Contract,
        num_rows: int = 5,
        is_smart_depth: bool = False,
    ) -> Optional[DepthData]:
        """
        Subscribe to Level 2 market depth for a contract.

        Args:
            contract: The contract to subscribe to
            num_rows: Number of order book rows (default 5)
            is_smart_depth: Use SMART depth aggregation

        Returns:
            Initial DepthData snapshot or None if subscription failed
        """
        contract_id = contract.conId
        symbol = contract.symbol or str(contract_id)

        self._logger.info(f"Subscribing to Level 2 data for {symbol}")

        # Get or create subscription
        if contract_id not in self._subscriptions:
            self._subscriptions[contract_id] = ContractSubscription(
                contract=contract,
                symbol=symbol,
            )

        sub = self._subscriptions[contract_id]

        # Skip if already subscribed
        if sub.level_2_subscribed:
            self._logger.warning(f"Already subscribed to Level 2 for {symbol}")
            return sub.depth_data

        try:
            # Request market depth from IB
            ticker = await self._ib_client.subscribe_depth(
                contract,
                num_rows=num_rows,
                is_smart_depth=is_smart_depth,
            )

            sub.level_2_ticker = ticker
            sub.level_2_subscribed = True
            sub.level_2_last_update = datetime.now(timezone.utc)

            # Set up event handler for depth updates
            ticker.updateEvent += lambda t: self._handle_depth_update(contract_id, t)

            self._logger.info(f"Successfully subscribed to Level 2 for {symbol}")

            # Return initial snapshot
            return self._extract_depth_data(contract_id, ticker)

        except Exception as e:
            self._logger.error(f"Failed to subscribe to Level 2 for {symbol}: {e}")
            return None

    async def subscribe_all(
        self,
        contract: Contract,
        num_rows: int = 5,
    ) -> tuple:
        """
        Subscribe to both Level 1 and Level 2 data for a contract.

        Args:
            contract: The contract to subscribe to
            num_rows: Number of L2 order book rows

        Returns:
            Tuple of (TickData, DepthData) or (None, None) on failure
        """
        tick = await self.subscribe_level_1(contract)
        depth = await self.subscribe_level_2(contract, num_rows=num_rows)
        return tick, depth

    async def unsubscribe_level_1(self, contract: Contract) -> None:
        """Unsubscribe from Level 1 data for a contract."""
        contract_id = contract.conId

        if contract_id not in self._subscriptions:
            return

        sub = self._subscriptions[contract_id]
        symbol = sub.symbol

        if sub.level_1_subscribed:
            try:
                self._ib_client.cancel_market_data(contract)
                sub.level_1_subscribed = False
                sub.level_1_ticker = None
                self._logger.info(f"Unsubscribed from Level 1 for {symbol}")
            except Exception as e:
                self._logger.error(f"Error unsubscribing from Level 1 for {symbol}: {e}")

        # Clean up subscription if no data streams remain
        self._cleanup_subscription(contract_id)

    async def unsubscribe_level_2(self, contract: Contract) -> None:
        """Unsubscribe from Level 2 data for a contract."""
        contract_id = contract.conId

        if contract_id not in self._subscriptions:
            return

        sub = self._subscriptions[contract_id]
        symbol = sub.symbol

        if sub.level_2_subscribed:
            try:
                self._ib_client.cancel_depth(contract)
                sub.level_2_subscribed = False
                sub.level_2_ticker = None
                self._logger.info(f"Unsubscribed from Level 2 for {symbol}")
            except Exception as e:
                self._logger.error(f"Error unsubscribing from Level 2 for {symbol}: {e}")

        # Clean up subscription if no data streams remain
        self._cleanup_subscription(contract_id)

    async def unsubscribe_all(self, contract: Contract) -> None:
        """Unsubscribe from all data for a contract."""
        await self.unsubscribe_level_1(contract)
        await self.unsubscribe_level_2(contract)

    def _cleanup_subscription(self, contract_id: int) -> None:
        """Remove subscription if no data streams remain."""
        if contract_id not in self._subscriptions:
            return

        sub = self._subscriptions[contract_id]
        if not sub.level_1_subscribed and not sub.level_2_subscribed:
            del self._subscriptions[contract_id]

    def get_tick_data(self, contract: Contract) -> Optional[TickData]:
        """Get latest Level 1 tick data for a contract."""
        contract_id = contract.conId
        if contract_id not in self._subscriptions:
            return None
        return self._subscriptions[contract_id].tick_data

    def get_depth_data(self, contract: Contract) -> Optional[DepthData]:
        """Get latest Level 2 depth data for a contract."""
        contract_id = contract.conId
        if contract_id not in self._subscriptions:
            return None
        return self._subscriptions[contract_id].depth_data

    def get_all_tick_data(self) -> Dict[str, TickData]:
        """Get all current tick data keyed by symbol."""
        result = {}
        for sub in self._subscriptions.values():
            if sub.tick_data:
                result[sub.symbol] = sub.tick_data
        return result

    def get_all_depth_data(self) -> Dict[str, DepthData]:
        """Get all current depth data keyed by symbol."""
        result = {}
        for sub in self._subscriptions.values():
            if sub.depth_data:
                result[sub.symbol] = sub.depth_data
        return result

    def is_data_fresh(self, contract: Contract, data_type: DataType) -> bool:
        """
        Check if data for a contract is fresh (not stale).

        Args:
            contract: The contract to check
            data_type: LEVEL_1 or LEVEL_2

        Returns:
            True if data is fresh, False if stale or missing
        """
        contract_id = contract.conId
        if contract_id not in self._subscriptions:
            return False

        sub = self._subscriptions[contract_id]

        if data_type == DataType.LEVEL_1:
            if not sub.tick_data:
                return False
            return not sub.tick_data.is_stale(self._staleness_threshold)
        else:
            if not sub.depth_data:
                return False
            return not sub.depth_data.is_stale(self._staleness_threshold)

    def get_stale_contracts(self) -> List[tuple]:
        """
        Get list of contracts with stale data.

        Returns:
            List of (symbol, DataType) tuples for stale data
        """
        stale = []
        for sub in self._subscriptions.values():
            if sub.tick_data and sub.tick_data.is_stale(self._staleness_threshold):
                stale.append((sub.symbol, DataType.LEVEL_1))
            if sub.depth_data and sub.depth_data.is_stale(self._staleness_threshold):
                stale.append((sub.symbol, DataType.LEVEL_2))
        return stale

    def _handle_ticker_update(self, contract_id: int, ticker: Ticker) -> None:
        """Handle Level 1 ticker update event."""
        if contract_id not in self._subscriptions:
            return

        sub = self._subscriptions[contract_id]

        # Extract updated tick data
        tick_data = self._extract_tick_data(contract_id, ticker)
        sub.tick_data = tick_data
        sub.level_1_last_update = tick_data.timestamp

        # Invoke callback if set
        if self._on_tick:
            asyncio.create_task(self._on_tick(tick_data))

    def _handle_depth_update(self, contract_id: int, ticker: Ticker) -> None:
        """Handle Level 2 depth update event."""
        if contract_id not in self._subscriptions:
            return

        sub = self._subscriptions[contract_id]

        # Extract updated depth data
        depth_data = self._extract_depth_data(contract_id, ticker)
        sub.depth_data = depth_data
        sub.level_2_last_update = depth_data.timestamp

        # Invoke callback if set
        if self._on_depth:
            asyncio.create_task(self._on_depth(depth_data))

    def _safe_int(self, value: Any) -> Optional[int]:
        """Safely convert a value to int, returning None if NaN or invalid."""
        if value is None:
            return None
        try:
            if isinstance(value, float) and math.isnan(value):
                return None
            return int(value)
        except (ValueError, TypeError):
            return None

    def _extract_tick_data(self, contract_id: int, ticker: Ticker) -> TickData:
        """Extract TickData from IB Ticker object."""
        sub = self._subscriptions[contract_id]

        return TickData(
            contract_id=contract_id,
            symbol=sub.symbol,
            timestamp=datetime.now(timezone.utc),
            bid=ticker.bid if hasattr(ticker, 'bid') and ticker.bid else None,
            bid_size=self._safe_int(ticker.bidSize) if hasattr(ticker, 'bidSize') else None,
            ask=ticker.ask if hasattr(ticker, 'ask') and ticker.ask else None,
            ask_size=self._safe_int(ticker.askSize) if hasattr(ticker, 'askSize') else None,
            last=ticker.last if hasattr(ticker, 'last') and ticker.last else None,
            last_size=self._safe_int(ticker.lastSize) if hasattr(ticker, 'lastSize') else None,
            volume=self._safe_int(ticker.volume) if hasattr(ticker, 'volume') else None,
            high=ticker.high if hasattr(ticker, 'high') and ticker.high else None,
            low=ticker.low if hasattr(ticker, 'low') and ticker.low else None,
            open=ticker.open if hasattr(ticker, 'open') and ticker.open else None,
            close=ticker.close if hasattr(ticker, 'close') and ticker.close else None,
        )

    def _extract_depth_data(self, contract_id: int, ticker: Ticker) -> DepthData:
        """Extract DepthData from IB Ticker object with domBids/domAsks."""
        sub = self._subscriptions[contract_id]

        # Extract order book from domBids and domAsks
        bids = []
        asks = []

        # domBids and domAsks are lists of DOMLevel objects
        if hasattr(ticker, 'domBids') and ticker.domBids:
            for level in ticker.domBids:
                if level.price and level.size:
                    size = self._safe_int(level.size)
                    if size is not None:
                        bids.append(OrderBookLevel(
                            price=level.price,
                            size=size,
                            market_maker=level.marketMaker if hasattr(level, 'marketMaker') else "",
                        ))

        if hasattr(ticker, 'domAsks') and ticker.domAsks:
            for level in ticker.domAsks:
                if level.price and level.size:
                    size = self._safe_int(level.size)
                    if size is not None:
                        asks.append(OrderBookLevel(
                            price=level.price,
                            size=size,
                            market_maker=level.marketMaker if hasattr(level, 'marketMaker') else "",
                        ))

        return DepthData(
            contract_id=contract_id,
            symbol=sub.symbol,
            timestamp=datetime.now(timezone.utc),
            bids=bids,
            asks=asks,
        )

    async def start_staleness_monitoring(self, interval_seconds: float = 1.0) -> None:
        """
        Start background task to monitor data staleness.

        Args:
            interval_seconds: Check interval in seconds
        """
        if self._running:
            return

        self._running = True
        self._update_task = asyncio.create_task(
            self._staleness_monitor_loop(interval_seconds)
        )
        self._logger.info("Started staleness monitoring")

    async def stop_staleness_monitoring(self) -> None:
        """Stop the staleness monitoring task."""
        self._running = False
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
            self._update_task = None
        self._logger.info("Stopped staleness monitoring")

    async def _staleness_monitor_loop(self, interval: float) -> None:
        """Background loop to check for stale data."""
        while self._running:
            try:
                stale_contracts = self.get_stale_contracts()

                # Notify about stale data
                if stale_contracts and self._on_stale:
                    for symbol, data_type in stale_contracts:
                        await self._on_stale(symbol, data_type)

                await asyncio.sleep(interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Error in staleness monitor: {e}")
                await asyncio.sleep(interval)

    def get_summary(self) -> dict:
        """Get summary of current market data state."""
        stale = self.get_stale_contracts()
        return {
            'total_subscriptions': len(self._subscriptions),
            'level_1_count': self.level_1_count,
            'level_2_count': self.level_2_count,
            'stale_count': len(stale),
            'stale_contracts': [s[0] for s in stale],
            'staleness_threshold_seconds': self._staleness_threshold,
        }
