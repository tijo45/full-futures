"""
Contract Discovery Module - Auto-discover tradeable futures contracts.

Automatically discovers all tradeable futures contracts from IB on startup.
Uses Future() contracts (NOT ContFuture which is for historical data only).
Prioritizes contracts by volume and liquidity.

Key Features:
- Auto-discovery of futures contracts on startup
- Exchange-based discovery (CME, CBOT, COMEX, NYMEX)
- Volume/liquidity prioritization
- Front-month and near-month contract detection
- Configurable contract limits for IB API constraints (100 data lines)
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from enum import Enum

# Support both ib_async (preferred, Python 3.10+) and ib_insync (fallback)
try:
    from ib_async import Future, Contract, ContractDetails
    IB_LIBRARY = "ib_async"
except ImportError:
    try:
        from ib_insync import Future, Contract, ContractDetails
        IB_LIBRARY = "ib_insync"
    except ImportError:
        # Allow import for type checking even without IB library installed
        Future = Any  # type: ignore
        Contract = Any  # type: ignore
        ContractDetails = Any  # type: ignore
        IB_LIBRARY = None

if TYPE_CHECKING:
    from src.core.ib_client import IBClient


class DiscoveryStatus(Enum):
    """Contract discovery status enumeration."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class DiscoveredContract:
    """
    Represents a discovered futures contract with metadata.

    Attributes:
        contract: The qualified IB contract object
        symbol: Contract symbol (e.g., 'ES', 'NQ', 'CL')
        exchange: Exchange code (e.g., 'CME', 'NYMEX')
        expiry: Contract expiration date (YYYYMMDD format)
        local_symbol: Full local symbol (e.g., 'ESH4')
        multiplier: Contract multiplier (point value)
        min_tick: Minimum tick size
        volume_rank: Estimated volume ranking (lower is higher volume)
        is_front_month: Whether this is the front month contract
        discovered_at: Timestamp when contract was discovered
    """
    contract: Any  # IB Contract object
    symbol: str
    exchange: str
    expiry: str
    local_symbol: str = ""
    multiplier: float = 1.0
    min_tick: float = 0.01
    volume_rank: int = 0
    is_front_month: bool = False
    discovered_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        """Extract contract details if not provided."""
        if hasattr(self.contract, 'localSymbol') and not self.local_symbol:
            self.local_symbol = self.contract.localSymbol or ""
        if hasattr(self.contract, 'multiplier') and self.multiplier == 1.0:
            try:
                self.multiplier = float(self.contract.multiplier or 1.0)
            except (ValueError, TypeError):
                self.multiplier = 1.0


class ContractDiscovery:
    """
    Auto-discover tradeable futures contracts from Interactive Brokers.

    Discovers futures contracts on startup, prioritizes by volume/liquidity,
    and respects IB API limits (100 simultaneous data lines).

    Key Features:
    - Exchange-based discovery for major futures exchanges
    - Automatic front-month detection
    - Volume-based prioritization
    - Configurable maximum contracts
    - Async discovery with parallel requests

    Usage:
        discovery = ContractDiscovery(ib_client)
        contracts = await discovery.discover_all()
        front_months = await discovery.get_front_month_contracts()
    """

    # Default symbols to discover for each exchange
    # These are dynamically discoverable but provide good defaults
    DEFAULT_SYMBOLS: Dict[str, List[str]] = {
        "CME": ["ES", "NQ", "RTY", "MES", "MNQ", "M2K", "6E", "6J", "6B", "6A", "6C", "6S"],
        "CBOT": ["ZB", "ZN", "ZF", "ZT", "ZC", "ZS", "ZW", "ZM", "ZL", "YM"],
        "NYMEX": ["CL", "NG", "RB", "HO", "GC", "SI", "HG", "PL", "PA", "MCL"],
        "COMEX": ["GC", "SI", "HG", "MGC", "SIL"],
    }

    # Maximum contracts to discover per exchange (IB API limit consideration)
    DEFAULT_MAX_CONTRACTS_PER_EXCHANGE = 25

    # Maximum total contracts (IB API has 100 data line limit)
    DEFAULT_MAX_TOTAL_CONTRACTS = 50

    def __init__(
        self,
        ib_client: Optional["IBClient"] = None,
        symbols: Optional[Dict[str, List[str]]] = None,
        max_contracts_per_exchange: Optional[int] = None,
        max_total_contracts: Optional[int] = None,
    ):
        """
        Initialize ContractDiscovery.

        Args:
            ib_client: IBClient instance for IB API calls
            symbols: Custom symbol dictionary by exchange (default uses DEFAULT_SYMBOLS)
            max_contracts_per_exchange: Max contracts per exchange (default 25)
            max_total_contracts: Max total contracts across all exchanges (default 50)
        """
        self._ib_client = ib_client
        self._symbols = symbols or self.DEFAULT_SYMBOLS.copy()
        self._max_per_exchange = max_contracts_per_exchange or self.DEFAULT_MAX_CONTRACTS_PER_EXCHANGE
        self._max_total = max_total_contracts or self.DEFAULT_MAX_TOTAL_CONTRACTS

        # Discovery state
        self._status = DiscoveryStatus.PENDING
        self._discovered_contracts: List[DiscoveredContract] = []
        self._contracts_by_symbol: Dict[str, List[DiscoveredContract]] = {}
        self._front_months: Dict[str, DiscoveredContract] = {}

        # Logging
        self._logger = logging.getLogger(__name__)

    def set_ib_client(self, ib_client: "IBClient") -> None:
        """
        Set the IB client for discovery operations.

        Args:
            ib_client: IBClient instance for IB API calls
        """
        self._ib_client = ib_client

    @property
    def status(self) -> DiscoveryStatus:
        """Get current discovery status."""
        return self._status

    @property
    def discovered_contracts(self) -> List[DiscoveredContract]:
        """Get list of all discovered contracts."""
        return self._discovered_contracts.copy()

    @property
    def front_month_contracts(self) -> Dict[str, DiscoveredContract]:
        """Get dictionary of front month contracts by symbol."""
        return self._front_months.copy()

    async def discover_all(self) -> List[DiscoveredContract]:
        """
        Discover all tradeable futures contracts.

        Returns:
            List of discovered contracts, prioritized by volume/liquidity
        """
        if self._ib_client is None:
            raise ValueError("IB client not set. Call set_ib_client() first.")

        if not self._ib_client.is_connected:
            raise ConnectionError("IB client not connected")

        self._status = DiscoveryStatus.IN_PROGRESS
        self._logger.info("Starting contract discovery...")

        try:
            all_contracts: List[DiscoveredContract] = []

            # Discover contracts for each exchange in parallel
            discovery_tasks = []
            for exchange, symbols in self._symbols.items():
                task = self._discover_exchange(exchange, symbols)
                discovery_tasks.append(task)

            results = await asyncio.gather(*discovery_tasks, return_exceptions=True)

            # Collect results, handling any errors
            for i, result in enumerate(results):
                exchange = list(self._symbols.keys())[i]
                if isinstance(result, Exception):
                    self._logger.error(f"Discovery failed for {exchange}: {result}")
                elif result:
                    all_contracts.extend(result)

            # Sort by volume rank and limit to max total
            all_contracts.sort(key=lambda c: (c.volume_rank, c.symbol, c.expiry))
            self._discovered_contracts = all_contracts[:self._max_total]

            # Build lookup structures
            self._build_contract_indexes()

            self._status = DiscoveryStatus.COMPLETED
            self._logger.info(
                f"Contract discovery completed. Found {len(self._discovered_contracts)} contracts "
                f"across {len(self._front_months)} symbols"
            )

            return self._discovered_contracts

        except Exception as e:
            self._status = DiscoveryStatus.FAILED
            self._logger.error(f"Contract discovery failed: {e}")
            raise

    async def _discover_exchange(
        self,
        exchange: str,
        symbols: List[str]
    ) -> List[DiscoveredContract]:
        """
        Discover contracts for a single exchange.

        Args:
            exchange: Exchange code (e.g., 'CME')
            symbols: List of symbols to discover

        Returns:
            List of discovered contracts for this exchange
        """
        self._logger.debug(f"Discovering contracts on {exchange}")
        contracts: List[DiscoveredContract] = []

        for symbol in symbols:
            try:
                symbol_contracts = await self._discover_symbol(symbol, exchange)
                contracts.extend(symbol_contracts)

                if len(contracts) >= self._max_per_exchange:
                    self._logger.debug(
                        f"Reached max contracts ({self._max_per_exchange}) for {exchange}"
                    )
                    break

            except Exception as e:
                self._logger.warning(f"Failed to discover {symbol} on {exchange}: {e}")
                continue

        return contracts[:self._max_per_exchange]

    async def _discover_symbol(
        self,
        symbol: str,
        exchange: str
    ) -> List[DiscoveredContract]:
        """
        Discover all contracts for a single symbol.

        Args:
            symbol: Contract symbol (e.g., 'ES')
            exchange: Exchange code (e.g., 'CME')

        Returns:
            List of discovered contracts for this symbol
        """
        # IMPORTANT: Use Future() NOT ContFuture() for real-time data
        # ContFuture is for historical data ONLY
        futures_filter = Future(symbol=symbol, exchange=exchange)

        try:
            # Request contract details from IB
            contract_details_list = await self._ib_client.ib.reqContractDetailsAsync(futures_filter)

            if not contract_details_list:
                self._logger.debug(f"No contracts found for {symbol} on {exchange}")
                return []

            discovered = []
            current_date = datetime.now().strftime("%Y%m%d")

            for i, details in enumerate(contract_details_list):
                contract = details.contract

                # Skip expired contracts
                if hasattr(contract, 'lastTradeDateOrContractMonth'):
                    expiry = contract.lastTradeDateOrContractMonth
                    if expiry and expiry < current_date:
                        continue

                # Create a fully qualified Future contract for trading
                # IMPORTANT: Use Future() not ContFuture() for real-time data/orders
                qualified = Future(
                    symbol=contract.symbol,
                    exchange=contract.exchange,
                    lastTradeDateOrContractMonth=contract.lastTradeDateOrContractMonth,
                    currency=contract.currency or "USD",
                )

                # Qualify the contract with IB
                try:
                    qualified_list = await self._ib_client.qualify_contracts(qualified)
                    if qualified_list:
                        qualified = qualified_list[0]
                except Exception as e:
                    self._logger.warning(f"Failed to qualify {symbol} contract: {e}")

                discovered_contract = DiscoveredContract(
                    contract=qualified,
                    symbol=symbol,
                    exchange=exchange,
                    expiry=contract.lastTradeDateOrContractMonth or "",
                    local_symbol=contract.localSymbol or "",
                    multiplier=self._extract_multiplier(details),
                    min_tick=self._extract_min_tick(details),
                    volume_rank=i,  # Initial rank by position in response
                    is_front_month=(i == 0),  # First contract is typically front month
                )

                discovered.append(discovered_contract)

            self._logger.debug(f"Discovered {len(discovered)} contracts for {symbol}")
            return discovered

        except Exception as e:
            self._logger.error(f"Error discovering {symbol} on {exchange}: {e}")
            return []

    def _extract_multiplier(self, details: Any) -> float:
        """Extract contract multiplier from contract details."""
        try:
            if hasattr(details, 'contract') and hasattr(details.contract, 'multiplier'):
                return float(details.contract.multiplier or 1.0)
        except (ValueError, TypeError):
            pass
        return 1.0

    def _extract_min_tick(self, details: Any) -> float:
        """Extract minimum tick size from contract details."""
        try:
            if hasattr(details, 'minTick'):
                return float(details.minTick or 0.01)
        except (ValueError, TypeError):
            pass
        return 0.01

    def _build_contract_indexes(self) -> None:
        """Build lookup indexes for discovered contracts."""
        self._contracts_by_symbol = {}
        self._front_months = {}

        for contract in self._discovered_contracts:
            # Index by symbol
            if contract.symbol not in self._contracts_by_symbol:
                self._contracts_by_symbol[contract.symbol] = []
            self._contracts_by_symbol[contract.symbol].append(contract)

            # Track front month (first by expiry for each symbol)
            if contract.symbol not in self._front_months:
                self._front_months[contract.symbol] = contract

    def get_contracts_for_symbol(self, symbol: str) -> List[DiscoveredContract]:
        """
        Get all discovered contracts for a specific symbol.

        Args:
            symbol: Contract symbol (e.g., 'ES')

        Returns:
            List of contracts for the symbol, empty if not found
        """
        return self._contracts_by_symbol.get(symbol, [])

    def get_front_month(self, symbol: str) -> Optional[DiscoveredContract]:
        """
        Get the front month contract for a specific symbol.

        Args:
            symbol: Contract symbol (e.g., 'ES')

        Returns:
            Front month DiscoveredContract or None if not found
        """
        return self._front_months.get(symbol)

    def get_tradeable_contracts(
        self,
        min_volume_rank: Optional[int] = None,
        exchanges: Optional[List[str]] = None,
        front_month_only: bool = False,
    ) -> List[DiscoveredContract]:
        """
        Get filtered list of tradeable contracts.

        Args:
            min_volume_rank: Filter to contracts with volume rank <= this value
            exchanges: Filter to specific exchanges
            front_month_only: Only return front month contracts

        Returns:
            Filtered list of discoveredcontracts
        """
        contracts = self._discovered_contracts

        if front_month_only:
            contracts = list(self._front_months.values())

        if exchanges:
            contracts = [c for c in contracts if c.exchange in exchanges]

        if min_volume_rank is not None:
            contracts = [c for c in contracts if c.volume_rank <= min_volume_rank]

        return contracts

    def add_symbol(self, symbol: str, exchange: str) -> None:
        """
        Add a symbol to be discovered.

        Args:
            symbol: Contract symbol (e.g., 'ES')
            exchange: Exchange code (e.g., 'CME')
        """
        if exchange not in self._symbols:
            self._symbols[exchange] = []
        if symbol not in self._symbols[exchange]:
            self._symbols[exchange].append(symbol)
            self._logger.debug(f"Added {symbol} on {exchange} for discovery")

    def remove_symbol(self, symbol: str, exchange: str) -> None:
        """
        Remove a symbol from discovery.

        Args:
            symbol: Contract symbol (e.g., 'ES')
            exchange: Exchange code (e.g., 'CME')
        """
        if exchange in self._symbols and symbol in self._symbols[exchange]:
            self._symbols[exchange].remove(symbol)
            self._logger.debug(f"Removed {symbol} on {exchange} from discovery")

    async def refresh(self) -> List[DiscoveredContract]:
        """
        Refresh contract discovery (re-discover all contracts).

        Returns:
            Updated list of discovered contracts
        """
        self._logger.info("Refreshing contract discovery...")
        self._discovered_contracts = []
        self._contracts_by_symbol = {}
        self._front_months = {}
        self._status = DiscoveryStatus.PENDING

        return await self.discover_all()

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of discovered contracts.

        Returns:
            Dictionary with discovery statistics
        """
        return {
            "status": self._status.value,
            "total_contracts": len(self._discovered_contracts),
            "unique_symbols": len(self._contracts_by_symbol),
            "front_month_count": len(self._front_months),
            "exchanges": list(set(c.exchange for c in self._discovered_contracts)),
            "symbols": list(self._contracts_by_symbol.keys()),
            "front_months": {
                symbol: contract.local_symbol
                for symbol, contract in self._front_months.items()
            },
        }
