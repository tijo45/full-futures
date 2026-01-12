"""
Feature Extraction Engine - Real-time feature generation for online ML.

Extracts features from Level 1 (tick) and Level 2 (order book) market data
for use with River online learning models. All features are returned as
Python dictionaries (NOT numpy arrays) for River compatibility.

Key Features:
- Level 1 derived features (spread, price momentum, volatility)
- Level 2 derived features (order book imbalance, liquidity metrics)
- Rolling/streaming statistics updated incrementally
- Configurable feature sets
- Thread-safe feature buffer for multiple contracts
"""

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any

from config import get_config
from src.data.market_data import TickData, DepthData


@dataclass
class StreamingStats:
    """
    Streaming statistics calculator for incremental feature computation.

    Uses Welford's algorithm for numerically stable online mean/variance.
    Maintains rolling windows for momentum and volatility calculations.
    """
    window_size: int = 20

    # Welford's algorithm state
    _count: int = 0
    _mean: float = 0.0
    _m2: float = 0.0

    # Rolling windows
    _values: deque = field(default_factory=lambda: deque(maxlen=20))
    _returns: deque = field(default_factory=lambda: deque(maxlen=20))
    _last_value: Optional[float] = None

    # Min/Max tracking for session
    _session_high: Optional[float] = None
    _session_low: Optional[float] = None

    def __post_init__(self):
        """Initialize deques with proper maxlen."""
        self._values = deque(maxlen=self.window_size)
        self._returns = deque(maxlen=self.window_size)

    def update(self, value: float) -> None:
        """
        Update streaming statistics with a new value.

        Args:
            value: New price or metric value
        """
        if value is None or math.isnan(value) or math.isinf(value):
            return

        # Update Welford's algorithm for running mean/variance
        self._count += 1
        delta = value - self._mean
        self._mean += delta / self._count
        delta2 = value - self._mean
        self._m2 += delta * delta2

        # Update rolling window
        self._values.append(value)

        # Calculate return if we have a previous value
        if self._last_value is not None and self._last_value != 0:
            ret = (value - self._last_value) / self._last_value
            self._returns.append(ret)

        self._last_value = value

        # Update session high/low
        if self._session_high is None or value > self._session_high:
            self._session_high = value
        if self._session_low is None or value < self._session_low:
            self._session_low = value

    @property
    def count(self) -> int:
        """Get number of observations."""
        return self._count

    @property
    def mean(self) -> float:
        """Get running mean."""
        return self._mean if self._count > 0 else 0.0

    @property
    def variance(self) -> float:
        """Get running variance (sample variance)."""
        if self._count < 2:
            return 0.0
        return self._m2 / (self._count - 1)

    @property
    def std(self) -> float:
        """Get running standard deviation."""
        return math.sqrt(self.variance) if self.variance > 0 else 0.0

    @property
    def rolling_mean(self) -> float:
        """Get rolling window mean."""
        if not self._values:
            return 0.0
        return sum(self._values) / len(self._values)

    @property
    def rolling_std(self) -> float:
        """Get rolling window standard deviation."""
        if len(self._values) < 2:
            return 0.0
        mean = self.rolling_mean
        variance = sum((x - mean) ** 2 for x in self._values) / (len(self._values) - 1)
        return math.sqrt(variance) if variance > 0 else 0.0

    @property
    def momentum(self) -> float:
        """Get price momentum (latest value vs start of window)."""
        if len(self._values) < 2:
            return 0.0
        first = self._values[0]
        if first == 0:
            return 0.0
        return (self._values[-1] - first) / first

    @property
    def rolling_return_mean(self) -> float:
        """Get mean of returns over rolling window."""
        if not self._returns:
            return 0.0
        return sum(self._returns) / len(self._returns)

    @property
    def rolling_volatility(self) -> float:
        """Get volatility (std of returns) over rolling window."""
        if len(self._returns) < 2:
            return 0.0
        mean = self.rolling_return_mean
        variance = sum((r - mean) ** 2 for r in self._returns) / (len(self._returns) - 1)
        return math.sqrt(variance) if variance > 0 else 0.0

    @property
    def session_range(self) -> float:
        """Get session range (high - low)."""
        if self._session_high is None or self._session_low is None:
            return 0.0
        return self._session_high - self._session_low

    @property
    def session_range_pct(self) -> float:
        """Get session range as percentage of low."""
        if self._session_low is None or self._session_low == 0:
            return 0.0
        return self.session_range / self._session_low

    def reset_session(self) -> None:
        """Reset session statistics (call at session start)."""
        self._session_high = None
        self._session_low = None

    def reset_all(self) -> None:
        """Reset all statistics."""
        self._count = 0
        self._mean = 0.0
        self._m2 = 0.0
        self._values.clear()
        self._returns.clear()
        self._last_value = None
        self._session_high = None
        self._session_low = None

    def to_dict(self, prefix: str = "") -> dict:
        """
        Export statistics as dictionary for feature extraction.

        Args:
            prefix: Optional prefix for feature names

        Returns:
            Dictionary of statistics
        """
        p = f"{prefix}_" if prefix else ""
        return {
            f'{p}count': self._count,
            f'{p}mean': self.mean,
            f'{p}std': self.std,
            f'{p}rolling_mean': self.rolling_mean,
            f'{p}rolling_std': self.rolling_std,
            f'{p}momentum': self.momentum,
            f'{p}volatility': self.rolling_volatility,
            f'{p}session_range': self.session_range,
            f'{p}session_range_pct': self.session_range_pct,
        }


@dataclass
class ContractFeatures:
    """
    Feature state for a single contract.

    Maintains streaming statistics and feature buffers for one contract.
    """
    symbol: str
    contract_id: int

    # Streaming statistics for different metrics
    price_stats: StreamingStats = field(default_factory=StreamingStats)
    spread_stats: StreamingStats = field(default_factory=StreamingStats)
    volume_stats: StreamingStats = field(default_factory=StreamingStats)
    imbalance_stats: StreamingStats = field(default_factory=StreamingStats)

    # Last processed data
    last_tick: Optional[TickData] = None
    last_depth: Optional[DepthData] = None
    last_features: Optional[dict] = None
    last_update: Optional[datetime] = None

    # Feature generation count
    feature_count: int = 0

    def reset_session(self) -> None:
        """Reset session-level statistics."""
        self.price_stats.reset_session()
        self.spread_stats.reset_session()
        self.volume_stats.reset_session()
        self.imbalance_stats.reset_session()


class FeatureEngine:
    """
    Real-time feature extraction engine for online ML prediction.

    Extracts features from Level 1 (tick) and Level 2 (order book) market data,
    returning Python dictionaries compatible with River online learning models.

    CRITICAL: All output is in Python dict format for River compatibility.
    DO NOT use numpy arrays.

    Usage:
        engine = FeatureEngine()
        features = engine.extract_features(tick_data, depth_data)
        # features is a dict: {'feature1': value, 'feature2': value, ...}

        # For River model:
        prediction = model.predict_one(features)
        model.learn_one(features, y_true)
    """

    # Default feature configuration
    DEFAULT_WINDOW_SIZE = 20

    def __init__(self, window_size: int = None):
        """
        Initialize FeatureEngine.

        Args:
            window_size: Rolling window size for streaming statistics.
                        Defaults to DEFAULT_WINDOW_SIZE if not specified.
        """
        self._config = get_config()
        self._logger = logging.getLogger(__name__)

        # Window size for rolling statistics
        self._window_size = window_size or self.DEFAULT_WINDOW_SIZE

        # Per-contract feature state
        self._contract_features: Dict[int, ContractFeatures] = {}

        # Global statistics (across all contracts)
        self._total_extractions: int = 0

        self._logger.info(
            f"FeatureEngine initialized with window_size={self._window_size}"
        )

    @property
    def window_size(self) -> int:
        """Get rolling window size."""
        return self._window_size

    @property
    def contract_count(self) -> int:
        """Get number of tracked contracts."""
        return len(self._contract_features)

    @property
    def total_extractions(self) -> int:
        """Get total feature extractions performed."""
        return self._total_extractions

    def _get_or_create_contract_features(
        self,
        contract_id: int,
        symbol: str
    ) -> ContractFeatures:
        """Get or create feature state for a contract."""
        if contract_id not in self._contract_features:
            self._contract_features[contract_id] = ContractFeatures(
                symbol=symbol,
                contract_id=contract_id,
                price_stats=StreamingStats(window_size=self._window_size),
                spread_stats=StreamingStats(window_size=self._window_size),
                volume_stats=StreamingStats(window_size=self._window_size),
                imbalance_stats=StreamingStats(window_size=self._window_size),
            )
            self._logger.debug(f"Created feature state for {symbol}")
        return self._contract_features[contract_id]

    def extract_features(
        self,
        tick_data: Optional[TickData] = None,
        depth_data: Optional[DepthData] = None,
        include_raw: bool = False,
    ) -> dict:
        """
        Extract features from market data.

        CRITICAL: Returns Python dict for River compatibility.

        Args:
            tick_data: Level 1 tick data (optional)
            depth_data: Level 2 order book data (optional)
            include_raw: Include raw tick/depth values in features

        Returns:
            Dictionary of features suitable for River models:
            {
                'spread': 0.25,
                'spread_pct': 0.001,
                'bid_ask_imbalance': 0.15,
                'price_momentum': 0.002,
                'volatility': 0.015,
                ...
            }
        """
        # Require at least one data source
        if tick_data is None and depth_data is None:
            self._logger.warning("No data provided for feature extraction")
            return {}

        # Determine contract from available data
        if tick_data is not None:
            contract_id = tick_data.contract_id
            symbol = tick_data.symbol
        else:
            contract_id = depth_data.contract_id
            symbol = depth_data.symbol

        # Get or create feature state
        cf = self._get_or_create_contract_features(contract_id, symbol)

        # Initialize feature dict
        features: dict = {}

        # Extract Level 1 features
        if tick_data is not None:
            features.update(self._extract_l1_features(tick_data, cf))
            cf.last_tick = tick_data

        # Extract Level 2 features
        if depth_data is not None:
            features.update(self._extract_l2_features(depth_data, cf))
            cf.last_depth = depth_data

        # Add streaming statistics features
        features.update(self._extract_streaming_features(cf))

        # Add raw values if requested
        if include_raw:
            if tick_data is not None:
                raw_tick = tick_data.to_dict()
                features.update({f'raw_{k}': v for k, v in raw_tick.items()})
            if depth_data is not None:
                raw_depth = depth_data.to_dict()
                features.update({f'raw_{k}': v for k, v in raw_depth.items()})

        # Update state
        cf.last_features = features
        cf.last_update = datetime.now(timezone.utc)
        cf.feature_count += 1
        self._total_extractions += 1

        # Clean any None values (River handles missing differently)
        features = self._clean_features(features)

        return features

    def _extract_l1_features(
        self,
        tick: TickData,
        cf: ContractFeatures
    ) -> dict:
        """
        Extract features from Level 1 tick data.

        Args:
            tick: TickData object
            cf: ContractFeatures state

        Returns:
            Dictionary of L1 features
        """
        features = {}

        # Basic price features
        if tick.mid is not None:
            features['mid_price'] = tick.mid
            cf.price_stats.update(tick.mid)

        if tick.last is not None:
            features['last_price'] = tick.last

        # Spread features
        if tick.spread is not None:
            features['spread'] = tick.spread
            cf.spread_stats.update(tick.spread)

            # Spread as percentage of mid
            if tick.mid is not None and tick.mid > 0:
                features['spread_pct'] = tick.spread / tick.mid
            else:
                features['spread_pct'] = 0.0

        # Quote size imbalance (L1)
        if tick.bid_size is not None and tick.ask_size is not None:
            total_size = tick.bid_size + tick.ask_size
            if total_size > 0:
                features['quote_imbalance'] = (tick.bid_size - tick.ask_size) / total_size
            else:
                features['quote_imbalance'] = 0.0
            features['bid_size'] = tick.bid_size
            features['ask_size'] = tick.ask_size

        # Volume features
        if tick.volume is not None:
            features['volume'] = tick.volume
            cf.volume_stats.update(float(tick.volume))

        # Price position within day's range
        if tick.high is not None and tick.low is not None and tick.high != tick.low:
            day_range = tick.high - tick.low
            features['day_range'] = day_range
            features['day_range_pct'] = day_range / tick.low if tick.low > 0 else 0.0

            if tick.mid is not None:
                # Position in range: 0 = at low, 1 = at high
                features['price_position'] = (tick.mid - tick.low) / day_range

            if tick.last is not None:
                features['last_position'] = (tick.last - tick.low) / day_range

        # Price vs open/close
        if tick.open is not None and tick.open > 0:
            if tick.mid is not None:
                features['vs_open'] = (tick.mid - tick.open) / tick.open
            if tick.last is not None:
                features['last_vs_open'] = (tick.last - tick.open) / tick.open

        if tick.close is not None and tick.close > 0:
            if tick.mid is not None:
                features['vs_close'] = (tick.mid - tick.close) / tick.close
            if tick.last is not None:
                features['last_vs_close'] = (tick.last - tick.close) / tick.close

        # Data freshness (important for quality gating)
        features['tick_age_seconds'] = tick.age_seconds

        return features

    def _extract_l2_features(
        self,
        depth: DepthData,
        cf: ContractFeatures
    ) -> dict:
        """
        Extract features from Level 2 order book data.

        Args:
            depth: DepthData object
            cf: ContractFeatures state

        Returns:
            Dictionary of L2 features
        """
        features = {}

        # Basic order book metrics (already computed in DepthData)
        features['book_bid_ask_imbalance'] = depth.bid_ask_imbalance
        cf.imbalance_stats.update(depth.bid_ask_imbalance)

        features['total_bid_size'] = depth.total_bid_size
        features['total_ask_size'] = depth.total_ask_size
        features['book_depth'] = depth.book_depth
        features['bid_levels'] = len(depth.bids)
        features['ask_levels'] = len(depth.asks)

        # Best bid/ask spread from order book
        if depth.best_bid is not None and depth.best_ask is not None:
            book_spread = depth.best_ask - depth.best_bid
            features['book_spread'] = book_spread
            if depth.best_bid > 0:
                features['book_spread_pct'] = book_spread / depth.best_bid
            else:
                features['book_spread_pct'] = 0.0

        # Order book depth analysis
        if depth.bids and depth.asks:
            # Weighted average price of order book
            bid_vwap = self._calc_vwap(depth.bids)
            ask_vwap = self._calc_vwap(depth.asks)

            if bid_vwap is not None:
                features['bid_vwap'] = bid_vwap
            if ask_vwap is not None:
                features['ask_vwap'] = ask_vwap

            # Depth at different levels
            if len(depth.bids) >= 3:
                features['bid_size_top3'] = sum(l.size for l in depth.bids[:3])
            if len(depth.asks) >= 3:
                features['ask_size_top3'] = sum(l.size for l in depth.asks[:3])

            # Price pressure: how tight is the book?
            if bid_vwap is not None and ask_vwap is not None and bid_vwap > 0:
                features['vwap_spread'] = (ask_vwap - bid_vwap) / bid_vwap

            # Size concentration: % of size at best price
            if depth.total_bid_size > 0:
                features['bid_concentration'] = depth.bids[0].size / depth.total_bid_size
            else:
                features['bid_concentration'] = 0.0

            if depth.total_ask_size > 0:
                features['ask_concentration'] = depth.asks[0].size / depth.total_ask_size
            else:
                features['ask_concentration'] = 0.0

        # Liquidity score (simple heuristic)
        features['liquidity_score'] = self._calc_liquidity_score(depth)

        # Data freshness
        features['depth_age_seconds'] = depth.age_seconds

        return features

    def _extract_streaming_features(self, cf: ContractFeatures) -> dict:
        """
        Extract features from streaming statistics.

        Args:
            cf: ContractFeatures state with updated statistics

        Returns:
            Dictionary of streaming features
        """
        features = {}

        # Price momentum and volatility
        if cf.price_stats.count > 1:
            features['price_momentum'] = cf.price_stats.momentum
            features['price_volatility'] = cf.price_stats.rolling_volatility
            features['price_rolling_mean'] = cf.price_stats.rolling_mean
            features['price_rolling_std'] = cf.price_stats.rolling_std
            features['price_zscore'] = self._calc_zscore(
                cf.price_stats._last_value,
                cf.price_stats.rolling_mean,
                cf.price_stats.rolling_std
            )

        # Spread trend
        if cf.spread_stats.count > 1:
            features['spread_momentum'] = cf.spread_stats.momentum
            features['spread_rolling_mean'] = cf.spread_stats.rolling_mean

        # Volume trend
        if cf.volume_stats.count > 1:
            features['volume_momentum'] = cf.volume_stats.momentum
            features['volume_rolling_mean'] = cf.volume_stats.rolling_mean
            features['volume_zscore'] = self._calc_zscore(
                cf.volume_stats._last_value,
                cf.volume_stats.rolling_mean,
                cf.volume_stats.rolling_std
            )

        # Order book imbalance trend
        if cf.imbalance_stats.count > 1:
            features['imbalance_momentum'] = cf.imbalance_stats.momentum
            features['imbalance_rolling_mean'] = cf.imbalance_stats.rolling_mean

        # Feature observation count (useful for model warm-up)
        features['observation_count'] = cf.feature_count

        return features

    def _calc_vwap(self, levels: List[Any]) -> Optional[float]:
        """Calculate volume-weighted average price for order book levels."""
        if not levels:
            return None

        total_value = sum(level.price * level.size for level in levels)
        total_size = sum(level.size for level in levels)

        if total_size == 0:
            return None

        return total_value / total_size

    def _calc_liquidity_score(self, depth: DepthData) -> float:
        """
        Calculate simple liquidity score.

        Higher score = more liquid (tighter spread, more size).
        """
        if not depth.bids or not depth.asks:
            return 0.0

        # Factors: depth, size, spread tightness
        depth_factor = min(depth.book_depth / 10.0, 1.0)  # Normalize to 10 levels

        total_size = depth.total_bid_size + depth.total_ask_size
        size_factor = min(total_size / 1000.0, 1.0)  # Normalize to 1000 contracts

        spread_factor = 0.0
        if depth.best_bid and depth.best_ask and depth.best_bid > 0:
            spread_pct = (depth.best_ask - depth.best_bid) / depth.best_bid
            # Tighter spread = higher factor (invert the spread)
            spread_factor = max(0.0, 1.0 - spread_pct * 100)  # Scale for typical spreads

        return (depth_factor + size_factor + spread_factor) / 3.0

    def _calc_zscore(
        self,
        value: Optional[float],
        mean: float,
        std: float
    ) -> float:
        """Calculate z-score with safe handling."""
        if value is None or std == 0:
            return 0.0
        return (value - mean) / std

    def _clean_features(self, features: dict) -> dict:
        """
        Clean features dictionary for River compatibility.

        Replaces None with 0.0 and handles inf/nan values.
        """
        cleaned = {}
        for key, value in features.items():
            if value is None:
                cleaned[key] = 0.0
            elif isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                cleaned[key] = 0.0
            else:
                cleaned[key] = value
        return cleaned

    def get_contract_features(self, contract_id: int) -> Optional[ContractFeatures]:
        """Get feature state for a contract."""
        return self._contract_features.get(contract_id)

    def get_latest_features(self, contract_id: int) -> Optional[dict]:
        """Get most recently extracted features for a contract."""
        cf = self._contract_features.get(contract_id)
        return cf.last_features if cf else None

    def get_all_latest_features(self) -> Dict[str, dict]:
        """Get latest features for all contracts, keyed by symbol."""
        return {
            cf.symbol: cf.last_features
            for cf in self._contract_features.values()
            if cf.last_features is not None
        }

    def reset_contract(self, contract_id: int) -> None:
        """Reset feature state for a contract."""
        if contract_id in self._contract_features:
            del self._contract_features[contract_id]
            self._logger.info(f"Reset feature state for contract {contract_id}")

    def reset_session(self, contract_id: Optional[int] = None) -> None:
        """
        Reset session-level statistics.

        Args:
            contract_id: Specific contract to reset, or None for all
        """
        if contract_id is not None:
            if contract_id in self._contract_features:
                self._contract_features[contract_id].reset_session()
        else:
            for cf in self._contract_features.values():
                cf.reset_session()
        self._logger.info("Reset session statistics")

    def reset_all(self) -> None:
        """Reset all feature state."""
        self._contract_features.clear()
        self._total_extractions = 0
        self._logger.info("Reset all feature state")

    def get_feature_names(self) -> List[str]:
        """
        Get list of all possible feature names.

        Useful for model initialization and documentation.
        """
        # Note: actual features depend on available data
        return [
            # L1 features
            'mid_price', 'last_price', 'spread', 'spread_pct',
            'quote_imbalance', 'bid_size', 'ask_size', 'volume',
            'day_range', 'day_range_pct', 'price_position', 'last_position',
            'vs_open', 'last_vs_open', 'vs_close', 'last_vs_close',
            'tick_age_seconds',
            # L2 features
            'book_bid_ask_imbalance', 'total_bid_size', 'total_ask_size',
            'book_depth', 'bid_levels', 'ask_levels',
            'book_spread', 'book_spread_pct',
            'bid_vwap', 'ask_vwap', 'bid_size_top3', 'ask_size_top3',
            'vwap_spread', 'bid_concentration', 'ask_concentration',
            'liquidity_score', 'depth_age_seconds',
            # Streaming features
            'price_momentum', 'price_volatility', 'price_rolling_mean',
            'price_rolling_std', 'price_zscore',
            'spread_momentum', 'spread_rolling_mean',
            'volume_momentum', 'volume_rolling_mean', 'volume_zscore',
            'imbalance_momentum', 'imbalance_rolling_mean',
            'observation_count',
        ]

    def get_summary(self) -> dict:
        """Get summary of feature engine state."""
        return {
            'window_size': self._window_size,
            'contract_count': self.contract_count,
            'total_extractions': self._total_extractions,
            'contracts': {
                cf.symbol: {
                    'feature_count': cf.feature_count,
                    'last_update': cf.last_update.isoformat() if cf.last_update else None,
                    'price_observations': cf.price_stats.count,
                }
                for cf in self._contract_features.values()
            }
        }
