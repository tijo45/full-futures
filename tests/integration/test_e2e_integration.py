"""
End-to-End Integration Verification for Autonomous Futures Trading Bot.

This module provides comprehensive integration testing with TWS/Gateway connection.
Requires a running TWS or IB Gateway instance on port 7497.

Verification Steps:
1. Start TWS/Gateway on port 7497
2. Run: python tests/integration/test_e2e_integration.py
3. Verify all checks pass

Usage:
    # Full verification (requires TWS/Gateway)
    python tests/integration/test_e2e_integration.py

    # Component-only verification (no TWS required)
    python tests/integration/test_e2e_integration.py --components-only
"""

import asyncio
import logging
import sys
import argparse
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Add project root to path
sys.path.insert(0, '.')

from config import get_config


class VerificationStatus(Enum):
    """Verification step status."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class VerificationResult:
    """Result of a verification step."""
    name: str
    status: VerificationStatus
    message: str
    duration_ms: float = 0.0
    details: Optional[Dict] = None


class IntegrationVerifier:
    """
    End-to-end integration verifier for the trading system.

    Verifies all components work together correctly:
    - IB connection establishment
    - Contract discovery
    - Market data subscription and flow
    - Feature extraction
    - Prediction generation
    - Dashboard data provider integration
    """

    def __init__(self, skip_ib: bool = False):
        """
        Initialize the verifier.

        Args:
            skip_ib: Skip IB-dependent tests (for component-only verification)
        """
        self._skip_ib = skip_ib
        self._config = get_config()
        self._results: List[VerificationResult] = []
        self._logger = logging.getLogger(__name__)

        # Component references
        self._ib_client = None
        self._contract_discovery = None
        self._market_data_handler = None
        self._session_manager = None
        self._health_monitor = None
        self._feature_engine = None
        self._predictor = None
        self._confidence_tracker = None
        self._risk_manager = None
        self._position_manager = None
        self._executor = None
        self._dashboard_provider = None

        # Discovered contracts
        self._contracts = []

    async def run_all_verifications(self) -> bool:
        """
        Run all verification steps.

        Returns:
            True if all verifications passed, False otherwise.
        """
        self._logger.info("=" * 60)
        self._logger.info("Starting End-to-End Integration Verification")
        self._logger.info("=" * 60)

        # Step 1: Verify component imports
        await self._verify_component_imports()

        # Step 2: Verify component initialization
        await self._verify_component_initialization()

        if not self._skip_ib:
            # Step 3: Verify IB connection
            await self._verify_ib_connection()

            # Step 4: Verify contract discovery
            await self._verify_contract_discovery()

            # Step 5: Verify market data subscription
            await self._verify_market_data_subscription()

            # Step 6: Verify market data flow
            await self._verify_market_data_flow()
        else:
            self._results.append(VerificationResult(
                name="IB Connection",
                status=VerificationStatus.SKIPPED,
                message="Skipped (--components-only mode)"
            ))
            self._results.append(VerificationResult(
                name="Contract Discovery",
                status=VerificationStatus.SKIPPED,
                message="Skipped (--components-only mode)"
            ))
            self._results.append(VerificationResult(
                name="Market Data Subscription",
                status=VerificationStatus.SKIPPED,
                message="Skipped (--components-only mode)"
            ))
            self._results.append(VerificationResult(
                name="Market Data Flow",
                status=VerificationStatus.SKIPPED,
                message="Skipped (--components-only mode)"
            ))

        # Step 7: Verify feature engine
        await self._verify_feature_engine()

        # Step 8: Verify predictor
        await self._verify_predictor()

        # Step 9: Verify confidence tracker
        await self._verify_confidence_tracker()

        # Step 10: Verify risk manager
        await self._verify_risk_manager()

        # Step 11: Verify health monitor
        await self._verify_health_monitor()

        # Step 12: Verify dashboard integration
        await self._verify_dashboard_integration()

        # Cleanup
        await self._cleanup()

        # Print summary
        return self._print_summary()

    async def _verify_component_imports(self) -> None:
        """Verify all components can be imported."""
        start = datetime.now(timezone.utc)

        try:
            # Core components
            from src.core.ib_client import IBClient, ConnectionState
            from src.core.contract_discovery import ContractDiscovery, DiscoveredContract
            from src.core.session_manager import SessionManager
            from src.core.health_monitor import HealthMonitor, HealthState, TradingGate

            # Data components
            from src.data.market_data import MarketDataHandler, TickData, DepthData
            from src.data.feature_engine import FeatureEngine

            # Learning components
            from src.learning.online_learner import OnlineLearner
            from src.learning.drift_detector import DriftDetector

            # Trading components
            from src.trading.predictor import Predictor, PredictionResult, PredictionSignal
            from src.trading.confidence import ConfidenceTracker, ConfidenceLevel, GateDecision
            from src.trading.executor import Executor
            from src.trading.position_manager import PositionManager
            from src.trading.risk_manager import RiskManager, TradingState

            # Dashboard components
            from src.dashboard.app import create_app, run_dashboard
            from src.dashboard.callbacks import DashboardDataProvider, register_callbacks
            from src.dashboard.layouts import create_layout

            duration = (datetime.now(timezone.utc) - start).total_seconds() * 1000

            self._results.append(VerificationResult(
                name="Component Imports",
                status=VerificationStatus.PASSED,
                message="All components imported successfully",
                duration_ms=duration,
                details={"components": 16}
            ))

        except ImportError as e:
            duration = (datetime.now(timezone.utc) - start).total_seconds() * 1000
            self._results.append(VerificationResult(
                name="Component Imports",
                status=VerificationStatus.FAILED,
                message=f"Import failed: {e}",
                duration_ms=duration
            ))

    async def _verify_component_initialization(self) -> None:
        """Verify all components can be initialized."""
        start = datetime.now(timezone.utc)

        try:
            from src.core.ib_client import IBClient
            from src.core.contract_discovery import ContractDiscovery
            from src.core.session_manager import SessionManager
            from src.core.health_monitor import HealthMonitor
            from src.data.market_data import MarketDataHandler
            from src.data.feature_engine import FeatureEngine
            from src.learning.online_learner import OnlineLearner
            from src.learning.drift_detector import DriftDetector
            from src.trading.predictor import Predictor
            from src.trading.confidence import ConfidenceTracker
            from src.trading.executor import Executor
            from src.trading.position_manager import PositionManager
            from src.trading.risk_manager import RiskManager
            from src.dashboard.callbacks import DashboardDataProvider

            # Initialize all components
            self._ib_client = IBClient()
            self._contract_discovery = ContractDiscovery()
            self._session_manager = SessionManager()
            self._health_monitor = HealthMonitor()
            self._feature_engine = FeatureEngine()
            self._predictor = Predictor(
                feature_engine=self._feature_engine,
                use_multi_contract=True
            )
            self._confidence_tracker = ConfidenceTracker()
            self._executor = Executor()
            self._position_manager = PositionManager()
            self._risk_manager = RiskManager()
            self._dashboard_provider = DashboardDataProvider()

            duration = (datetime.now(timezone.utc) - start).total_seconds() * 1000

            self._results.append(VerificationResult(
                name="Component Initialization",
                status=VerificationStatus.PASSED,
                message="All components initialized successfully",
                duration_ms=duration,
                details={"components": 11}
            ))

        except Exception as e:
            duration = (datetime.now(timezone.utc) - start).total_seconds() * 1000
            self._results.append(VerificationResult(
                name="Component Initialization",
                status=VerificationStatus.FAILED,
                message=f"Initialization failed: {e}",
                duration_ms=duration
            ))

    async def _verify_ib_connection(self) -> None:
        """Verify IB connection can be established."""
        start = datetime.now(timezone.utc)

        try:
            self._logger.info(
                f"Connecting to IB at {self._config.IB_HOST}:{self._config.IB_PORT}"
            )

            await self._ib_client.connect()

            if self._ib_client.is_connected:
                duration = (datetime.now(timezone.utc) - start).total_seconds() * 1000
                self._results.append(VerificationResult(
                    name="IB Connection",
                    status=VerificationStatus.PASSED,
                    message=f"Connected to IB at {self._config.IB_HOST}:{self._config.IB_PORT}",
                    duration_ms=duration,
                    details={
                        "host": self._config.IB_HOST,
                        "port": self._config.IB_PORT,
                        "client_id": self._config.IB_CLIENT_ID
                    }
                ))
            else:
                duration = (datetime.now(timezone.utc) - start).total_seconds() * 1000
                self._results.append(VerificationResult(
                    name="IB Connection",
                    status=VerificationStatus.FAILED,
                    message="Connection established but client reports not connected",
                    duration_ms=duration
                ))

        except Exception as e:
            duration = (datetime.now(timezone.utc) - start).total_seconds() * 1000
            self._results.append(VerificationResult(
                name="IB Connection",
                status=VerificationStatus.FAILED,
                message=f"Connection failed: {e}",
                duration_ms=duration
            ))

    async def _verify_contract_discovery(self) -> None:
        """Verify contract discovery works."""
        start = datetime.now(timezone.utc)

        # Check if IB connection succeeded
        if not self._ib_client or not self._ib_client.is_connected:
            self._results.append(VerificationResult(
                name="Contract Discovery",
                status=VerificationStatus.SKIPPED,
                message="Skipped (IB not connected)"
            ))
            return

        try:
            self._contract_discovery.set_ib_client(self._ib_client)
            self._contracts = await self._contract_discovery.discover_all()

            if self._contracts:
                front_months = self._contract_discovery.front_month_contracts
                duration = (datetime.now(timezone.utc) - start).total_seconds() * 1000

                self._results.append(VerificationResult(
                    name="Contract Discovery",
                    status=VerificationStatus.PASSED,
                    message=f"Discovered {len(self._contracts)} contracts, {len(front_months)} front-month",
                    duration_ms=duration,
                    details={
                        "total_contracts": len(self._contracts),
                        "front_month_contracts": len(front_months),
                        "symbols": [c.symbol for c in list(front_months.values())[:5]]
                    }
                ))
            else:
                duration = (datetime.now(timezone.utc) - start).total_seconds() * 1000
                self._results.append(VerificationResult(
                    name="Contract Discovery",
                    status=VerificationStatus.FAILED,
                    message="No contracts discovered - check IB data subscription",
                    duration_ms=duration
                ))

        except Exception as e:
            duration = (datetime.now(timezone.utc) - start).total_seconds() * 1000
            self._results.append(VerificationResult(
                name="Contract Discovery",
                status=VerificationStatus.FAILED,
                message=f"Discovery failed: {e}",
                duration_ms=duration
            ))

    async def _verify_market_data_subscription(self) -> None:
        """Verify market data subscription works."""
        start = datetime.now(timezone.utc)

        # Check if we have contracts
        if not self._contracts:
            self._results.append(VerificationResult(
                name="Market Data Subscription",
                status=VerificationStatus.SKIPPED,
                message="Skipped (no contracts discovered)"
            ))
            return

        try:
            from src.data.market_data import MarketDataHandler

            self._market_data_handler = MarketDataHandler(self._ib_client)

            # Subscribe to first few front-month contracts
            front_months = list(self._contract_discovery.front_month_contracts.values())[:3]
            subscribed = 0

            for dc in front_months:
                try:
                    await self._market_data_handler.subscribe_all(
                        dc.contract,
                        num_rows=5
                    )
                    subscribed += 1
                except Exception as e:
                    self._logger.warning(f"Failed to subscribe to {dc.symbol}: {e}")

            if subscribed > 0:
                duration = (datetime.now(timezone.utc) - start).total_seconds() * 1000
                self._results.append(VerificationResult(
                    name="Market Data Subscription",
                    status=VerificationStatus.PASSED,
                    message=f"Subscribed to {subscribed} contracts",
                    duration_ms=duration,
                    details={
                        "l1_subscriptions": self._market_data_handler.level_1_count,
                        "l2_subscriptions": self._market_data_handler.level_2_count
                    }
                ))
            else:
                duration = (datetime.now(timezone.utc) - start).total_seconds() * 1000
                self._results.append(VerificationResult(
                    name="Market Data Subscription",
                    status=VerificationStatus.FAILED,
                    message="Failed to subscribe to any contracts",
                    duration_ms=duration
                ))

        except Exception as e:
            duration = (datetime.now(timezone.utc) - start).total_seconds() * 1000
            self._results.append(VerificationResult(
                name="Market Data Subscription",
                status=VerificationStatus.FAILED,
                message=f"Subscription failed: {e}",
                duration_ms=duration
            ))

    async def _verify_market_data_flow(self) -> None:
        """Verify market data is flowing."""
        start = datetime.now(timezone.utc)

        if not self._market_data_handler:
            self._results.append(VerificationResult(
                name="Market Data Flow",
                status=VerificationStatus.SKIPPED,
                message="Skipped (no market data handler)"
            ))
            return

        try:
            # Wait for data to flow
            await asyncio.sleep(3.0)

            # Check for tick data
            front_months = list(self._contract_discovery.front_month_contracts.values())[:3]
            data_received = 0

            for dc in front_months:
                tick_data = self._market_data_handler.get_tick_data(dc.contract)
                if tick_data and (tick_data.bid or tick_data.ask or tick_data.last):
                    data_received += 1

            if data_received > 0:
                duration = (datetime.now(timezone.utc) - start).total_seconds() * 1000
                self._results.append(VerificationResult(
                    name="Market Data Flow",
                    status=VerificationStatus.PASSED,
                    message=f"Received data for {data_received} contracts",
                    duration_ms=duration,
                    details={"contracts_with_data": data_received}
                ))
            else:
                duration = (datetime.now(timezone.utc) - start).total_seconds() * 1000
                self._results.append(VerificationResult(
                    name="Market Data Flow",
                    status=VerificationStatus.FAILED,
                    message="No market data received - check market hours and subscription",
                    duration_ms=duration
                ))

        except Exception as e:
            duration = (datetime.now(timezone.utc) - start).total_seconds() * 1000
            self._results.append(VerificationResult(
                name="Market Data Flow",
                status=VerificationStatus.FAILED,
                message=f"Data flow check failed: {e}",
                duration_ms=duration
            ))

    async def _verify_feature_engine(self) -> None:
        """Verify feature engine works with synthetic data."""
        start = datetime.now(timezone.utc)

        try:
            from src.data.market_data import TickData, DepthData, OrderBookLevel

            # Create synthetic tick data
            tick_data = TickData(
                contract_id=123,
                symbol="ES",
                bid=5000.25,
                ask=5000.50,
                last=5000.25,
                bid_size=100,
                ask_size=150,
                volume=50000,
                timestamp=datetime.now(timezone.utc)
            )

            # Create synthetic depth data with OrderBookLevel objects
            depth_data = DepthData(
                contract_id=123,
                symbol="ES",
                bids=[
                    OrderBookLevel(price=5000.25, size=100),
                    OrderBookLevel(price=5000.00, size=200),
                    OrderBookLevel(price=4999.75, size=300),
                ],
                asks=[
                    OrderBookLevel(price=5000.50, size=150),
                    OrderBookLevel(price=5000.75, size=250),
                    OrderBookLevel(price=5001.00, size=350),
                ],
                timestamp=datetime.now(timezone.utc)
            )

            # Extract features
            features = self._feature_engine.extract_features(
                tick_data=tick_data,
                depth_data=depth_data,
                symbol="ES"
            )

            if features and isinstance(features, dict) and len(features) > 0:
                duration = (datetime.now(timezone.utc) - start).total_seconds() * 1000
                self._results.append(VerificationResult(
                    name="Feature Engine",
                    status=VerificationStatus.PASSED,
                    message=f"Extracted {len(features)} features",
                    duration_ms=duration,
                    details={
                        "feature_count": len(features),
                        "sample_features": list(features.keys())[:5]
                    }
                ))
            else:
                duration = (datetime.now(timezone.utc) - start).total_seconds() * 1000
                self._results.append(VerificationResult(
                    name="Feature Engine",
                    status=VerificationStatus.FAILED,
                    message="No features extracted or invalid format",
                    duration_ms=duration
                ))

        except Exception as e:
            duration = (datetime.now(timezone.utc) - start).total_seconds() * 1000
            self._results.append(VerificationResult(
                name="Feature Engine",
                status=VerificationStatus.FAILED,
                message=f"Feature extraction failed: {e}",
                duration_ms=duration
            ))

    async def _verify_predictor(self) -> None:
        """Verify predictor works with synthetic data."""
        start = datetime.now(timezone.utc)

        try:
            from src.data.market_data import TickData, DepthData, OrderBookLevel
            from src.trading.predictor import PredictionSignal

            # Create synthetic tick data
            tick_data = TickData(
                contract_id=123,
                symbol="ES",
                bid=5000.25,
                ask=5000.50,
                last=5000.25,
                bid_size=100,
                ask_size=150,
                volume=50000,
                timestamp=datetime.now(timezone.utc)
            )

            # Create synthetic depth data with OrderBookLevel objects
            depth_data = DepthData(
                contract_id=123,
                symbol="ES",
                bids=[
                    OrderBookLevel(price=5000.25, size=100),
                    OrderBookLevel(price=5000.00, size=200),
                    OrderBookLevel(price=4999.75, size=300),
                ],
                asks=[
                    OrderBookLevel(price=5000.50, size=150),
                    OrderBookLevel(price=5000.75, size=250),
                    OrderBookLevel(price=5001.00, size=350),
                ],
                timestamp=datetime.now(timezone.utc)
            )

            # Generate prediction
            prediction = self._predictor.predict(tick_data, depth_data)

            if prediction and hasattr(prediction, 'signal') and hasattr(prediction, 'confidence'):
                duration = (datetime.now(timezone.utc) - start).total_seconds() * 1000
                self._results.append(VerificationResult(
                    name="Predictor",
                    status=VerificationStatus.PASSED,
                    message=f"Prediction generated: {prediction.signal.name}",
                    duration_ms=duration,
                    details={
                        "signal": prediction.signal.name,
                        "confidence": prediction.confidence,
                        "is_valid": prediction.is_valid
                    }
                ))
            else:
                duration = (datetime.now(timezone.utc) - start).total_seconds() * 1000
                self._results.append(VerificationResult(
                    name="Predictor",
                    status=VerificationStatus.FAILED,
                    message="Invalid prediction result",
                    duration_ms=duration
                ))

        except Exception as e:
            duration = (datetime.now(timezone.utc) - start).total_seconds() * 1000
            self._results.append(VerificationResult(
                name="Predictor",
                status=VerificationStatus.FAILED,
                message=f"Prediction failed: {e}",
                duration_ms=duration
            ))

    async def _verify_confidence_tracker(self) -> None:
        """Verify confidence tracker works."""
        start = datetime.now(timezone.utc)

        try:
            from src.trading.confidence import GateDecision

            # Calculate confidence
            confidence_level = self._confidence_tracker.calculate_confidence(
                model_confidence=0.75,
                data_quality=0.9,
                regime_stability=0.85,
                recent_accuracy=0.65
            )

            # Test gate decision
            gate_decision = self._confidence_tracker.gate_entry(
                confidence_level,
                contract_id=123
            )

            if confidence_level and hasattr(confidence_level, 'value'):
                duration = (datetime.now(timezone.utc) - start).total_seconds() * 1000
                self._results.append(VerificationResult(
                    name="Confidence Tracker",
                    status=VerificationStatus.PASSED,
                    message=f"Confidence calculated: {confidence_level.value:.3f}",
                    duration_ms=duration,
                    details={
                        "confidence_value": confidence_level.value,
                        "gate_decision": gate_decision.name,
                        "factors": confidence_level.factors
                    }
                ))
            else:
                duration = (datetime.now(timezone.utc) - start).total_seconds() * 1000
                self._results.append(VerificationResult(
                    name="Confidence Tracker",
                    status=VerificationStatus.FAILED,
                    message="Invalid confidence level result",
                    duration_ms=duration
                ))

        except Exception as e:
            duration = (datetime.now(timezone.utc) - start).total_seconds() * 1000
            self._results.append(VerificationResult(
                name="Confidence Tracker",
                status=VerificationStatus.FAILED,
                message=f"Confidence calculation failed: {e}",
                duration_ms=duration
            ))

    async def _verify_risk_manager(self) -> None:
        """Verify risk manager works."""
        start = datetime.now(timezone.utc)

        try:
            # Test position sizing
            position_size = self._risk_manager.calculate_position_size(
                contract_value=100000.0,
                confidence=0.75,
                volatility=0.02,
                minutes_to_close=60
            )

            # Test risk assessment
            assessment = self._risk_manager.get_risk_assessment()

            if position_size >= 0 and assessment:
                duration = (datetime.now(timezone.utc) - start).total_seconds() * 1000
                self._results.append(VerificationResult(
                    name="Risk Manager",
                    status=VerificationStatus.PASSED,
                    message=f"Position size: {position_size}, State: {self._risk_manager.trading_state.name}",
                    duration_ms=duration,
                    details={
                        "position_size": position_size,
                        "trading_state": self._risk_manager.trading_state.name,
                        "risk_level": self._risk_manager.risk_level.name,
                        "can_trade": self._risk_manager.can_trade()
                    }
                ))
            else:
                duration = (datetime.now(timezone.utc) - start).total_seconds() * 1000
                self._results.append(VerificationResult(
                    name="Risk Manager",
                    status=VerificationStatus.FAILED,
                    message="Invalid risk manager result",
                    duration_ms=duration
                ))

        except Exception as e:
            duration = (datetime.now(timezone.utc) - start).total_seconds() * 1000
            self._results.append(VerificationResult(
                name="Risk Manager",
                status=VerificationStatus.FAILED,
                message=f"Risk calculation failed: {e}",
                duration_ms=duration
            ))

    async def _verify_health_monitor(self) -> None:
        """Verify health monitor works."""
        start = datetime.now(timezone.utc)

        try:
            from src.core.health_monitor import HealthState, TradingGate

            # Get health status
            health_status = self._health_monitor.check_health()

            if health_status:
                duration = (datetime.now(timezone.utc) - start).total_seconds() * 1000
                self._results.append(VerificationResult(
                    name="Health Monitor",
                    status=VerificationStatus.PASSED,
                    message=f"Health: {self._health_monitor.overall_state.name}, Gate: {self._health_monitor.trading_gate.name}",
                    duration_ms=duration,
                    details={
                        "overall_state": self._health_monitor.overall_state.name,
                        "trading_gate": self._health_monitor.trading_gate.name,
                        "is_trading_allowed": self._health_monitor.is_trading_allowed
                    }
                ))
            else:
                duration = (datetime.now(timezone.utc) - start).total_seconds() * 1000
                self._results.append(VerificationResult(
                    name="Health Monitor",
                    status=VerificationStatus.FAILED,
                    message="Health check returned no status",
                    duration_ms=duration
                ))

        except Exception as e:
            duration = (datetime.now(timezone.utc) - start).total_seconds() * 1000
            self._results.append(VerificationResult(
                name="Health Monitor",
                status=VerificationStatus.FAILED,
                message=f"Health check failed: {e}",
                duration_ms=duration
            ))

    async def _verify_dashboard_integration(self) -> None:
        """Verify dashboard integration works."""
        start = datetime.now(timezone.utc)

        try:
            from src.dashboard.callbacks import set_data_provider

            # Configure data provider with components
            self._dashboard_provider.set_confidence_tracker(self._confidence_tracker)
            self._dashboard_provider.set_position_manager(self._position_manager)
            self._dashboard_provider.set_risk_manager(self._risk_manager)
            self._dashboard_provider.set_health_monitor(self._health_monitor)
            self._dashboard_provider.set_executor(self._executor)
            self._dashboard_provider.set_session_manager(self._session_manager)

            if self._market_data_handler:
                self._dashboard_provider.set_market_data_handler(self._market_data_handler)

            # Set as global provider
            set_data_provider(self._dashboard_provider)

            # Test data retrieval - using actual method names from DashboardDataProvider
            confidence_data = self._dashboard_provider.get_confidence_state()
            trading_data = self._dashboard_provider.get_trading_state()
            health_data = self._dashboard_provider.get_health_state()

            if confidence_data is not None and trading_data is not None and health_data is not None:
                duration = (datetime.now(timezone.utc) - start).total_seconds() * 1000
                self._results.append(VerificationResult(
                    name="Dashboard Integration",
                    status=VerificationStatus.PASSED,
                    message="Dashboard data provider configured and working",
                    duration_ms=duration,
                    details={
                        "has_confidence_data": confidence_data is not None,
                        "has_trading_data": trading_data is not None,
                        "has_health_data": health_data is not None
                    }
                ))
            else:
                duration = (datetime.now(timezone.utc) - start).total_seconds() * 1000
                self._results.append(VerificationResult(
                    name="Dashboard Integration",
                    status=VerificationStatus.FAILED,
                    message="Dashboard data provider not returning data",
                    duration_ms=duration
                ))

        except Exception as e:
            duration = (datetime.now(timezone.utc) - start).total_seconds() * 1000
            self._results.append(VerificationResult(
                name="Dashboard Integration",
                status=VerificationStatus.FAILED,
                message=f"Dashboard integration failed: {e}",
                duration_ms=duration
            ))

    async def _cleanup(self) -> None:
        """Clean up resources."""
        if self._market_data_handler:
            try:
                await self._market_data_handler.stop_staleness_monitoring()
            except Exception:
                pass

        if self._health_monitor:
            try:
                await self._health_monitor.stop()
            except Exception:
                pass

        if self._ib_client and self._ib_client.is_connected:
            try:
                await self._ib_client.disconnect()
            except Exception:
                pass

    def _print_summary(self) -> bool:
        """Print verification summary and return success status."""
        self._logger.info("")
        self._logger.info("=" * 60)
        self._logger.info("Verification Summary")
        self._logger.info("=" * 60)

        passed = 0
        failed = 0
        skipped = 0

        for result in self._results:
            status_symbol = {
                VerificationStatus.PASSED: "✓",
                VerificationStatus.FAILED: "✗",
                VerificationStatus.SKIPPED: "○"
            }.get(result.status, "?")

            status_color = {
                VerificationStatus.PASSED: "\033[92m",  # Green
                VerificationStatus.FAILED: "\033[91m",  # Red
                VerificationStatus.SKIPPED: "\033[93m"  # Yellow
            }.get(result.status, "")

            reset_color = "\033[0m"

            self._logger.info(
                f"  {status_color}{status_symbol}{reset_color} {result.name}: "
                f"{result.message} ({result.duration_ms:.0f}ms)"
            )

            if result.status == VerificationStatus.PASSED:
                passed += 1
            elif result.status == VerificationStatus.FAILED:
                failed += 1
            elif result.status == VerificationStatus.SKIPPED:
                skipped += 1

        self._logger.info("")
        self._logger.info(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
        self._logger.info("=" * 60)

        if failed == 0:
            self._logger.info("\033[92mAll verifications passed!\033[0m")
            return True
        else:
            self._logger.info("\033[91mSome verifications failed.\033[0m")
            return False


async def main(skip_ib: bool = False) -> bool:
    """Run the integration verification."""
    verifier = IntegrationVerifier(skip_ib=skip_ib)
    return await verifier.run_all_verifications()


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="End-to-end integration verification for the trading system"
    )
    parser.add_argument(
        "--components-only",
        action="store_true",
        help="Skip IB-dependent tests (for component verification without TWS)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    args = parser.parse_args()

    # Configure logging
    config = get_config()
    log_level = logging.DEBUG if args.verbose else getattr(logging, config.LOG_LEVEL)
    logging.basicConfig(
        level=log_level,
        format=config.LOG_FORMAT,
    )

    # Run verification
    success = asyncio.run(main(skip_ib=args.components_only))
    sys.exit(0 if success else 1)
