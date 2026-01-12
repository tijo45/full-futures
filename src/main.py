"""
Main Orchestrator - Application entry point and component coordination.

Integrates all trading system components into a unified autonomous trading bot:
- IB connection management via IBClient
- Automatic contract discovery on startup
- Real-time market data subscription (Level 1 and Level 2)
- Prediction generation with confidence scoring
- Confidence-gated order execution
- Continuous online learning from trade outcomes
- Session and calendar awareness
- Health monitoring with trading halt/resume
- Position and risk management

Entry point: python src/main.py

Architecture:
    Main Orchestrator coordinates:
    - IBClient (connection to TWS/Gateway)
    - ContractDiscovery (auto-discover futures)
    - SessionManager (exchange calendars)
    - HealthMonitor (connectivity/data watchdog)
    - MarketDataHandler (Level 1 + Level 2 data)
    - FeatureEngine (real-time feature extraction)
    - Predictor (River-based online ML)
    - ConfidenceTracker (dynamic threshold management)
    - Executor (confidence-gated order execution)
    - PositionManager (authoritative position tracking)
    - RiskManager (Sharpe optimization, drawdown protection)
    - Dashboard data provider integration
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime, timezone
from typing import Optional, List

from config import get_config

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


class TradingBot:
    """
    Main orchestrator for the autonomous futures trading system.

    Coordinates all components and manages the trading lifecycle:
    1. Connect to IB Gateway/TWS
    2. Discover tradeable futures contracts
    3. Subscribe to market data feeds
    4. Run prediction loop with confidence gating
    5. Execute trades when confidence thresholds met
    6. Learn from trade outcomes
    7. Monitor health and manage risk

    Usage:
        bot = TradingBot()
        await bot.start()
        # Bot runs until stopped
        await bot.stop()
    """

    def __init__(self):
        """Initialize the trading bot with all components."""
        self._config = get_config()
        self._setup_logging()
        self._logger = logging.getLogger(__name__)

        # Running state
        self._running = False
        self._shutdown_event = asyncio.Event()

        # Initialize all components
        self._init_components()

        # Wire up component callbacks
        self._setup_callbacks()

        self._logger.info("TradingBot initialized")

    def _setup_logging(self) -> None:
        """Configure logging based on config."""
        logging.basicConfig(
            level=getattr(logging, self._config.LOG_LEVEL),
            format=self._config.LOG_FORMAT,
            handlers=[
                logging.StreamHandler(sys.stdout),
            ]
        )

    def _init_components(self) -> None:
        """Initialize all trading system components."""
        self._logger.info("Initializing components...")

        # Core: IB Client
        self._ib_client = IBClient()

        # Core: Contract Discovery
        self._contract_discovery = ContractDiscovery()

        # Core: Session Manager
        self._session_manager = SessionManager()

        # Core: Health Monitor
        self._health_monitor = HealthMonitor(
            restoration_seconds=30,
            check_interval_seconds=self._config.HEALTH_CHECK_INTERVAL_SECONDS,
        )

        # Data: Market Data Handler (set IB client after init)
        self._market_data_handler: Optional[MarketDataHandler] = None

        # Data: Feature Engine
        self._feature_engine = FeatureEngine()

        # Learning: Online Learner (used within Predictor)
        self._online_learner = OnlineLearner()

        # Learning: Drift Detector
        self._drift_detector = DriftDetector()

        # Trading: Predictor (integrates feature engine + learner)
        self._predictor = Predictor(
            feature_engine=self._feature_engine,
            use_multi_contract=True,  # Separate model per contract
        )

        # Trading: Confidence Tracker
        self._confidence_tracker = ConfidenceTracker()

        # Trading: Position Manager
        self._position_manager = PositionManager()

        # Trading: Risk Manager
        self._risk_manager = RiskManager()

        # Trading: Executor (set dependencies after init)
        self._executor = Executor()

        # Active contracts for trading
        self._active_contracts: List[DiscoveredContract] = []

        # Prediction loop task
        self._prediction_task: Optional[asyncio.Task] = None

        # Reconciliation task
        self._reconciliation_task: Optional[asyncio.Task] = None

        self._logger.info("All components initialized")

    def _setup_callbacks(self) -> None:
        """Wire up callbacks between components."""
        self._logger.info("Setting up component callbacks...")

        # IB Client callbacks
        self._ib_client.set_on_connected(self._on_ib_connected)
        self._ib_client.set_on_disconnected(self._on_ib_disconnected)
        self._ib_client.set_on_error(self._on_ib_error)

        # Health Monitor callbacks
        self._health_monitor.set_on_health_change(self._on_health_change)
        self._health_monitor.set_on_trading_gate_change(self._on_trading_gate_change)

        # Predictor drift callback
        self._predictor.register_drift_callback(self._on_drift_detected)

        # Risk Manager callbacks
        self._risk_manager.register_halt_callback(self._on_trading_halt)
        self._risk_manager.register_drawdown_callback(self._on_drawdown_alert)

        # Executor fill callback (for learning)
        self._executor.register_fill_callback(self._on_order_filled)

        self._logger.info("Callbacks configured")

    async def start(self) -> None:
        """
        Start the trading bot.

        Performs the startup sequence:
        1. Connect to IB
        2. Discover contracts
        3. Subscribe to market data
        4. Start health monitoring
        5. Start prediction loop
        """
        if self._running:
            self._logger.warning("Trading bot already running")
            return

        self._running = True
        self._shutdown_event.clear()

        self._logger.info("=" * 60)
        self._logger.info("Starting Autonomous Futures Trading Bot")
        self._logger.info("=" * 60)

        try:
            # Step 1: Connect to IB
            await self._connect_to_ib()

            # Step 2: Discover contracts
            await self._discover_contracts()

            # Step 3: Initialize market data handler
            self._market_data_handler = MarketDataHandler(self._ib_client)
            self._market_data_handler.set_on_tick(self._on_tick_update)
            self._market_data_handler.set_on_stale(self._on_stale_data)

            # Step 4: Subscribe to market data
            await self._subscribe_market_data()

            # Step 5: Configure health monitor with dependencies
            self._health_monitor.set_ib_client(self._ib_client)
            self._health_monitor.set_market_data_handler(self._market_data_handler)

            # Step 6: Configure executor with dependencies
            self._executor.set_ib_client(self._ib_client)
            self._executor.set_confidence_tracker(self._confidence_tracker)
            self._executor.set_session_manager(self._session_manager)

            # Step 7: Start health monitoring
            await self._health_monitor.start()

            # Step 8: Start staleness monitoring
            await self._market_data_handler.start_staleness_monitoring()

            # Step 9: Start prediction loop
            self._prediction_task = asyncio.create_task(self._prediction_loop())

            # Step 10: Start periodic reconciliation
            self._reconciliation_task = asyncio.create_task(self._reconciliation_loop())

            # Step 11: Setup dashboard data provider if available
            self._setup_dashboard_provider()

            self._logger.info("=" * 60)
            self._logger.info("Trading bot started successfully")
            self._logger.info("=" * 60)

            # Wait for shutdown signal
            await self._shutdown_event.wait()

        except Exception as e:
            self._logger.error(f"Error during startup: {e}")
            raise
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Stop the trading bot gracefully."""
        if not self._running:
            return

        self._logger.info("Stopping trading bot...")
        self._running = False

        # Cancel tasks
        if self._prediction_task:
            self._prediction_task.cancel()
            try:
                await self._prediction_task
            except asyncio.CancelledError:
                pass

        if self._reconciliation_task:
            self._reconciliation_task.cancel()
            try:
                await self._reconciliation_task
            except asyncio.CancelledError:
                pass

        # Stop health monitoring
        await self._health_monitor.stop()

        # Stop staleness monitoring
        if self._market_data_handler:
            await self._market_data_handler.stop_staleness_monitoring()

        # Cancel all pending orders
        if self._executor:
            await self._executor.cancel_all_orders()

        # Disconnect from IB
        await self._ib_client.disconnect()

        self._logger.info("Trading bot stopped")

    async def _connect_to_ib(self) -> None:
        """Connect to Interactive Brokers."""
        self._logger.info(
            f"Connecting to IB at {self._config.IB_HOST}:{self._config.IB_PORT}"
        )

        await self._ib_client.connect()

        if not self._ib_client.is_connected:
            raise ConnectionError("Failed to connect to IB")

        self._logger.info("Successfully connected to IB")

    async def _discover_contracts(self) -> None:
        """Discover tradeable futures contracts."""
        self._logger.info("Discovering futures contracts...")

        self._contract_discovery.set_ib_client(self._ib_client)
        contracts = await self._contract_discovery.discover_all()

        if not contracts:
            self._logger.warning("No contracts discovered - check IB subscription")
            return

        # Use front-month contracts for trading
        front_months = self._contract_discovery.front_month_contracts
        self._active_contracts = list(front_months.values())

        self._logger.info(
            f"Discovered {len(contracts)} contracts, "
            f"using {len(self._active_contracts)} front-month contracts"
        )

        # Log discovered contracts
        for dc in self._active_contracts[:10]:  # Log first 10
            self._logger.info(f"  - {dc.symbol} ({dc.exchange}): {dc.local_symbol}")

    async def _subscribe_market_data(self) -> None:
        """Subscribe to market data for active contracts."""
        self._logger.info(f"Subscribing to market data for {len(self._active_contracts)} contracts...")

        for dc in self._active_contracts:
            try:
                # Subscribe to both Level 1 and Level 2 data
                await self._market_data_handler.subscribe_all(
                    dc.contract,
                    num_rows=5,  # 5 levels of order book depth
                )

                # Set contract multiplier for position manager
                self._position_manager.set_contract_multiplier(
                    dc.contract.conId,
                    dc.multiplier,
                )

                self._logger.debug(f"Subscribed to {dc.symbol}")

            except Exception as e:
                self._logger.error(f"Failed to subscribe to {dc.symbol}: {e}")

        self._logger.info(
            f"Market data subscriptions: "
            f"L1={self._market_data_handler.level_1_count}, "
            f"L2={self._market_data_handler.level_2_count}"
        )

    async def _prediction_loop(self) -> None:
        """
        Main prediction and trading loop.

        Continuously:
        1. Check if trading is allowed (health, session, risk)
        2. Get market data for each contract
        3. Generate predictions with confidence scores
        4. Execute trades when confidence threshold met
        5. Update risk and position tracking
        """
        self._logger.info("Starting prediction loop")

        prediction_interval = 1.0  # Predict every second

        while self._running:
            try:
                await self._run_prediction_cycle()
                await asyncio.sleep(prediction_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Error in prediction loop: {e}")
                await asyncio.sleep(prediction_interval)

        self._logger.info("Prediction loop stopped")

    async def _run_prediction_cycle(self) -> None:
        """Run a single prediction cycle for all contracts."""
        # Check trading gates
        if not self._can_trade():
            return

        # Iterate through active contracts
        for dc in self._active_contracts:
            try:
                await self._process_contract(dc)
            except Exception as e:
                self._logger.error(f"Error processing {dc.symbol}: {e}")

        # Periodically clean up stale pending predictions
        self._predictor.clear_stale_pending(max_age_seconds=3600)

    def _can_trade(self) -> bool:
        """Check if trading is currently allowed."""
        # Check health monitor
        if not self._health_monitor.is_trading_allowed:
            return False

        # Check risk manager
        if not self._risk_manager.can_trade():
            return False

        # Check executor
        if not self._executor.is_ready:
            return False

        return True

    async def _process_contract(self, dc: DiscoveredContract) -> None:
        """Process a single contract for prediction and potential execution."""
        contract = dc.contract
        contract_id = contract.conId
        symbol = dc.symbol
        exchange = dc.exchange

        # Check session
        if not self._session_manager.should_allow_entry(exchange):
            return

        # Get market data
        tick_data = self._market_data_handler.get_tick_data(contract)
        depth_data = self._market_data_handler.get_depth_data(contract)

        if tick_data is None:
            return

        # Generate prediction
        prediction = self._predictor.predict(tick_data, depth_data)

        if not prediction.is_valid:
            return

        # Skip neutral signals
        if prediction.signal == PredictionSignal.NEUTRAL:
            return

        # Calculate confidence level for gating
        confidence_level = self._confidence_tracker.calculate_confidence(
            model_confidence=prediction.confidence,
            data_quality=1.0 - (prediction.data_age_seconds / 30.0),  # Scale by staleness
            regime_stability=self._get_regime_stability(str(contract_id)),
            recent_accuracy=self._get_recent_accuracy(contract_id),
        )

        # Check confidence gate
        gate_decision = self._confidence_tracker.gate_entry(
            confidence_level,
            contract_id=contract_id,
        )

        if gate_decision != GateDecision.ALLOWED:
            return

        # Check position limits
        current_position = self._position_manager.get_position_quantity(contract_id)
        if prediction.signal == PredictionSignal.BUY and current_position > 0:
            return  # Already long
        if prediction.signal == PredictionSignal.SELL and current_position < 0:
            return  # Already short

        # Calculate position size from risk manager
        contract_value = self._get_contract_value(tick_data, dc.multiplier)
        minutes_to_close = self._session_manager.minutes_to_close(exchange)

        position_size = self._risk_manager.calculate_position_size(
            contract_value=contract_value,
            confidence=confidence_level.value,
            volatility=0.02,  # TODO: Calculate from feature engine
            minutes_to_close=minutes_to_close,
        )

        if position_size <= 0:
            return

        # Execute trade
        result = await self._executor.execute_prediction(
            prediction=prediction,
            confidence=confidence_level,
            contract=contract,
            quantity=position_size,
        )

        if result.decision.value == "execute":
            self._logger.info(
                f"Trade executed: {prediction.signal.name} {position_size} {symbol} "
                f"(confidence={confidence_level.value:.3f})"
            )

    def _get_contract_value(self, tick_data: TickData, multiplier: float) -> float:
        """Calculate contract value from tick data."""
        price = tick_data.mid or tick_data.last or tick_data.bid or 0.0
        return price * multiplier

    def _get_regime_stability(self, contract_id: str) -> float:
        """Get regime stability factor for a contract."""
        # Get from drift detector if available
        if hasattr(self._predictor, '_multi_learner'):
            learner = self._predictor._multi_learner._learners.get(contract_id)
            if learner:
                drift_info = learner._drift.get_state()
                # Return inverse of drift activity
                return max(0.0, 1.0 - drift_info.get('drift_count', 0) * 0.1)
        return 1.0

    def _get_recent_accuracy(self, contract_id: int) -> float:
        """Get recent prediction accuracy for a contract."""
        stats = self._predictor.get_contract_summary(contract_id)
        if stats and 'learner' in stats:
            return stats['learner'].get('accuracy', 0.5)
        return 0.5

    async def _reconciliation_loop(self) -> None:
        """Periodic position reconciliation with IB."""
        self._logger.info("Starting reconciliation loop")

        reconciliation_interval = 60.0  # Reconcile every minute

        while self._running:
            try:
                if self._ib_client.is_connected:
                    report = await self._position_manager.reconcile_with_ib(
                        self._ib_client,
                        auto_correct=True,
                    )

                    if not report.is_clean:
                        self._logger.warning(
                            f"Position reconciliation found issues: "
                            f"mismatches={report.mismatches}, "
                            f"corrections={report.corrections_applied}"
                        )

                await asyncio.sleep(reconciliation_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Error in reconciliation loop: {e}")
                await asyncio.sleep(reconciliation_interval)

    def _setup_dashboard_provider(self) -> None:
        """Set up dashboard data provider for real-time visualization."""
        try:
            from src.dashboard.callbacks import (
                DashboardDataProvider,
                set_data_provider,
            )

            provider = DashboardDataProvider()
            provider.set_confidence_tracker(self._confidence_tracker)
            provider.set_position_manager(self._position_manager)
            provider.set_risk_manager(self._risk_manager)
            provider.set_health_monitor(self._health_monitor)
            provider.set_market_data_handler(self._market_data_handler)
            provider.set_executor(self._executor)
            provider.set_session_manager(self._session_manager)

            # Set as global provider
            set_data_provider(provider)

            self._logger.info("Dashboard data provider configured")

        except ImportError:
            self._logger.debug("Dashboard not available")
        except Exception as e:
            self._logger.warning(f"Failed to setup dashboard provider: {e}")

    # ==================== Callbacks ====================

    async def _on_ib_connected(self) -> None:
        """Handle IB connection established."""
        self._logger.info("IB connection callback: connected")

    async def _on_ib_disconnected(self) -> None:
        """Handle IB disconnection."""
        self._logger.warning("IB connection callback: disconnected")
        # Health monitor will detect this and close trading gate

    async def _on_ib_error(self, error: Exception) -> None:
        """Handle IB error."""
        self._logger.error(f"IB error callback: {error}")

    async def _on_health_change(
        self,
        old_state: HealthState,
        new_state: HealthState,
    ) -> None:
        """Handle health state change."""
        self._logger.info(f"Health state changed: {old_state.value} -> {new_state.value}")

        if new_state == HealthState.CRITICAL:
            self._logger.warning("CRITICAL: System health degraded - trading halted")

    async def _on_trading_gate_change(
        self,
        old_gate: TradingGate,
        new_gate: TradingGate,
    ) -> None:
        """Handle trading gate change."""
        self._logger.info(f"Trading gate changed: {old_gate.value} -> {new_gate.value}")

        if new_gate == TradingGate.CLOSED:
            self._logger.warning("Trading gate CLOSED")
        elif new_gate == TradingGate.OPEN:
            self._logger.info("Trading gate OPEN - trading resumed")

    async def _on_tick_update(self, tick_data: TickData) -> None:
        """Handle tick data update."""
        # Update unrealized P&L with current prices
        if tick_data.mid or tick_data.last:
            price = tick_data.mid or tick_data.last
            self._position_manager.update_market_prices({
                tick_data.contract_id: price,
            })

            # Update risk manager equity
            self._risk_manager.update_equity(
                self._risk_manager.current_equity +
                self._position_manager.total_unrealized_pnl
            )

    async def _on_stale_data(self, symbol: str, data_type) -> None:
        """Handle stale data detection."""
        self._logger.warning(f"Stale data detected for {symbol}: {data_type.value}")

    def _on_drift_detected(self, contract_id: str, drift_info: dict) -> None:
        """Handle model drift detection."""
        self._logger.warning(
            f"Drift detected for contract {contract_id}: {drift_info}"
        )

        # Increase confidence threshold temporarily
        self._confidence_tracker.set_threshold_mode('CAUTIOUS')

    def _on_trading_halt(self, reason: str) -> None:
        """Handle trading halt from risk manager."""
        self._logger.warning(f"Trading halted by risk manager: {reason}")

    def _on_drawdown_alert(self, state, drawdown_pct: float) -> None:
        """Handle drawdown alert."""
        self._logger.warning(
            f"Drawdown alert: state={state.value}, drawdown={drawdown_pct:.1%}"
        )

    async def _on_order_filled(self, result) -> None:
        """Handle order fill for learning feedback."""
        try:
            # Record fill in position manager
            if result.request and result.request.prediction:
                prediction = result.request.prediction

                # Determine outcome direction based on fill vs prediction
                # For now, we don't know the outcome yet - it will be determined
                # when the position is closed

                self._logger.info(
                    f"Order filled: prediction_id={prediction.prediction_id}, "
                    f"filled={result.filled_quantity} @ {result.average_fill_price:.2f}"
                )

        except Exception as e:
            self._logger.error(f"Error handling fill for learning: {e}")

    # ==================== Control Methods ====================

    def request_shutdown(self) -> None:
        """Request graceful shutdown."""
        self._logger.info("Shutdown requested")
        self._shutdown_event.set()

    def get_status(self) -> dict:
        """Get current bot status."""
        return {
            'running': self._running,
            'ib_connected': self._ib_client.is_connected if self._ib_client else False,
            'health_state': self._health_monitor.overall_state.value if self._health_monitor else 'unknown',
            'trading_gate': self._health_monitor.trading_gate.value if self._health_monitor else 'unknown',
            'active_contracts': len(self._active_contracts),
            'open_positions': len(self._position_manager.get_open_positions()),
            'total_exposure': self._position_manager.get_total_exposure(),
            'session_pnl': self._risk_manager.session_pnl,
            'risk_level': self._risk_manager.risk_level.value,
            'trading_state': self._risk_manager.trading_state.value,
        }


def main() -> None:
    """
    Application entry point.

    Creates and runs the trading bot with proper signal handling.
    """
    bot = TradingBot()

    # Setup signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        logging.info(f"Received signal {sig}, initiating shutdown...")
        bot.request_shutdown()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run the bot
    try:
        asyncio.run(bot.start())
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received")
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
