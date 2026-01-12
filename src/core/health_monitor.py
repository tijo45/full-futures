"""
Health Monitor - Connectivity, data freshness, and system stability watchdog.

Continuously monitors system health and halts trading on any degradation.
Only resumes trading when health is fully restored.

Key Features:
- IB connectivity monitoring with connection state tracking
- Data freshness monitoring (>30s = stale by default)
- System resource monitoring (CPU, memory)
- Trading halt/resume based on health state
- Callback system for health state changes
- Comprehensive health reporting for dashboard
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Dict, Callable, Awaitable, Any, List

# psutil is optional - system resource monitoring will be limited without it
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False

from config import get_config


class HealthState(Enum):
    """Overall system health state."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ComponentState(Enum):
    """Individual component health state."""
    OK = "ok"
    WARNING = "warning"
    ERROR = "error"
    UNKNOWN = "unknown"


class TradingGate(Enum):
    """Trading gate state - controls whether trading is allowed."""
    OPEN = "open"          # Trading allowed
    CLOSED = "closed"      # Trading halted due to health issues
    PENDING = "pending"    # Waiting for restoration confirmation


@dataclass
class ComponentHealth:
    """Health status for a single component."""
    name: str
    state: ComponentState
    message: str = ""
    last_check: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_ok(self) -> bool:
        """Check if component is healthy."""
        return self.state == ComponentState.OK

    @property
    def is_error(self) -> bool:
        """Check if component is in error state."""
        return self.state == ComponentState.ERROR

    def to_dict(self) -> dict:
        """Convert to dictionary for reporting."""
        return {
            'name': self.name,
            'state': self.state.value,
            'message': self.message,
            'last_check': self.last_check.isoformat(),
            'details': self.details,
        }


@dataclass
class SystemResources:
    """System resource usage snapshot."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    disk_percent: float
    disk_available_gb: float

    @classmethod
    def capture(cls) -> Optional['SystemResources']:
        """Capture current system resource usage."""
        if not PSUTIL_AVAILABLE:
            # Return None if psutil is not available
            return None

        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        return cls(
            timestamp=datetime.now(timezone.utc),
            cpu_percent=psutil.cpu_percent(interval=None),
            memory_percent=memory.percent,
            memory_available_mb=memory.available / (1024 * 1024),
            disk_percent=disk.percent,
            disk_available_gb=disk.free / (1024 * 1024 * 1024),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for reporting."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'memory_available_mb': self.memory_available_mb,
            'disk_percent': self.disk_percent,
            'disk_available_gb': self.disk_available_gb,
        }


@dataclass
class HealthReport:
    """Comprehensive health report for the system."""
    timestamp: datetime
    overall_state: HealthState
    trading_gate: TradingGate
    components: Dict[str, ComponentHealth]
    resources: Optional[SystemResources]
    issues: List[str]
    restoration_time: Optional[datetime] = None

    @property
    def is_trading_allowed(self) -> bool:
        """Check if trading is currently allowed."""
        return self.trading_gate == TradingGate.OPEN

    @property
    def critical_issues(self) -> List[str]:
        """Get list of critical issues."""
        return [
            c.message for c in self.components.values()
            if c.state == ComponentState.ERROR
        ]

    @property
    def warning_issues(self) -> List[str]:
        """Get list of warning issues."""
        return [
            c.message for c in self.components.values()
            if c.state == ComponentState.WARNING
        ]

    def to_dict(self) -> dict:
        """Convert to dictionary for dashboard/reporting."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'overall_state': self.overall_state.value,
            'trading_gate': self.trading_gate.value,
            'is_trading_allowed': self.is_trading_allowed,
            'components': {k: v.to_dict() for k, v in self.components.items()},
            'resources': self.resources.to_dict() if self.resources else None,
            'issues': self.issues,
            'critical_issues': self.critical_issues,
            'warning_issues': self.warning_issues,
            'restoration_time': self.restoration_time.isoformat() if self.restoration_time else None,
        }


class HealthMonitor:
    """
    System health monitor for connectivity, data freshness, and stability.

    Monitors all critical system components and halts trading when health
    degrades. Trading is only resumed after a configurable restoration
    period of sustained healthy state.

    Key Responsibilities:
    - Monitor IB connection state
    - Monitor market data freshness (detect stale data)
    - Monitor system resources (CPU, memory)
    - Control trading gate (halt/resume trading)
    - Provide health reports for dashboard

    Usage:
        monitor = HealthMonitor()
        monitor.set_ib_client(ib_client)
        monitor.set_market_data_handler(market_data_handler)
        await monitor.start()

        # Check if trading is allowed
        if monitor.is_trading_allowed:
            # Execute trades
            pass

        report = monitor.get_health_report()
    """

    # Component names for consistency
    COMPONENT_IB_CONNECTION = "ib_connection"
    COMPONENT_DATA_FRESHNESS = "data_freshness"
    COMPONENT_SYSTEM_RESOURCES = "system_resources"

    # Resource thresholds (as percentages)
    CPU_WARNING_THRESHOLD = 70.0
    CPU_CRITICAL_THRESHOLD = 90.0
    MEMORY_WARNING_THRESHOLD = 70.0
    MEMORY_CRITICAL_THRESHOLD = 90.0
    DISK_WARNING_THRESHOLD = 85.0
    DISK_CRITICAL_THRESHOLD = 95.0

    def __init__(
        self,
        restoration_seconds: int = 30,
        check_interval_seconds: Optional[int] = None,
    ):
        """
        Initialize HealthMonitor.

        Args:
            restoration_seconds: Seconds of sustained health before reopening trading gate
            check_interval_seconds: Health check interval (default from config)
        """
        self._config = get_config()
        self._logger = logging.getLogger(__name__)

        # Check interval from config or parameter
        self._check_interval = (
            check_interval_seconds or
            self._config.HEALTH_CHECK_INTERVAL_SECONDS
        )
        self._restoration_seconds = restoration_seconds

        # Component references (set externally)
        self._ib_client: Optional[Any] = None
        self._market_data_handler: Optional[Any] = None

        # Health state
        self._components: Dict[str, ComponentHealth] = {}
        self._resources: Optional[SystemResources] = None
        self._overall_state = HealthState.UNKNOWN
        self._trading_gate = TradingGate.PENDING
        self._issues: List[str] = []

        # Restoration tracking
        self._last_unhealthy_time: Optional[datetime] = None
        self._restoration_start: Optional[datetime] = None

        # Running state
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None

        # Callbacks
        self._on_health_change: Optional[Callable[[HealthState, HealthState], Awaitable[None]]] = None
        self._on_trading_gate_change: Optional[Callable[[TradingGate, TradingGate], Awaitable[None]]] = None
        self._on_component_change: Optional[Callable[[str, ComponentHealth], Awaitable[None]]] = None

        # Initialize components
        self._initialize_components()

    def _initialize_components(self) -> None:
        """Initialize component health tracking."""
        for name in [
            self.COMPONENT_IB_CONNECTION,
            self.COMPONENT_DATA_FRESHNESS,
            self.COMPONENT_SYSTEM_RESOURCES,
        ]:
            self._components[name] = ComponentHealth(
                name=name,
                state=ComponentState.UNKNOWN,
                message="Not yet checked",
            )

    @property
    def is_trading_allowed(self) -> bool:
        """Check if trading is currently allowed."""
        return self._trading_gate == TradingGate.OPEN

    @property
    def overall_state(self) -> HealthState:
        """Get current overall health state."""
        return self._overall_state

    @property
    def trading_gate(self) -> TradingGate:
        """Get current trading gate state."""
        return self._trading_gate

    def set_ib_client(self, ib_client: Any) -> None:
        """Set the IB client for connection monitoring."""
        self._ib_client = ib_client
        self._logger.info("IB client set for health monitoring")

    def set_market_data_handler(self, handler: Any) -> None:
        """Set the market data handler for freshness monitoring."""
        self._market_data_handler = handler
        self._logger.info("Market data handler set for health monitoring")

    def set_on_health_change(
        self,
        callback: Callable[[HealthState, HealthState], Awaitable[None]]
    ) -> None:
        """Set callback for overall health state changes."""
        self._on_health_change = callback

    def set_on_trading_gate_change(
        self,
        callback: Callable[[TradingGate, TradingGate], Awaitable[None]]
    ) -> None:
        """Set callback for trading gate state changes."""
        self._on_trading_gate_change = callback

    def set_on_component_change(
        self,
        callback: Callable[[str, ComponentHealth], Awaitable[None]]
    ) -> None:
        """Set callback for individual component state changes."""
        self._on_component_change = callback

    async def start(self) -> None:
        """Start the health monitoring loop."""
        if self._running:
            self._logger.warning("Health monitor already running")
            return

        self._running = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        self._logger.info(
            f"Health monitor started with {self._check_interval}s interval, "
            f"{self._restoration_seconds}s restoration period"
        )

    async def stop(self) -> None:
        """Stop the health monitoring loop."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
        self._logger.info("Health monitor stopped")

    async def check_health(self) -> HealthReport:
        """
        Perform a complete health check.

        Returns:
            HealthReport with current system health status
        """
        # Capture system resources
        self._resources = SystemResources.capture()

        # Check each component
        await self._check_ib_connection()
        await self._check_data_freshness()
        await self._check_system_resources()

        # Determine overall state
        old_state = self._overall_state
        self._determine_overall_state()

        # Trigger callback if state changed
        if old_state != self._overall_state and self._on_health_change:
            asyncio.create_task(self._on_health_change(old_state, self._overall_state))

        # Update trading gate
        await self._update_trading_gate()

        # Build report
        return self.get_health_report()

    def get_health_report(self) -> HealthReport:
        """Get the current health report."""
        return HealthReport(
            timestamp=datetime.now(timezone.utc),
            overall_state=self._overall_state,
            trading_gate=self._trading_gate,
            components=self._components.copy(),
            resources=self._resources,
            issues=self._issues.copy(),
            restoration_time=self._restoration_start,
        )

    def get_component_health(self, name: str) -> Optional[ComponentHealth]:
        """Get health status for a specific component."""
        return self._components.get(name)

    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            try:
                await self.check_health()
                await asyncio.sleep(self._check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(self._check_interval)

    async def _check_ib_connection(self) -> None:
        """Check IB connection health."""
        component = self._components[self.COMPONENT_IB_CONNECTION]
        old_state = component.state

        if self._ib_client is None:
            component.state = ComponentState.ERROR
            component.message = "IB client not configured"
            component.details = {'configured': False}
        elif not hasattr(self._ib_client, 'is_connected'):
            component.state = ComponentState.ERROR
            component.message = "IB client missing is_connected property"
            component.details = {'configured': True, 'valid': False}
        elif self._ib_client.is_connected:
            component.state = ComponentState.OK
            component.message = "Connected to IB"
            component.details = {
                'configured': True,
                'connected': True,
                'state': self._ib_client.state.value if hasattr(self._ib_client, 'state') else 'unknown',
            }
        else:
            component.state = ComponentState.ERROR
            component.message = "Disconnected from IB"
            component.details = {
                'configured': True,
                'connected': False,
                'state': self._ib_client.state.value if hasattr(self._ib_client, 'state') else 'disconnected',
            }

        component.last_check = datetime.now(timezone.utc)

        # Trigger callback if state changed
        if old_state != component.state and self._on_component_change:
            asyncio.create_task(self._on_component_change(component.name, component))

    async def _check_data_freshness(self) -> None:
        """Check market data freshness."""
        component = self._components[self.COMPONENT_DATA_FRESHNESS]
        old_state = component.state

        if self._market_data_handler is None:
            component.state = ComponentState.WARNING
            component.message = "Market data handler not configured"
            component.details = {'configured': False}
        elif not hasattr(self._market_data_handler, 'get_stale_contracts'):
            component.state = ComponentState.WARNING
            component.message = "Market data handler missing staleness check"
            component.details = {'configured': True, 'valid': False}
        else:
            stale_contracts = self._market_data_handler.get_stale_contracts()
            total_subscriptions = len(self._market_data_handler.subscriptions)

            if total_subscriptions == 0:
                # No subscriptions yet - warning but not error
                component.state = ComponentState.WARNING
                component.message = "No market data subscriptions"
                component.details = {
                    'configured': True,
                    'subscriptions': 0,
                    'stale_count': 0,
                }
            elif len(stale_contracts) == 0:
                component.state = ComponentState.OK
                component.message = f"All {total_subscriptions} data feeds fresh"
                component.details = {
                    'configured': True,
                    'subscriptions': total_subscriptions,
                    'stale_count': 0,
                    'stale_contracts': [],
                }
            elif len(stale_contracts) < total_subscriptions:
                # Some stale - warning
                component.state = ComponentState.WARNING
                stale_symbols = [s[0] for s in stale_contracts]
                component.message = f"{len(stale_contracts)}/{total_subscriptions} data feeds stale"
                component.details = {
                    'configured': True,
                    'subscriptions': total_subscriptions,
                    'stale_count': len(stale_contracts),
                    'stale_contracts': stale_symbols,
                }
            else:
                # All stale - critical
                component.state = ComponentState.ERROR
                stale_symbols = [s[0] for s in stale_contracts]
                component.message = "All market data feeds are stale"
                component.details = {
                    'configured': True,
                    'subscriptions': total_subscriptions,
                    'stale_count': len(stale_contracts),
                    'stale_contracts': stale_symbols,
                }

        component.last_check = datetime.now(timezone.utc)

        # Trigger callback if state changed
        if old_state != component.state and self._on_component_change:
            asyncio.create_task(self._on_component_change(component.name, component))

    async def _check_system_resources(self) -> None:
        """Check system resource usage."""
        component = self._components[self.COMPONENT_SYSTEM_RESOURCES]
        old_state = component.state

        if not PSUTIL_AVAILABLE:
            component.state = ComponentState.WARNING
            component.message = "psutil not installed - resource monitoring disabled"
            component.details = {'psutil_available': False}
        elif self._resources is None:
            component.state = ComponentState.UNKNOWN
            component.message = "Resource data not available"
            component.details = {'psutil_available': True}
        else:
            issues = []
            state = ComponentState.OK

            # Check CPU
            if self._resources.cpu_percent >= self.CPU_CRITICAL_THRESHOLD:
                issues.append(f"CPU critical: {self._resources.cpu_percent:.1f}%")
                state = ComponentState.ERROR
            elif self._resources.cpu_percent >= self.CPU_WARNING_THRESHOLD:
                issues.append(f"CPU high: {self._resources.cpu_percent:.1f}%")
                if state == ComponentState.OK:
                    state = ComponentState.WARNING

            # Check memory
            if self._resources.memory_percent >= self.MEMORY_CRITICAL_THRESHOLD:
                issues.append(f"Memory critical: {self._resources.memory_percent:.1f}%")
                state = ComponentState.ERROR
            elif self._resources.memory_percent >= self.MEMORY_WARNING_THRESHOLD:
                issues.append(f"Memory high: {self._resources.memory_percent:.1f}%")
                if state == ComponentState.OK:
                    state = ComponentState.WARNING

            # Check disk
            if self._resources.disk_percent >= self.DISK_CRITICAL_THRESHOLD:
                issues.append(f"Disk critical: {self._resources.disk_percent:.1f}%")
                state = ComponentState.ERROR
            elif self._resources.disk_percent >= self.DISK_WARNING_THRESHOLD:
                issues.append(f"Disk high: {self._resources.disk_percent:.1f}%")
                if state == ComponentState.OK:
                    state = ComponentState.WARNING

            component.state = state
            component.message = "; ".join(issues) if issues else "Resources OK"
            component.details = {
                'cpu_percent': self._resources.cpu_percent,
                'memory_percent': self._resources.memory_percent,
                'memory_available_mb': self._resources.memory_available_mb,
                'disk_percent': self._resources.disk_percent,
                'disk_available_gb': self._resources.disk_available_gb,
            }

        component.last_check = datetime.now(timezone.utc)

        # Trigger callback if state changed
        if old_state != component.state and self._on_component_change:
            asyncio.create_task(self._on_component_change(component.name, component))

    def _determine_overall_state(self) -> None:
        """Determine overall health state from component states."""
        self._issues = []

        # Count states
        error_count = 0
        warning_count = 0
        unknown_count = 0

        for component in self._components.values():
            if component.state == ComponentState.ERROR:
                error_count += 1
                self._issues.append(f"[ERROR] {component.name}: {component.message}")
            elif component.state == ComponentState.WARNING:
                warning_count += 1
                self._issues.append(f"[WARNING] {component.name}: {component.message}")
            elif component.state == ComponentState.UNKNOWN:
                unknown_count += 1
                self._issues.append(f"[UNKNOWN] {component.name}: {component.message}")

        # Determine overall state
        # IB connection error is always critical
        ib_component = self._components.get(self.COMPONENT_IB_CONNECTION)
        if ib_component and ib_component.state == ComponentState.ERROR:
            self._overall_state = HealthState.CRITICAL
        elif error_count > 0:
            self._overall_state = HealthState.CRITICAL
        elif warning_count > 0 or unknown_count > 0:
            self._overall_state = HealthState.DEGRADED
        else:
            self._overall_state = HealthState.HEALTHY

    async def _update_trading_gate(self) -> None:
        """Update trading gate based on health state."""
        old_gate = self._trading_gate
        now = datetime.now(timezone.utc)

        if self._overall_state == HealthState.CRITICAL:
            # Critical - immediately close trading gate
            self._trading_gate = TradingGate.CLOSED
            self._last_unhealthy_time = now
            self._restoration_start = None
            self._logger.warning("Trading halted due to critical health state")

        elif self._overall_state == HealthState.DEGRADED:
            # Degraded - close gate but less urgently
            if self._trading_gate == TradingGate.OPEN:
                self._trading_gate = TradingGate.CLOSED
                self._last_unhealthy_time = now
                self._restoration_start = None
                self._logger.warning("Trading halted due to degraded health state")

        elif self._overall_state == HealthState.HEALTHY:
            # Healthy - start or continue restoration period
            if self._trading_gate == TradingGate.CLOSED:
                # Start restoration period
                self._trading_gate = TradingGate.PENDING
                self._restoration_start = now
                self._logger.info(
                    f"Health restored, waiting {self._restoration_seconds}s before "
                    "reopening trading gate"
                )

            elif self._trading_gate == TradingGate.PENDING:
                # Check if restoration period complete
                if self._restoration_start:
                    elapsed = (now - self._restoration_start).total_seconds()
                    if elapsed >= self._restoration_seconds:
                        self._trading_gate = TradingGate.OPEN
                        self._restoration_start = None
                        self._logger.info("Trading gate reopened after restoration period")

        # Trigger callback if gate changed
        if old_gate != self._trading_gate and self._on_trading_gate_change:
            asyncio.create_task(self._on_trading_gate_change(old_gate, self._trading_gate))

    def force_trading_halt(self, reason: str = "Manual halt") -> None:
        """
        Force trading to halt immediately.

        Args:
            reason: Reason for the halt
        """
        old_gate = self._trading_gate
        self._trading_gate = TradingGate.CLOSED
        self._last_unhealthy_time = datetime.now(timezone.utc)
        self._restoration_start = None
        self._issues.append(f"[MANUAL] Trading halted: {reason}")
        self._logger.warning(f"Trading manually halted: {reason}")

        if old_gate != self._trading_gate and self._on_trading_gate_change:
            asyncio.create_task(self._on_trading_gate_change(old_gate, self._trading_gate))

    def force_trading_resume(self, reason: str = "Manual resume") -> None:
        """
        Force trading to resume immediately.

        WARNING: Use with caution - bypasses health checks.

        Args:
            reason: Reason for the resume
        """
        if self._overall_state == HealthState.CRITICAL:
            self._logger.error(
                "Cannot force resume trading while in critical state. "
                "Fix critical issues first."
            )
            return

        old_gate = self._trading_gate
        self._trading_gate = TradingGate.OPEN
        self._restoration_start = None
        self._logger.warning(f"Trading manually resumed: {reason}")

        if old_gate != self._trading_gate and self._on_trading_gate_change:
            asyncio.create_task(self._on_trading_gate_change(old_gate, self._trading_gate))

    def get_summary(self) -> dict:
        """Get health summary for dashboard."""
        return {
            'overall_state': self._overall_state.value,
            'trading_gate': self._trading_gate.value,
            'is_trading_allowed': self.is_trading_allowed,
            'check_interval_seconds': self._check_interval,
            'restoration_seconds': self._restoration_seconds,
            'components': {
                name: {
                    'state': comp.state.value,
                    'message': comp.message,
                    'is_ok': comp.is_ok,
                }
                for name, comp in self._components.items()
            },
            'resources': self._resources.to_dict() if self._resources else None,
            'issues_count': len(self._issues),
            'issues': self._issues[:5],  # Limit to 5 most recent
            'restoration_progress': self._get_restoration_progress(),
        }

    def _get_restoration_progress(self) -> Optional[dict]:
        """Get restoration progress if pending."""
        if self._trading_gate != TradingGate.PENDING or not self._restoration_start:
            return None

        now = datetime.now(timezone.utc)
        elapsed = (now - self._restoration_start).total_seconds()
        remaining = max(0, self._restoration_seconds - elapsed)
        progress = min(100, (elapsed / self._restoration_seconds) * 100)

        return {
            'elapsed_seconds': elapsed,
            'remaining_seconds': remaining,
            'progress_percent': progress,
            'started_at': self._restoration_start.isoformat(),
        }
