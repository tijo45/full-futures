"""
Dashboard Callbacks - Real-time update callbacks for all dashboard components.

Implements the callback pattern from Dash using dcc.Interval for polling updates.
Each callback updates specific dashboard panels based on trading system state.

Key Features:
- Multi-interval updates (fast: 1s, medium: 5s, slow: 30s)
- Shared data store for efficient state management
- Automatic downsampling for performance
- Graceful handling of missing data/components
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

from dash import callback, Input, Output, State, no_update, ctx
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from dash import html
import plotly.graph_objects as go

from .layouts import (
    COLORS,
    create_positions_table,
    create_contracts_list,
    create_recent_trades_list,
    _create_empty_confidence_chart,
    _create_empty_pnl_chart,
    _create_empty_drawdown_chart,
)


# Initialize logging
logger = logging.getLogger(__name__)


# Global data provider reference - set by main.py
_data_provider: Optional['DashboardDataProvider'] = None


class DashboardDataProvider:
    """
    Data provider interface for dashboard callbacks.

    This class provides a unified interface for accessing trading system state.
    It should be set by main.py after initializing all components.

    Usage:
        provider = DashboardDataProvider()
        provider.set_confidence_tracker(tracker)
        provider.set_position_manager(manager)
        set_data_provider(provider)
    """

    def __init__(self):
        """Initialize data provider with None references."""
        self._confidence_tracker = None
        self._position_manager = None
        self._risk_manager = None
        self._health_monitor = None
        self._market_data_handler = None
        self._online_learner = None
        self._drift_detector = None
        self._executor = None
        self._session_manager = None

        # Local data caches for when components aren't available
        self._confidence_history: List[Dict] = []
        self._pnl_history: List[Dict] = []
        self._drawdown_history: List[Dict] = []
        self._trades_history: List[Dict] = []
        self._positions: List[Dict] = []
        self._contracts: List[Dict] = []

        # Maximum history sizes for charts
        self._max_chart_points = 200

        logger.info("DashboardDataProvider initialized")

    # Component setters
    def set_confidence_tracker(self, tracker) -> None:
        """Set confidence tracker reference."""
        self._confidence_tracker = tracker

    def set_position_manager(self, manager) -> None:
        """Set position manager reference."""
        self._position_manager = manager

    def set_risk_manager(self, manager) -> None:
        """Set risk manager reference."""
        self._risk_manager = manager

    def set_health_monitor(self, monitor) -> None:
        """Set health monitor reference."""
        self._health_monitor = monitor

    def set_market_data_handler(self, handler) -> None:
        """Set market data handler reference."""
        self._market_data_handler = handler

    def set_online_learner(self, learner) -> None:
        """Set online learner reference."""
        self._online_learner = learner

    def set_drift_detector(self, detector) -> None:
        """Set drift detector reference."""
        self._drift_detector = detector

    def set_executor(self, executor) -> None:
        """Set executor reference."""
        self._executor = executor

    def set_session_manager(self, manager) -> None:
        """Set session manager reference."""
        self._session_manager = manager

    # Data retrieval methods
    def get_trading_state(self) -> dict:
        """Get current trading state summary."""
        state = {
            'status': 'UNKNOWN',
            'is_trading': False,
            'session_pnl': 0.0,
            'unrealized_pnl': 0.0,
            'realized_pnl': 0.0,
            'total_pnl': 0.0,
            'trade_count': 0,
            'win_count': 0,
            'loss_count': 0,
            'sharpe_ratio': 0.0,
            'drawdown_pct': 0.0,
            'win_rate': 0.0,
        }

        # Get from risk manager
        if self._risk_manager:
            try:
                summary = self._risk_manager.get_summary()
                state['status'] = summary.get('trading_state', 'UNKNOWN').upper()
                state['is_trading'] = summary.get('trading_state') == 'active'
                state['session_pnl'] = summary.get('session_pnl', 0.0)
                state['sharpe_ratio'] = summary.get('sharpe_ratio', 0.0)
                state['drawdown_pct'] = summary.get('drawdown_pct', 0.0) * 100
                state['win_rate'] = summary.get('win_rate', 0.0) * 100
            except Exception as e:
                logger.debug(f"Error getting risk manager state: {e}")

        # Get from position manager
        if self._position_manager:
            try:
                pnl = self._position_manager.get_pnl_summary()
                state['unrealized_pnl'] = pnl.get('unrealized', 0.0)
                state['realized_pnl'] = pnl.get('realized', 0.0)
                state['total_pnl'] = pnl.get('total', 0.0)
            except Exception as e:
                logger.debug(f"Error getting position manager state: {e}")

        # Get trade counts from executor
        if self._executor:
            try:
                stats = self._executor.get_stats()
                state['trade_count'] = stats.get('total_trades', 0)
                state['win_count'] = stats.get('successful_trades', 0)
                state['loss_count'] = stats.get('failed_trades', 0)
            except Exception as e:
                logger.debug(f"Error getting executor state: {e}")

        return state

    def get_confidence_state(self) -> dict:
        """Get confidence tracking state."""
        state = {
            'current': 0.0,
            'threshold': 50.0,
            'gate_status': 'UNKNOWN',
            'mode': 'NORMAL',
            'factors': {
                'model_confidence': 0.0,
                'data_quality': 0.0,
                'regime_stability': 0.0,
                'recent_accuracy': 0.0,
            },
            'history': [],
        }

        if self._confidence_tracker:
            try:
                summary = self._confidence_tracker.get_summary()
                state['current'] = summary.get('recent_confidence', 0.0) * 100
                state['threshold'] = summary.get('threshold', 0.5) * 100
                state['mode'] = summary.get('mode', 'normal').upper()

                # Determine gate status based on allow rate
                allow_rate = summary.get('allow_rate', 0.5)
                if allow_rate > 0.8:
                    state['gate_status'] = 'OPEN'
                elif allow_rate > 0.3:
                    state['gate_status'] = 'GATING'
                else:
                    state['gate_status'] = 'STRICT'

                # Get factors from last confidence calculation
                full_state = self._confidence_tracker.get_state()
                history = full_state.get('history', {})
                if 'mean' in history:
                    state['factors']['model_confidence'] = history.get('mean', 0) * 100
            except Exception as e:
                logger.debug(f"Error getting confidence state: {e}")

        # Add cached history
        state['history'] = self._downsample_history(
            self._confidence_history,
            self._max_chart_points
        )

        return state

    def get_positions(self) -> List[Dict]:
        """Get open positions."""
        if self._position_manager:
            try:
                positions = self._position_manager.get_open_positions()
                return [
                    {
                        'symbol': p.get('symbol', '--'),
                        'side': 'LONG' if p.get('quantity', 0) > 0 else 'SHORT',
                        'quantity': abs(p.get('quantity', 0)),
                        'entry_price': p.get('average_entry_price', 0.0),
                        'last_price': p.get('last_price', 0.0),
                        'unrealized_pnl': p.get('unrealized_pnl', 0.0),
                    }
                    for p in positions
                ]
            except Exception as e:
                logger.debug(f"Error getting positions: {e}")

        return self._positions

    def get_exposure(self) -> dict:
        """Get exposure summary."""
        exposure = {
            'gross': 0,
            'net': 0,
            'max_allowed': 10,  # From config
        }

        if self._position_manager:
            try:
                exp = self._position_manager.get_exposure()
                exposure['gross'] = exp.get('gross', 0)
                exposure['net'] = exp.get('net', 0)
            except Exception as e:
                logger.debug(f"Error getting exposure: {e}")

        return exposure

    def get_contracts(self) -> List[Dict]:
        """Get active contract subscriptions."""
        if self._market_data_handler:
            try:
                contracts = []
                subscriptions = self._market_data_handler.subscriptions
                stale_contracts = set(s[0] for s in self._market_data_handler.get_stale_contracts())

                for symbol, ticker in subscriptions.items():
                    tick_data = self._market_data_handler.get_tick_data(symbol)
                    contracts.append({
                        'symbol': symbol,
                        'exchange': getattr(ticker, 'exchange', '--'),
                        'last_price': tick_data.last if tick_data else 0.0,
                        'bid': tick_data.bid if tick_data else 0.0,
                        'ask': tick_data.ask if tick_data else 0.0,
                        'is_stale': symbol in stale_contracts,
                        'last_update': tick_data.timestamp if tick_data else None,
                    })
                return contracts
            except Exception as e:
                logger.debug(f"Error getting contracts: {e}")

        return self._contracts

    def get_training_state(self) -> dict:
        """Get online learning / training state."""
        state = {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'samples_seen': 0,
            'regime_state': 'STABLE',
            'drift_status': 'No Drift',
            'drift_severity': 'NONE',
            'last_drift_time': None,
        }

        if self._online_learner:
            try:
                metrics = self._online_learner.get_metrics()
                state['accuracy'] = metrics.get('accuracy', 0.0) * 100
                state['precision'] = metrics.get('precision', 0.0)
                state['recall'] = metrics.get('recall', 0.0)
                state['f1'] = metrics.get('f1', 0.0)
                state['samples_seen'] = metrics.get('samples_seen', 0)
            except Exception as e:
                logger.debug(f"Error getting online learner state: {e}")

        if self._drift_detector:
            try:
                drift_state = self._drift_detector.get_state()
                state['regime_state'] = drift_state.get('regime_state', 'STABLE').upper()
                state['drift_severity'] = drift_state.get('severity', 'NONE').upper()
                if drift_state.get('drift_detected', False):
                    state['drift_status'] = 'Drift Detected'
                else:
                    state['drift_status'] = 'No Drift'
                state['last_drift_time'] = drift_state.get('last_drift_time')
            except Exception as e:
                logger.debug(f"Error getting drift detector state: {e}")

        return state

    def get_performance_metrics(self) -> dict:
        """Get performance metrics."""
        metrics = {
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'profit_factor': 0.0,
            'trading_state': 'UNKNOWN',
            'risk_level': 'NORMAL',
            'drawdown_state': 'NORMAL',
            'history': [],
        }

        if self._risk_manager:
            try:
                assessment = self._risk_manager.get_risk_assessment()
                metrics['sharpe_ratio'] = assessment.get('sharpe', {}).get('sharpe_ratio', 0.0)
                metrics['max_drawdown'] = assessment.get('drawdown', {}).get('max_drawdown_pct', 0.0) * 100
                metrics['profit_factor'] = assessment.get('sharpe', {}).get('profit_factor', 0.0)
                metrics['trading_state'] = assessment.get('trading_state', 'unknown').upper()
                metrics['risk_level'] = assessment.get('risk_level', 'normal').upper()
                metrics['drawdown_state'] = assessment.get('drawdown', {}).get('state', 'normal').upper()
            except Exception as e:
                logger.debug(f"Error getting performance metrics: {e}")

        # Add cached drawdown history
        metrics['history'] = self._downsample_history(
            self._drawdown_history,
            self._max_chart_points
        )

        return metrics

    def get_health_state(self) -> dict:
        """Get system health state."""
        state = {
            'overall_status': 'UNKNOWN',
            'trading_gate': 'CLOSED',
            'components': {
                'ib_connection': {'state': 'UNKNOWN', 'message': '--'},
                'data_freshness': {'state': 'UNKNOWN', 'message': '--'},
                'system_resources': {'state': 'UNKNOWN', 'message': '--'},
                'model_state': {'state': 'UNKNOWN', 'message': '--'},
            },
            'cpu_usage': None,
            'memory_usage': None,
        }

        if self._health_monitor:
            try:
                summary = self._health_monitor.get_summary()
                state['overall_status'] = summary.get('overall_state', 'unknown').upper()
                state['trading_gate'] = summary.get('trading_gate', 'closed').upper()

                components = summary.get('components', {})
                for name, comp in components.items():
                    if name in state['components']:
                        state['components'][name] = {
                            'state': comp.get('state', 'unknown').upper(),
                            'message': comp.get('message', '--'),
                        }

                resources = summary.get('resources', {})
                if resources:
                    state['cpu_usage'] = resources.get('cpu_percent')
                    state['memory_usage'] = resources.get('memory_percent')
            except Exception as e:
                logger.debug(f"Error getting health state: {e}")

        return state

    def get_recent_trades(self, limit: int = 20) -> List[Dict]:
        """Get recent trades."""
        if self._executor:
            try:
                trades = self._executor.get_recent_trades(limit=limit)
                return [
                    {
                        'symbol': t.get('symbol', '--'),
                        'side': t.get('side', '--'),
                        'quantity': t.get('quantity', 0),
                        'price': t.get('fill_price', 0.0),
                        'pnl': t.get('realized_pnl', 0.0),
                        'timestamp': t.get('timestamp'),
                        'outcome': 'WIN' if t.get('success', False) else 'LOSS',
                    }
                    for t in trades
                ]
            except Exception as e:
                logger.debug(f"Error getting recent trades: {e}")

        return self._trades_history[-limit:]

    def get_pnl_history(self) -> List[Dict]:
        """Get P&L history for chart."""
        return self._downsample_history(self._pnl_history, self._max_chart_points)

    # History recording methods (called by main.py)
    def record_confidence(self, value: float, threshold: float) -> None:
        """Record confidence value for history chart."""
        self._confidence_history.append({
            'timestamp': datetime.now(timezone.utc),
            'value': value * 100,
            'threshold': threshold * 100,
        })
        self._trim_history(self._confidence_history, 1000)

    def record_pnl(self, cumulative_pnl: float) -> None:
        """Record P&L for history chart."""
        self._pnl_history.append({
            'timestamp': datetime.now(timezone.utc),
            'pnl': cumulative_pnl,
        })
        self._trim_history(self._pnl_history, 1000)

    def record_drawdown(self, drawdown_pct: float) -> None:
        """Record drawdown for history chart."""
        self._drawdown_history.append({
            'timestamp': datetime.now(timezone.utc),
            'drawdown': drawdown_pct * 100,
        })
        self._trim_history(self._drawdown_history, 1000)

    def record_trade(self, trade: dict) -> None:
        """Record trade for history."""
        self._trades_history.append(trade)
        self._trim_history(self._trades_history, 100)

    # Helper methods
    def _trim_history(self, history: List, max_size: int) -> None:
        """Trim history list to max size."""
        while len(history) > max_size:
            history.pop(0)

    def _downsample_history(
        self,
        history: List[Dict],
        target_points: int
    ) -> List[Dict]:
        """Downsample history to target number of points."""
        if len(history) <= target_points:
            return history.copy()

        # Simple downsampling - take every nth point
        step = len(history) // target_points
        return history[::step]


def set_data_provider(provider: DashboardDataProvider) -> None:
    """Set the global data provider for callbacks."""
    global _data_provider
    _data_provider = provider
    logger.info("Dashboard data provider set")


def get_data_provider() -> Optional[DashboardDataProvider]:
    """Get the current data provider."""
    return _data_provider


def register_callbacks(app) -> None:
    """
    Register all dashboard callbacks with the Dash app.

    Args:
        app: Dash application instance
    """
    logger.info("Registering dashboard callbacks")

    # =========================================================================
    # Store Updates - Shared state management
    # =========================================================================

    @app.callback(
        Output('store-trading-state', 'data'),
        Input('interval-fast', 'n_intervals'),
    )
    def update_trading_state_store(n: int):
        """Update trading state store on fast interval."""
        if _data_provider is None:
            return {}
        return _data_provider.get_trading_state()

    @app.callback(
        Output('store-positions', 'data'),
        Input('interval-fast', 'n_intervals'),
    )
    def update_positions_store(n: int):
        """Update positions store on fast interval."""
        if _data_provider is None:
            return []
        return _data_provider.get_positions()

    @app.callback(
        Output('store-health', 'data'),
        Input('interval-slow', 'n_intervals'),
    )
    def update_health_store(n: int):
        """Update health store on slow interval."""
        if _data_provider is None:
            return {}
        return _data_provider.get_health_state()

    # =========================================================================
    # Top Metrics Row Updates
    # =========================================================================

    @app.callback(
        Output('metric-trading-status', 'children'),
        Output('metric-trading-status', 'className'),
        Input('store-trading-state', 'data'),
    )
    def update_trading_status(state: Optional[dict]):
        """Update trading status metric."""
        if not state:
            return "--", "fs-4 fw-bold text-muted"

        status = state.get('status', 'UNKNOWN')

        status_classes = {
            'ACTIVE': 'fs-4 fw-bold text-success',
            'REDUCED': 'fs-4 fw-bold text-warning',
            'CAUTIOUS': 'fs-4 fw-bold text-warning',
            'HALTED': 'fs-4 fw-bold text-danger',
        }

        return status, status_classes.get(status, 'fs-4 fw-bold text-muted')

    @app.callback(
        Output('metric-session-pnl', 'children'),
        Output('metric-session-pnl', 'className'),
        Input('store-trading-state', 'data'),
    )
    def update_session_pnl(state: Optional[dict]):
        """Update session P&L metric."""
        if not state:
            return "0.00", "fs-4 fw-bold"

        pnl = state.get('total_pnl', 0.0)
        pnl_class = 'fs-4 fw-bold text-success' if pnl >= 0 else 'fs-4 fw-bold text-danger'

        return f"{pnl:,.2f}", pnl_class

    @app.callback(
        Output('metric-confidence', 'children'),
        Input('interval-medium', 'n_intervals'),
    )
    def update_confidence_metric(n: int):
        """Update confidence metric."""
        if _data_provider is None:
            return "--"

        state = _data_provider.get_confidence_state()
        return f"{state.get('current', 0):.1f}"

    @app.callback(
        Output('metric-sharpe', 'children'),
        Input('store-trading-state', 'data'),
    )
    def update_sharpe_metric(state: Optional[dict]):
        """Update Sharpe ratio metric."""
        if not state:
            return "--"

        sharpe = state.get('sharpe_ratio', 0.0)
        return f"{sharpe:.2f}"

    @app.callback(
        Output('metric-drawdown', 'children'),
        Output('metric-drawdown', 'className'),
        Input('store-trading-state', 'data'),
    )
    def update_drawdown_metric(state: Optional[dict]):
        """Update drawdown metric."""
        if not state:
            return "0.0", "fs-4 fw-bold"

        dd = state.get('drawdown_pct', 0.0)

        if dd >= 10:
            dd_class = 'fs-4 fw-bold text-danger'
        elif dd >= 5:
            dd_class = 'fs-4 fw-bold text-warning'
        else:
            dd_class = 'fs-4 fw-bold text-success'

        return f"{dd:.1f}", dd_class

    @app.callback(
        Output('metric-win-rate', 'children'),
        Input('store-trading-state', 'data'),
    )
    def update_win_rate_metric(state: Optional[dict]):
        """Update win rate metric."""
        if not state:
            return "--"

        win_rate = state.get('win_rate', 0.0)
        return f"{win_rate:.1f}"

    # =========================================================================
    # Confidence Panel Updates
    # =========================================================================

    @app.callback(
        Output('confidence-current-value', 'children'),
        Output('confidence-threshold-value', 'children'),
        Output('confidence-gate-status', 'children'),
        Output('confidence-gate-status', 'color'),
        Output('confidence-factor-model-confidence', 'children'),
        Output('confidence-factor-data-quality', 'children'),
        Output('confidence-factor-regime-stability', 'children'),
        Output('confidence-factor-recent-accuracy', 'children'),
        Output('confidence-progress-model-confidence', 'value'),
        Output('confidence-progress-data-quality', 'value'),
        Output('confidence-progress-regime-stability', 'value'),
        Output('confidence-progress-recent-accuracy', 'value'),
        Input('interval-medium', 'n_intervals'),
    )
    def update_confidence_panel(n: int):
        """Update confidence panel."""
        if _data_provider is None:
            return (
                "--", "--", "--", "secondary",
                "--", "--", "--", "--",
                0, 0, 0, 0,
            )

        state = _data_provider.get_confidence_state()

        gate_colors = {
            'OPEN': 'success',
            'GATING': 'warning',
            'STRICT': 'danger',
        }

        factors = state.get('factors', {})

        return (
            f"{state.get('current', 0):.1f}",
            f"{state.get('threshold', 50):.1f}",
            state.get('gate_status', 'UNKNOWN'),
            gate_colors.get(state.get('gate_status', ''), 'secondary'),
            f"{factors.get('model_confidence', 0):.1f}%",
            f"{factors.get('data_quality', 0):.1f}%",
            f"{factors.get('regime_stability', 0):.1f}%",
            f"{factors.get('recent_accuracy', 0):.1f}%",
            int(factors.get('model_confidence', 0)),
            int(factors.get('data_quality', 0)),
            int(factors.get('regime_stability', 0)),
            int(factors.get('recent_accuracy', 0)),
        )

    @app.callback(
        Output('confidence-history-chart', 'figure'),
        Input('interval-medium', 'n_intervals'),
    )
    def update_confidence_chart(n: int):
        """Update confidence history chart."""
        if _data_provider is None:
            return _create_empty_confidence_chart()

        state = _data_provider.get_confidence_state()
        history = state.get('history', [])

        if not history:
            return _create_empty_confidence_chart()

        timestamps = [h['timestamp'] for h in history]
        values = [h['value'] for h in history]
        thresholds = [h.get('threshold', 50) for h in history]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=timestamps,
            y=values,
            mode='lines',
            name='Confidence',
            line=dict(color=COLORS['primary'], width=2),
        ))

        fig.add_trace(go.Scatter(
            x=timestamps,
            y=thresholds,
            mode='lines',
            name='Threshold',
            line=dict(color=COLORS['warning'], width=2, dash='dash'),
        ))

        fig.update_layout(
            margin=dict(l=40, r=20, t=10, b=30),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, color='#adb5bd'),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)',
                color='#adb5bd',
                range=[0, 100],
            ),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1,
                font=dict(size=10),
            ),
            showlegend=True,
        )

        return fig

    # =========================================================================
    # P&L Panel Updates
    # =========================================================================

    @app.callback(
        Output('pnl-realized', 'children'),
        Output('pnl-realized', 'className'),
        Output('pnl-unrealized', 'children'),
        Output('pnl-unrealized', 'className'),
        Output('pnl-total', 'children'),
        Output('pnl-total', 'className'),
        Output('pnl-trade-count', 'children'),
        Output('pnl-win-count', 'children'),
        Output('pnl-loss-count', 'children'),
        Output('pnl-average', 'children'),
        Input('store-trading-state', 'data'),
    )
    def update_pnl_panel(state: Optional[dict]):
        """Update P&L panel."""
        if not state:
            return (
                "$0.00", "fs-5 fw-bold",
                "$0.00", "fs-5 fw-bold",
                "$0.00", "fs-4 fw-bold",
                "0", "0", "0", "$0.00",
            )

        realized = state.get('realized_pnl', 0.0)
        unrealized = state.get('unrealized_pnl', 0.0)
        total = state.get('total_pnl', 0.0)

        trade_count = state.get('trade_count', 0)
        win_count = state.get('win_count', 0)
        loss_count = state.get('loss_count', 0)
        avg_pnl = total / max(1, trade_count)

        def pnl_class(value: float, size: str = 'fs-5') -> str:
            base = f'{size} fw-bold'
            if value > 0:
                return f'{base} text-success'
            elif value < 0:
                return f'{base} text-danger'
            return base

        return (
            f"${realized:,.2f}", pnl_class(realized),
            f"${unrealized:,.2f}", pnl_class(unrealized),
            f"${total:,.2f}", pnl_class(total, 'fs-4'),
            str(trade_count),
            str(win_count),
            str(loss_count),
            f"${avg_pnl:,.2f}",
        )

    @app.callback(
        Output('pnl-chart', 'figure'),
        Input('interval-medium', 'n_intervals'),
    )
    def update_pnl_chart(n: int):
        """Update P&L history chart."""
        if _data_provider is None:
            return _create_empty_pnl_chart()

        history = _data_provider.get_pnl_history()

        if not history:
            return _create_empty_pnl_chart()

        timestamps = [h['timestamp'] for h in history]
        pnl_values = [h['pnl'] for h in history]

        fig = go.Figure()

        # Determine fill color based on latest P&L
        fill_color = COLORS['positive'] if pnl_values[-1] >= 0 else COLORS['negative']

        fig.add_trace(go.Scatter(
            x=timestamps,
            y=pnl_values,
            mode='lines',
            name='Cumulative P&L',
            fill='tozeroy',
            line=dict(color=fill_color, width=2),
            fillcolor=f"rgba({int(fill_color[1:3], 16)}, {int(fill_color[3:5], 16)}, {int(fill_color[5:7], 16)}, 0.3)",
        ))

        fig.update_layout(
            margin=dict(l=40, r=20, t=10, b=30),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, color='#adb5bd'),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)',
                color='#adb5bd',
            ),
            showlegend=False,
        )

        return fig

    # =========================================================================
    # Positions Panel Updates
    # =========================================================================

    @app.callback(
        Output('positions-count-badge', 'children'),
        Output('exposure-gross', 'children'),
        Output('exposure-net', 'children'),
        Output('exposure-max', 'children'),
        Output('positions-table-container', 'children'),
        Input('store-positions', 'data'),
        Input('interval-fast', 'n_intervals'),
    )
    def update_positions_panel(positions: Optional[List], n: int):
        """Update positions panel."""
        if _data_provider is None:
            positions = []

        if positions is None:
            positions = []

        exposure = {'gross': 0, 'net': 0, 'max_allowed': 10}
        if _data_provider:
            exposure = _data_provider.get_exposure()

        position_count = len(positions)

        return (
            str(position_count),
            str(exposure.get('gross', 0)),
            str(exposure.get('net', 0)),
            str(exposure.get('max_allowed', 10)),
            create_positions_table(positions),
        )

    # =========================================================================
    # Contracts Panel Updates
    # =========================================================================

    @app.callback(
        Output('contracts-count-badge', 'children'),
        Output('contracts-live-count', 'children'),
        Output('contracts-stale-count', 'children'),
        Output('contracts-error-count', 'children'),
        Output('contracts-list-container', 'children'),
        Input('interval-fast', 'n_intervals'),
    )
    def update_contracts_panel(n: int):
        """Update contracts panel."""
        if _data_provider is None:
            return "0", "0", "0", "0", create_contracts_list([])

        contracts = _data_provider.get_contracts()

        total_count = len(contracts)
        stale_count = sum(1 for c in contracts if c.get('is_stale', False))
        live_count = total_count - stale_count
        error_count = 0  # Can be extended to track error states

        return (
            str(total_count),
            str(live_count),
            str(stale_count),
            str(error_count),
            create_contracts_list(contracts),
        )

    # =========================================================================
    # Training State Panel Updates
    # =========================================================================

    @app.callback(
        Output('training-accuracy', 'children'),
        Output('training-samples', 'children'),
        Output('training-regime-state', 'children'),
        Output('training-regime-state', 'color'),
        Output('drift-status', 'children'),
        Output('drift-status', 'color'),
        Output('drift-severity', 'children'),
        Output('drift-severity', 'color'),
        Output('drift-last-detected', 'children'),
        Output('training-precision', 'children'),
        Output('training-recall', 'children'),
        Output('training-f1', 'children'),
        Input('interval-medium', 'n_intervals'),
    )
    def update_training_state_panel(n: int):
        """Update training state panel."""
        if _data_provider is None:
            return (
                "--%", "0", "STABLE", "success",
                "No Drift", "success", "NONE", "secondary", "--",
                "0.00", "0.00", "0.00",
            )

        state = _data_provider.get_training_state()

        regime_colors = {
            'STABLE': 'success',
            'TRANSITIONING': 'warning',
            'VOLATILE': 'danger',
            'RECOVERING': 'info',
        }

        severity_colors = {
            'NONE': 'secondary',
            'MILD': 'info',
            'MODERATE': 'warning',
            'SEVERE': 'danger',
        }

        drift_color = 'danger' if state.get('drift_status') == 'Drift Detected' else 'success'

        last_drift = state.get('last_drift_time')
        if last_drift:
            if isinstance(last_drift, datetime):
                last_drift_str = last_drift.strftime('%H:%M:%S')
            else:
                last_drift_str = str(last_drift)
        else:
            last_drift_str = "--"

        return (
            f"{state.get('accuracy', 0):.1f}%",
            str(state.get('samples_seen', 0)),
            state.get('regime_state', 'STABLE'),
            regime_colors.get(state.get('regime_state', 'STABLE'), 'success'),
            state.get('drift_status', 'No Drift'),
            drift_color,
            state.get('drift_severity', 'NONE'),
            severity_colors.get(state.get('drift_severity', 'NONE'), 'secondary'),
            last_drift_str,
            f"{state.get('precision', 0):.2f}",
            f"{state.get('recall', 0):.2f}",
            f"{state.get('f1', 0):.2f}",
        )

    # =========================================================================
    # Performance Metrics Panel Updates
    # =========================================================================

    @app.callback(
        Output('perf-sharpe', 'children'),
        Output('perf-max-drawdown', 'children'),
        Output('perf-profit-factor', 'children'),
        Output('perf-trading-state', 'children'),
        Output('perf-trading-state', 'color'),
        Output('perf-risk-level', 'children'),
        Output('perf-risk-level', 'color'),
        Output('perf-drawdown-state', 'children'),
        Output('perf-drawdown-state', 'color'),
        Input('interval-medium', 'n_intervals'),
    )
    def update_performance_metrics_panel(n: int):
        """Update performance metrics panel."""
        if _data_provider is None:
            return (
                "--", "0%", "--",
                "UNKNOWN", "secondary",
                "NORMAL", "success",
                "NORMAL", "success",
            )

        metrics = _data_provider.get_performance_metrics()

        trading_state_colors = {
            'ACTIVE': 'success',
            'REDUCED': 'warning',
            'CAUTIOUS': 'warning',
            'HALTED': 'danger',
        }

        risk_level_colors = {
            'LOW': 'success',
            'NORMAL': 'success',
            'ELEVATED': 'warning',
            'HIGH': 'danger',
            'CRITICAL': 'danger',
        }

        drawdown_state_colors = {
            'NORMAL': 'success',
            'RECOVERY': 'info',
            'WARNING': 'warning',
            'DANGER': 'danger',
            'CRITICAL': 'danger',
        }

        sharpe = metrics.get('sharpe_ratio', 0)
        max_dd = metrics.get('max_drawdown', 0)
        pf = metrics.get('profit_factor', 0)
        trading_state = metrics.get('trading_state', 'UNKNOWN')
        risk_level = metrics.get('risk_level', 'NORMAL')
        dd_state = metrics.get('drawdown_state', 'NORMAL')

        pf_str = f"{pf:.2f}" if pf != float('inf') else "∞"

        return (
            f"{sharpe:.2f}",
            f"{max_dd:.1f}%",
            pf_str,
            trading_state,
            trading_state_colors.get(trading_state, 'secondary'),
            risk_level,
            risk_level_colors.get(risk_level, 'success'),
            dd_state,
            drawdown_state_colors.get(dd_state, 'success'),
        )

    @app.callback(
        Output('drawdown-chart', 'figure'),
        Input('interval-medium', 'n_intervals'),
    )
    def update_drawdown_chart(n: int):
        """Update drawdown history chart."""
        if _data_provider is None:
            return _create_empty_drawdown_chart()

        metrics = _data_provider.get_performance_metrics()
        history = metrics.get('history', [])

        if not history:
            return _create_empty_drawdown_chart()

        timestamps = [h['timestamp'] for h in history]
        drawdown_values = [-h['drawdown'] for h in history]  # Negate for display

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=timestamps,
            y=drawdown_values,
            mode='lines',
            name='Drawdown',
            fill='tozeroy',
            line=dict(color=COLORS['danger'], width=2),
            fillcolor='rgba(231, 76, 60, 0.2)',
        ))

        fig.update_layout(
            margin=dict(l=40, r=20, t=10, b=30),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, color='#adb5bd'),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)',
                color='#adb5bd',
                autorange='reversed',
            ),
            showlegend=False,
        )

        return fig

    # =========================================================================
    # Health Panel Updates
    # =========================================================================

    @app.callback(
        Output('health-overall-status', 'children'),
        Output('health-overall-status', 'color'),
        Output('health-trading-gate', 'children'),
        Output('health-trading-gate', 'color'),
        Output('health-ib-connection', 'children'),
        Output('health-ib-connection', 'color'),
        Output('health-data-freshness', 'children'),
        Output('health-data-freshness', 'color'),
        Output('health-cpu-usage', 'children'),
        Output('health-cpu-usage', 'color'),
        Output('health-memory-usage', 'children'),
        Output('health-memory-usage', 'color'),
        Output('health-model-state', 'children'),
        Output('health-model-state', 'color'),
        Input('store-health', 'data'),
    )
    def update_health_panel(health: Optional[dict]):
        """Update health panel."""
        if not health:
            return (
                "UNKNOWN", "secondary",
                "CLOSED", "danger",
                "--", "secondary",
                "--", "secondary",
                "--", "secondary",
                "--", "secondary",
                "--", "secondary",
            )

        status_colors = {
            'HEALTHY': 'success',
            'DEGRADED': 'warning',
            'CRITICAL': 'danger',
            'UNKNOWN': 'secondary',
        }

        gate_colors = {
            'OPEN': 'success',
            'PENDING': 'warning',
            'CLOSED': 'danger',
        }

        component_colors = {
            'OK': 'success',
            'WARNING': 'warning',
            'ERROR': 'danger',
            'UNKNOWN': 'secondary',
        }

        overall_status = health.get('overall_status', 'UNKNOWN')
        trading_gate = health.get('trading_gate', 'CLOSED')

        components = health.get('components', {})
        ib_conn = components.get('ib_connection', {})
        data_fresh = components.get('data_freshness', {})
        sys_res = components.get('system_resources', {})
        model = components.get('model_state', {})

        # Format CPU and memory usage
        cpu = health.get('cpu_usage')
        mem = health.get('memory_usage')

        if cpu is not None:
            cpu_str = f"{cpu:.0f}%"
            cpu_color = 'success' if cpu < 70 else ('warning' if cpu < 90 else 'danger')
        else:
            cpu_str = "--"
            cpu_color = 'secondary'

        if mem is not None:
            mem_str = f"{mem:.0f}%"
            mem_color = 'success' if mem < 70 else ('warning' if mem < 90 else 'danger')
        else:
            mem_str = "--"
            mem_color = 'secondary'

        return (
            overall_status,
            status_colors.get(overall_status, 'secondary'),
            trading_gate,
            gate_colors.get(trading_gate, 'danger'),
            ib_conn.get('state', 'UNKNOWN'),
            component_colors.get(ib_conn.get('state', 'UNKNOWN'), 'secondary'),
            data_fresh.get('state', 'UNKNOWN'),
            component_colors.get(data_fresh.get('state', 'UNKNOWN'), 'secondary'),
            cpu_str,
            cpu_color,
            mem_str,
            mem_color,
            model.get('state', 'UNKNOWN'),
            component_colors.get(model.get('state', 'UNKNOWN'), 'secondary'),
        )

    # =========================================================================
    # Recent Trades Panel Updates
    # =========================================================================

    @app.callback(
        Output('recent-trades-container', 'children'),
        Input('interval-medium', 'n_intervals'),
    )
    def update_recent_trades_panel(n: int):
        """Update recent trades panel."""
        if _data_provider is None:
            return create_recent_trades_list([])

        trades = _data_provider.get_recent_trades(limit=20)
        return create_recent_trades_list(trades)

    logger.info("Dashboard callbacks registered successfully")
