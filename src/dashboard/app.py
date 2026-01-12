"""
Dashboard Application Entry Point - Real-time monitoring with Dash.

Provides web-based real-time monitoring interface for the autonomous
futures trading bot. Uses dcc.Interval for polling updates.

Key Features:
- Real-time data polling via dcc.Interval (1-second updates)
- Bootstrap-styled responsive layout
- Multiple dashboard panels for different metrics
- Configurable host/port via environment variables

Usage:
    python src/dashboard/app.py

Dashboard URL: http://localhost:8050 (default)
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import dash
from dash import Dash, dcc, html, callback, Input, Output
import dash_bootstrap_components as dbc

from config import get_config


# Initialize logging
logger = logging.getLogger(__name__)


def create_app() -> Dash:
    """
    Create and configure the Dash application.

    Returns:
        Configured Dash application instance
    """
    config = get_config()

    # Initialize Dash with Bootstrap theme for styling
    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.DARKLY],
        suppress_callback_exceptions=True,
        title="Futures Trading Bot Dashboard",
        update_title="Updating...",
    )

    # Define the layout with dcc.Interval for real-time polling
    app.layout = html.Div([
        # Polling intervals for real-time updates
        dcc.Interval(
            id='interval-fast',
            interval=1000,  # 1 second for fast updates (prices, positions)
            n_intervals=0,
        ),
        dcc.Interval(
            id='interval-medium',
            interval=5000,  # 5 seconds for medium updates (predictions, confidence)
            n_intervals=0,
        ),
        dcc.Interval(
            id='interval-slow',
            interval=30000,  # 30 seconds for slow updates (system health, stats)
            n_intervals=0,
        ),

        # Store components for shared state
        dcc.Store(id='store-trading-state', storage_type='memory'),
        dcc.Store(id='store-positions', storage_type='memory'),
        dcc.Store(id='store-health', storage_type='memory'),

        # Navigation bar
        dbc.Navbar(
            dbc.Container([
                dbc.NavbarBrand("Futures Trading Bot", className="ms-2"),
                dbc.Nav([
                    dbc.NavItem(dbc.NavLink("Dashboard", href="/", active=True)),
                    dbc.NavItem(dbc.NavLink("Contracts", href="/contracts")),
                    dbc.NavItem(dbc.NavLink("History", href="/history")),
                ], className="ms-auto", navbar=True),
                # Connection status indicator
                html.Div(
                    id='connection-status',
                    className="ms-3",
                    children=[
                        dbc.Badge("Connecting...", color="warning", id='status-badge'),
                    ],
                ),
            ], fluid=True),
            color="dark",
            dark=True,
            sticky="top",
        ),

        # Main content area
        dbc.Container([
            # Header row with timestamp
            dbc.Row([
                dbc.Col([
                    html.H5(id='last-update-time', className="text-muted mt-3"),
                ], width=12),
            ]),

            # Main dashboard content (placeholder for layouts.py)
            html.Div(id='main-content', children=[
                # Top metrics row
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Trading Status"),
                            dbc.CardBody([
                                html.H3(id='trading-status', children="--"),
                                html.P("Current trading state", className="text-muted"),
                            ]),
                        ], className="mb-3"),
                    ], md=3),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Session P&L"),
                            dbc.CardBody([
                                html.H3(id='session-pnl', children="$0.00"),
                                html.P("Unrealized + Realized", className="text-muted"),
                            ]),
                        ], className="mb-3"),
                    ], md=3),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Confidence"),
                            dbc.CardBody([
                                html.H3(id='current-confidence', children="--"),
                                html.P("Current model confidence", className="text-muted"),
                            ]),
                        ], className="mb-3"),
                    ], md=3),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Sharpe Ratio"),
                            dbc.CardBody([
                                html.H3(id='sharpe-ratio', children="--"),
                                html.P("Rolling Sharpe", className="text-muted"),
                            ]),
                        ], className="mb-3"),
                    ], md=3),
                ], className="mt-3"),

                # Middle row - positions and active contracts
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Open Positions"),
                            dbc.CardBody([
                                html.Div(id='positions-table', children=[
                                    html.P("No open positions", className="text-muted"),
                                ]),
                            ]),
                        ], className="mb-3"),
                    ], md=6),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Active Contracts"),
                            dbc.CardBody([
                                html.Div(id='contracts-list', children=[
                                    html.P("No contracts subscribed", className="text-muted"),
                                ]),
                            ]),
                        ], className="mb-3"),
                    ], md=6),
                ]),

                # Bottom row - system health and recent trades
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("System Health"),
                            dbc.CardBody([
                                html.Div(id='health-indicators', children=[
                                    _create_health_placeholder(),
                                ]),
                            ]),
                        ], className="mb-3"),
                    ], md=6),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Recent Trades"),
                            dbc.CardBody([
                                html.Div(id='recent-trades', children=[
                                    html.P("No recent trades", className="text-muted"),
                                ]),
                            ]),
                        ], className="mb-3"),
                    ], md=6),
                ]),
            ]),

            # Footer
            dbc.Row([
                dbc.Col([
                    html.Hr(),
                    html.P(
                        "Autonomous Futures Trading Bot - Demo Account Only",
                        className="text-center text-muted",
                    ),
                ]),
            ]),
        ], fluid=True),
    ])

    # Register core callbacks
    _register_callbacks(app)

    # Register full callbacks from callbacks module
    from .callbacks import register_callbacks
    register_callbacks(app)

    logger.info("Dash application created successfully")
    return app


def _create_health_placeholder() -> html.Div:
    """Create placeholder health indicators."""
    return html.Div([
        dbc.ListGroup([
            dbc.ListGroupItem([
                html.Span("IB Connection", className="me-2"),
                dbc.Badge("--", color="secondary", className="ms-auto"),
            ], className="d-flex justify-content-between align-items-center"),
            dbc.ListGroupItem([
                html.Span("Data Freshness", className="me-2"),
                dbc.Badge("--", color="secondary", className="ms-auto"),
            ], className="d-flex justify-content-between align-items-center"),
            dbc.ListGroupItem([
                html.Span("System Resources", className="me-2"),
                dbc.Badge("--", color="secondary", className="ms-auto"),
            ], className="d-flex justify-content-between align-items-center"),
        ], flush=True),
    ])


def _register_callbacks(app: Dash) -> None:
    """
    Register core dashboard callbacks.

    These are minimal callbacks to demonstrate dcc.Interval functionality.
    Full callbacks will be defined in callbacks.py.
    """

    @app.callback(
        Output('last-update-time', 'children'),
        Input('interval-fast', 'n_intervals'),
    )
    def update_timestamp(n: int) -> str:
        """Update the last update timestamp."""
        now = datetime.now(timezone.utc)
        return f"Last updated: {now.strftime('%Y-%m-%d %H:%M:%S')} UTC"

    @app.callback(
        Output('status-badge', 'children'),
        Output('status-badge', 'color'),
        Input('interval-medium', 'n_intervals'),
    )
    def update_connection_status(n: int) -> tuple:
        """
        Update connection status badge.

        This is a placeholder that returns 'Ready' status.
        Full implementation will check actual IB connection.
        """
        # Placeholder - will be connected to actual trading system
        # In real implementation, this queries the shared state
        return "Ready", "success"

    logger.debug("Core callbacks registered")


# Create the global app instance
app = create_app()


# Expose the server for WSGI deployments
server = app.server


def run_dashboard(
    host: Optional[str] = None,
    port: Optional[int] = None,
    debug: Optional[bool] = None,
) -> None:
    """
    Run the dashboard server.

    Args:
        host: Host to bind to (default from config)
        port: Port to bind to (default from config)
        debug: Enable debug mode (default from config)
    """
    config = get_config()

    # Use provided values or fall back to config
    host = host or config.DASHBOARD_HOST
    port = port or config.DASHBOARD_PORT
    debug = debug if debug is not None else config.DASHBOARD_DEBUG

    logger.info(f"Starting dashboard on http://{host}:{port}")
    logger.info(f"Debug mode: {debug}")

    app.run(
        host=host,
        port=port,
        debug=debug,
    )


if __name__ == "__main__":
    # Configure logging when running directly
    config = get_config()
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format=config.LOG_FORMAT,
    )

    # Run the dashboard
    run_dashboard()
