"""
Dashboard Layouts - UI layouts for all required dashboard panels.

Provides comprehensive layouts for monitoring the autonomous futures trading bot:
- Confidence Panel: Current and historical confidence tracking
- Positions Panel: Open positions and exposure display
- P&L Panel: Session P&L and Sharpe ratio
- Contracts Panel: Active contracts with market data
- Training State Panel: Online learning and drift detection status
- Performance Metrics Panel: Sharpe, drawdown, win rate

All layouts are Bootstrap-styled with responsive design using dash-bootstrap-components.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime

from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Color scheme for consistent styling
COLORS = {
    'success': '#28a745',
    'danger': '#dc3545',
    'warning': '#ffc107',
    'info': '#17a2b8',
    'primary': '#007bff',
    'secondary': '#6c757d',
    'light': '#f8f9fa',
    'dark': '#343a40',
    'positive': '#00bc8c',
    'negative': '#e74c3c',
    'neutral': '#adb5bd',
}


def create_layout() -> html.Div:
    """
    Create the main dashboard layout with all required panels.

    Returns:
        Complete dashboard layout as html.Div
    """
    return html.Div([
        # Top metrics row - key performance indicators
        _create_metrics_row(),

        # Second row - Confidence and P&L panels
        dbc.Row([
            dbc.Col([
                create_confidence_panel(),
            ], lg=6, md=12),
            dbc.Col([
                create_pnl_panel(),
            ], lg=6, md=12),
        ], className="mb-3"),

        # Third row - Positions and Contracts
        dbc.Row([
            dbc.Col([
                create_positions_panel(),
            ], lg=6, md=12),
            dbc.Col([
                create_contracts_panel(),
            ], lg=6, md=12),
        ], className="mb-3"),

        # Fourth row - Training State and Performance Metrics
        dbc.Row([
            dbc.Col([
                create_training_state_panel(),
            ], lg=6, md=12),
            dbc.Col([
                create_performance_metrics_panel(),
            ], lg=6, md=12),
        ], className="mb-3"),

        # Fifth row - System Health and Recent Trades
        dbc.Row([
            dbc.Col([
                create_health_panel(),
            ], lg=6, md=12),
            dbc.Col([
                create_recent_trades_panel(),
            ], lg=6, md=12),
        ], className="mb-3"),
    ])


def _create_metrics_row() -> dbc.Row:
    """Create the top row with key performance metrics."""
    return dbc.Row([
        dbc.Col([
            _create_metric_card(
                title="Trading Status",
                value_id="metric-trading-status",
                subtitle="Current state",
                icon="bi-activity",
            ),
        ], xl=2, lg=4, md=4, sm=6, xs=12),
        dbc.Col([
            _create_metric_card(
                title="Session P&L",
                value_id="metric-session-pnl",
                subtitle="Total P&L",
                icon="bi-cash-stack",
                prefix="$",
            ),
        ], xl=2, lg=4, md=4, sm=6, xs=12),
        dbc.Col([
            _create_metric_card(
                title="Confidence",
                value_id="metric-confidence",
                subtitle="Model confidence",
                icon="bi-shield-check",
                suffix="%",
            ),
        ], xl=2, lg=4, md=4, sm=6, xs=12),
        dbc.Col([
            _create_metric_card(
                title="Sharpe Ratio",
                value_id="metric-sharpe",
                subtitle="Rolling Sharpe",
                icon="bi-graph-up-arrow",
            ),
        ], xl=2, lg=4, md=4, sm=6, xs=12),
        dbc.Col([
            _create_metric_card(
                title="Drawdown",
                value_id="metric-drawdown",
                subtitle="Current DD",
                icon="bi-graph-down",
                suffix="%",
            ),
        ], xl=2, lg=4, md=4, sm=6, xs=12),
        dbc.Col([
            _create_metric_card(
                title="Win Rate",
                value_id="metric-win-rate",
                subtitle="Session wins",
                icon="bi-trophy",
                suffix="%",
            ),
        ], xl=2, lg=4, md=4, sm=6, xs=12),
    ], className="mt-3 mb-3")


def _create_metric_card(
    title: str,
    value_id: str,
    subtitle: str,
    icon: str = "bi-info-circle",
    prefix: str = "",
    suffix: str = "",
) -> dbc.Card:
    """
    Create a compact metric card for the top row.

    Args:
        title: Card title
        value_id: ID for the value element (for callback updates)
        subtitle: Subtitle text
        icon: Bootstrap icon class
        prefix: Text prefix for value (e.g., "$")
        suffix: Text suffix for value (e.g., "%")

    Returns:
        Styled metric card
    """
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.I(className=f"{icon} fs-4 text-primary me-2"),
                html.Span(title, className="text-muted small"),
            ], className="d-flex align-items-center mb-2"),
            html.Div([
                html.Span(prefix, className="text-muted"),
                html.Span(
                    "--",
                    id=value_id,
                    className="fs-4 fw-bold",
                ),
                html.Span(suffix, className="text-muted"),
            ]),
            html.Small(subtitle, className="text-muted"),
        ], className="py-2"),
    ], className="h-100")


def create_confidence_panel() -> dbc.Card:
    """
    Create the Confidence Panel showing current and historical confidence.

    Displays:
    - Current overall confidence level
    - Confidence threshold
    - Historical confidence chart
    - Confidence factors breakdown
    """
    return dbc.Card([
        dbc.CardHeader([
            html.I(className="bi-shield-check me-2"),
            "Confidence Tracking",
        ]),
        dbc.CardBody([
            # Current confidence display
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H6("Current Confidence", className="text-muted mb-1"),
                        html.Div([
                            html.Span(
                                "--",
                                id="confidence-current-value",
                                className="fs-3 fw-bold",
                            ),
                            html.Span("%", className="text-muted fs-5"),
                        ]),
                    ]),
                ], md=4),
                dbc.Col([
                    html.Div([
                        html.H6("Threshold", className="text-muted mb-1"),
                        html.Div([
                            html.Span(
                                "--",
                                id="confidence-threshold-value",
                                className="fs-3 fw-bold text-warning",
                            ),
                            html.Span("%", className="text-muted fs-5"),
                        ]),
                    ]),
                ], md=4),
                dbc.Col([
                    html.Div([
                        html.H6("Gate Status", className="text-muted mb-1"),
                        dbc.Badge(
                            "--",
                            id="confidence-gate-status",
                            color="secondary",
                            className="fs-6",
                        ),
                    ]),
                ], md=4),
            ], className="mb-3"),

            # Confidence factors breakdown
            html.H6("Confidence Factors", className="text-muted mb-2"),
            html.Div(id="confidence-factors", children=[
                _create_confidence_factor_row("Model Confidence", "--", "model-confidence"),
                _create_confidence_factor_row("Data Quality", "--", "data-quality"),
                _create_confidence_factor_row("Regime Stability", "--", "regime-stability"),
                _create_confidence_factor_row("Recent Accuracy", "--", "recent-accuracy"),
            ]),

            html.Hr(),

            # Historical confidence chart
            html.H6("Confidence History", className="text-muted mb-2"),
            dcc.Graph(
                id="confidence-history-chart",
                figure=_create_empty_confidence_chart(),
                config={'displayModeBar': False},
                style={'height': '200px'},
            ),
        ]),
    ], className="h-100")


def _create_confidence_factor_row(
    label: str,
    value: str,
    factor_id: str,
) -> html.Div:
    """Create a row for confidence factor display."""
    return html.Div([
        html.Div([
            html.Span(label, className="small"),
            html.Span(value, id=f"confidence-factor-{factor_id}", className="small fw-bold"),
        ], className="d-flex justify-content-between mb-1"),
        dbc.Progress(
            id=f"confidence-progress-{factor_id}",
            value=0,
            color="info",
            className="mb-2",
            style={'height': '8px'},
        ),
    ])


def _create_empty_confidence_chart() -> go.Figure:
    """Create an empty confidence history chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[],
        y=[],
        mode='lines',
        name='Confidence',
        line=dict(color=COLORS['primary'], width=2),
    ))
    fig.add_trace(go.Scatter(
        x=[],
        y=[],
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


def create_pnl_panel() -> dbc.Card:
    """
    Create the P&L Panel showing session P&L and Sharpe ratio.

    Displays:
    - Realized P&L
    - Unrealized P&L
    - Total P&L
    - P&L chart over time
    - Trade statistics
    """
    return dbc.Card([
        dbc.CardHeader([
            html.I(className="bi-cash-stack me-2"),
            "Profit & Loss",
        ]),
        dbc.CardBody([
            # P&L summary row
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H6("Realized", className="text-muted mb-1"),
                        html.Span(
                            "$0.00",
                            id="pnl-realized",
                            className="fs-5 fw-bold",
                        ),
                    ]),
                ], md=4),
                dbc.Col([
                    html.Div([
                        html.H6("Unrealized", className="text-muted mb-1"),
                        html.Span(
                            "$0.00",
                            id="pnl-unrealized",
                            className="fs-5 fw-bold",
                        ),
                    ]),
                ], md=4),
                dbc.Col([
                    html.Div([
                        html.H6("Total", className="text-muted mb-1"),
                        html.Span(
                            "$0.00",
                            id="pnl-total",
                            className="fs-4 fw-bold",
                        ),
                    ]),
                ], md=4),
            ], className="mb-3"),

            # Trade statistics
            html.H6("Session Statistics", className="text-muted mb-2"),
            dbc.Row([
                dbc.Col([
                    _create_stat_badge("Trades", "pnl-trade-count", "0"),
                ], md=3),
                dbc.Col([
                    _create_stat_badge("Wins", "pnl-win-count", "0", color="success"),
                ], md=3),
                dbc.Col([
                    _create_stat_badge("Losses", "pnl-loss-count", "0", color="danger"),
                ], md=3),
                dbc.Col([
                    _create_stat_badge("Avg P&L", "pnl-average", "$0.00"),
                ], md=3),
            ], className="mb-3"),

            html.Hr(),

            # P&L chart
            html.H6("P&L Over Time", className="text-muted mb-2"),
            dcc.Graph(
                id="pnl-chart",
                figure=_create_empty_pnl_chart(),
                config={'displayModeBar': False},
                style={'height': '200px'},
            ),
        ]),
    ], className="h-100")


def _create_stat_badge(
    label: str,
    value_id: str,
    default: str,
    color: str = "secondary",
) -> html.Div:
    """Create a stat badge with label and value."""
    return html.Div([
        html.Small(label, className="text-muted d-block"),
        dbc.Badge(
            default,
            id=value_id,
            color=color,
            className="fs-6",
        ),
    ], className="text-center")


def _create_empty_pnl_chart() -> go.Figure:
    """Create an empty P&L chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[],
        y=[],
        mode='lines',
        name='Cumulative P&L',
        fill='tozeroy',
        line=dict(color=COLORS['primary'], width=2),
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


def create_positions_panel() -> dbc.Card:
    """
    Create the Positions Panel showing open positions and exposure.

    Displays:
    - List of open positions with details
    - Position P&L (unrealized)
    - Total exposure (gross/net)
    - Position limits
    """
    return dbc.Card([
        dbc.CardHeader([
            html.Div([
                html.I(className="bi-briefcase me-2"),
                "Open Positions",
            ], className="d-inline"),
            dbc.Badge(
                "0",
                id="positions-count-badge",
                color="primary",
                className="ms-2",
            ),
        ]),
        dbc.CardBody([
            # Exposure summary
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H6("Gross Exposure", className="text-muted mb-1"),
                        html.Span(
                            "0",
                            id="exposure-gross",
                            className="fs-5 fw-bold",
                        ),
                        html.Small(" contracts", className="text-muted"),
                    ]),
                ], md=4),
                dbc.Col([
                    html.Div([
                        html.H6("Net Exposure", className="text-muted mb-1"),
                        html.Span(
                            "0",
                            id="exposure-net",
                            className="fs-5 fw-bold",
                        ),
                        html.Small(" contracts", className="text-muted"),
                    ]),
                ], md=4),
                dbc.Col([
                    html.Div([
                        html.H6("Max Allowed", className="text-muted mb-1"),
                        html.Span(
                            "--",
                            id="exposure-max",
                            className="fs-5 fw-bold text-warning",
                        ),
                        html.Small(" contracts", className="text-muted"),
                    ]),
                ], md=4),
            ], className="mb-3"),

            html.Hr(),

            # Positions table
            html.Div(
                id="positions-table-container",
                children=[
                    _create_empty_positions_table(),
                ],
                style={'maxHeight': '300px', 'overflowY': 'auto'},
            ),
        ]),
    ], className="h-100")


def _create_empty_positions_table() -> html.Div:
    """Create an empty positions table placeholder."""
    return html.Div([
        dbc.Table([
            html.Thead([
                html.Tr([
                    html.Th("Contract"),
                    html.Th("Side"),
                    html.Th("Qty"),
                    html.Th("Entry"),
                    html.Th("Last"),
                    html.Th("P&L"),
                ]),
            ]),
            html.Tbody([
                html.Tr([
                    html.Td(
                        "No open positions",
                        colSpan=6,
                        className="text-center text-muted",
                    ),
                ]),
            ]),
        ], bordered=True, hover=True, responsive=True, size="sm", striped=True),
    ])


def create_positions_table(positions: List[Dict[str, Any]]) -> dbc.Table:
    """
    Create a positions table with data.

    Args:
        positions: List of position dictionaries with keys:
            - symbol: Contract symbol
            - side: 'LONG' or 'SHORT'
            - quantity: Position size
            - entry_price: Average entry price
            - last_price: Current price
            - unrealized_pnl: Unrealized P&L

    Returns:
        Populated positions table
    """
    if not positions:
        return _create_empty_positions_table()

    rows = []
    for pos in positions:
        pnl = pos.get('unrealized_pnl', 0)
        pnl_color = 'text-success' if pnl >= 0 else 'text-danger'
        side_color = 'success' if pos.get('side') == 'LONG' else 'danger'

        rows.append(html.Tr([
            html.Td(pos.get('symbol', '--')),
            html.Td(dbc.Badge(pos.get('side', '--'), color=side_color, className="small")),
            html.Td(str(pos.get('quantity', 0))),
            html.Td(f"${pos.get('entry_price', 0):,.2f}"),
            html.Td(f"${pos.get('last_price', 0):,.2f}"),
            html.Td(f"${pnl:,.2f}", className=pnl_color),
        ]))

    return dbc.Table([
        html.Thead([
            html.Tr([
                html.Th("Contract"),
                html.Th("Side"),
                html.Th("Qty"),
                html.Th("Entry"),
                html.Th("Last"),
                html.Th("P&L"),
            ]),
        ]),
        html.Tbody(rows),
    ], bordered=True, hover=True, responsive=True, size="sm", striped=True)


def create_contracts_panel() -> dbc.Card:
    """
    Create the Contracts Panel showing active contracts.

    Displays:
    - List of subscribed contracts
    - Market data status for each
    - Last price/bid/ask
    - Data freshness indicators
    """
    return dbc.Card([
        dbc.CardHeader([
            html.Div([
                html.I(className="bi-list-check me-2"),
                "Active Contracts",
            ], className="d-inline"),
            dbc.Badge(
                "0",
                id="contracts-count-badge",
                color="primary",
                className="ms-2",
            ),
        ]),
        dbc.CardBody([
            # Data status summary
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H6("Live Feeds", className="text-muted mb-1"),
                        html.Span(
                            "0",
                            id="contracts-live-count",
                            className="fs-5 fw-bold text-success",
                        ),
                    ]),
                ], md=4),
                dbc.Col([
                    html.Div([
                        html.H6("Stale", className="text-muted mb-1"),
                        html.Span(
                            "0",
                            id="contracts-stale-count",
                            className="fs-5 fw-bold text-warning",
                        ),
                    ]),
                ], md=4),
                dbc.Col([
                    html.Div([
                        html.H6("Errors", className="text-muted mb-1"),
                        html.Span(
                            "0",
                            id="contracts-error-count",
                            className="fs-5 fw-bold text-danger",
                        ),
                    ]),
                ], md=4),
            ], className="mb-3"),

            html.Hr(),

            # Contracts list
            html.Div(
                id="contracts-list-container",
                children=[
                    _create_empty_contracts_list(),
                ],
                style={'maxHeight': '300px', 'overflowY': 'auto'},
            ),
        ]),
    ], className="h-100")


def _create_empty_contracts_list() -> html.Div:
    """Create an empty contracts list placeholder."""
    return html.Div([
        html.P(
            "No contracts subscribed",
            className="text-center text-muted my-4",
        ),
    ])


def create_contracts_list(contracts: List[Dict[str, Any]]) -> html.Div:
    """
    Create a contracts list with data.

    Args:
        contracts: List of contract dictionaries with keys:
            - symbol: Contract symbol
            - exchange: Exchange name
            - last_price: Last traded price
            - bid: Current bid
            - ask: Current ask
            - is_stale: Whether data is stale
            - last_update: Timestamp of last update

    Returns:
        Populated contracts list
    """
    if not contracts:
        return _create_empty_contracts_list()

    items = []
    for contract in contracts:
        is_stale = contract.get('is_stale', False)
        status_color = 'warning' if is_stale else 'success'
        status_text = 'Stale' if is_stale else 'Live'

        items.append(
            dbc.ListGroupItem([
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.Strong(contract.get('symbol', '--')),
                            html.Small(
                                f" ({contract.get('exchange', '--')})",
                                className="text-muted",
                            ),
                        ]),
                        html.Small([
                            f"Last: ${contract.get('last_price', 0):,.2f} | ",
                            f"Bid: ${contract.get('bid', 0):,.2f} | ",
                            f"Ask: ${contract.get('ask', 0):,.2f}",
                        ], className="text-muted"),
                    ], md=9),
                    dbc.Col([
                        dbc.Badge(
                            status_text,
                            color=status_color,
                            className="float-end",
                        ),
                    ], md=3, className="text-end"),
                ]),
            ])
        )

    return dbc.ListGroup(items, flush=True)


def create_training_state_panel() -> dbc.Card:
    """
    Create the Training State Panel showing online learning status.

    Displays:
    - Model accuracy/performance metrics
    - Drift detection status
    - Regime state
    - Learning statistics
    """
    return dbc.Card([
        dbc.CardHeader([
            html.I(className="bi-cpu me-2"),
            "Training State",
        ]),
        dbc.CardBody([
            # Model status row
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H6("Model Accuracy", className="text-muted mb-1"),
                        html.Span(
                            "--%",
                            id="training-accuracy",
                            className="fs-4 fw-bold",
                        ),
                    ]),
                ], md=4),
                dbc.Col([
                    html.Div([
                        html.H6("Samples Seen", className="text-muted mb-1"),
                        html.Span(
                            "0",
                            id="training-samples",
                            className="fs-4 fw-bold",
                        ),
                    ]),
                ], md=4),
                dbc.Col([
                    html.Div([
                        html.H6("Regime State", className="text-muted mb-1"),
                        dbc.Badge(
                            "STABLE",
                            id="training-regime-state",
                            color="success",
                            className="fs-6",
                        ),
                    ]),
                ], md=4),
            ], className="mb-3"),

            # Drift detection
            html.H6("Drift Detection", className="text-muted mb-2"),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Small("Status", className="text-muted d-block"),
                        dbc.Badge(
                            "No Drift",
                            id="drift-status",
                            color="success",
                        ),
                    ], className="text-center"),
                ], md=4),
                dbc.Col([
                    html.Div([
                        html.Small("Severity", className="text-muted d-block"),
                        dbc.Badge(
                            "NONE",
                            id="drift-severity",
                            color="secondary",
                        ),
                    ], className="text-center"),
                ], md=4),
                dbc.Col([
                    html.Div([
                        html.Small("Last Detected", className="text-muted d-block"),
                        html.Span(
                            "--",
                            id="drift-last-detected",
                            className="small",
                        ),
                    ], className="text-center"),
                ], md=4),
            ], className="mb-3"),

            html.Hr(),

            # Model metrics
            html.H6("Model Metrics", className="text-muted mb-2"),
            html.Div([
                _create_metric_row("Precision", "training-precision", "0.00"),
                _create_metric_row("Recall", "training-recall", "0.00"),
                _create_metric_row("F1 Score", "training-f1", "0.00"),
            ]),
        ]),
    ], className="h-100")


def _create_metric_row(label: str, value_id: str, default: str) -> html.Div:
    """Create a metric display row."""
    return html.Div([
        html.Div([
            html.Span(label, className="small"),
            html.Span(default, id=value_id, className="small fw-bold"),
        ], className="d-flex justify-content-between"),
    ], className="mb-2")


def create_performance_metrics_panel() -> dbc.Card:
    """
    Create the Performance Metrics Panel.

    Displays:
    - Sharpe ratio with history
    - Drawdown tracking
    - Win rate and profit factor
    - Risk metrics
    """
    return dbc.Card([
        dbc.CardHeader([
            html.I(className="bi-graph-up me-2"),
            "Performance Metrics",
        ]),
        dbc.CardBody([
            # Key metrics row
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H6("Sharpe Ratio", className="text-muted mb-1"),
                        html.Span(
                            "--",
                            id="perf-sharpe",
                            className="fs-4 fw-bold",
                        ),
                    ]),
                ], md=4),
                dbc.Col([
                    html.Div([
                        html.H6("Max Drawdown", className="text-muted mb-1"),
                        html.Span(
                            "0%",
                            id="perf-max-drawdown",
                            className="fs-4 fw-bold text-danger",
                        ),
                    ]),
                ], md=4),
                dbc.Col([
                    html.Div([
                        html.H6("Profit Factor", className="text-muted mb-1"),
                        html.Span(
                            "--",
                            id="perf-profit-factor",
                            className="fs-4 fw-bold",
                        ),
                    ]),
                ], md=4),
            ], className="mb-3"),

            # Risk state
            html.H6("Risk State", className="text-muted mb-2"),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Small("Trading State", className="text-muted d-block"),
                        dbc.Badge(
                            "ACTIVE",
                            id="perf-trading-state",
                            color="success",
                        ),
                    ], className="text-center"),
                ], md=4),
                dbc.Col([
                    html.Div([
                        html.Small("Risk Level", className="text-muted d-block"),
                        dbc.Badge(
                            "NORMAL",
                            id="perf-risk-level",
                            color="success",
                        ),
                    ], className="text-center"),
                ], md=4),
                dbc.Col([
                    html.Div([
                        html.Small("Drawdown State", className="text-muted d-block"),
                        dbc.Badge(
                            "NORMAL",
                            id="perf-drawdown-state",
                            color="success",
                        ),
                    ], className="text-center"),
                ], md=4),
            ], className="mb-3"),

            html.Hr(),

            # Drawdown chart
            html.H6("Drawdown History", className="text-muted mb-2"),
            dcc.Graph(
                id="drawdown-chart",
                figure=_create_empty_drawdown_chart(),
                config={'displayModeBar': False},
                style={'height': '150px'},
            ),
        ]),
    ], className="h-100")


def _create_empty_drawdown_chart() -> go.Figure:
    """Create an empty drawdown chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[],
        y=[],
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
            autorange='reversed',  # Drawdown is negative
        ),
        showlegend=False,
    )
    return fig


def create_health_panel() -> dbc.Card:
    """
    Create the System Health Panel.

    Displays:
    - IB connection status
    - Data freshness
    - System resources (CPU, memory)
    - Trading gate status
    """
    return dbc.Card([
        dbc.CardHeader([
            html.I(className="bi-heart-pulse me-2"),
            "System Health",
        ]),
        dbc.CardBody([
            # Overall health status
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H6("Overall Status", className="text-muted mb-1"),
                        dbc.Badge(
                            "HEALTHY",
                            id="health-overall-status",
                            color="success",
                            className="fs-5",
                        ),
                    ]),
                ], md=6),
                dbc.Col([
                    html.Div([
                        html.H6("Trading Gate", className="text-muted mb-1"),
                        dbc.Badge(
                            "OPEN",
                            id="health-trading-gate",
                            color="success",
                            className="fs-5",
                        ),
                    ]),
                ], md=6),
            ], className="mb-3"),

            html.Hr(),

            # Component status list
            html.H6("Components", className="text-muted mb-2"),
            dbc.ListGroup([
                _create_health_component_row("IB Connection", "health-ib-connection"),
                _create_health_component_row("Data Freshness", "health-data-freshness"),
                _create_health_component_row("CPU Usage", "health-cpu-usage"),
                _create_health_component_row("Memory Usage", "health-memory-usage"),
                _create_health_component_row("Model State", "health-model-state"),
            ], flush=True),
        ]),
    ], className="h-100")


def _create_health_component_row(label: str, status_id: str) -> dbc.ListGroupItem:
    """Create a health component status row."""
    return dbc.ListGroupItem([
        html.Div([
            html.Span(label),
            dbc.Badge(
                "--",
                id=status_id,
                color="secondary",
                className="ms-auto",
            ),
        ], className="d-flex justify-content-between align-items-center"),
    ])


def create_recent_trades_panel() -> dbc.Card:
    """
    Create the Recent Trades Panel.

    Displays:
    - List of recent trades with details
    - Trade outcome (win/loss)
    - P&L for each trade
    """
    return dbc.Card([
        dbc.CardHeader([
            html.I(className="bi-clock-history me-2"),
            "Recent Trades",
        ]),
        dbc.CardBody([
            html.Div(
                id="recent-trades-container",
                children=[
                    _create_empty_trades_list(),
                ],
                style={'maxHeight': '350px', 'overflowY': 'auto'},
            ),
        ]),
    ], className="h-100")


def _create_empty_trades_list() -> html.Div:
    """Create an empty trades list placeholder."""
    return html.Div([
        html.P(
            "No recent trades",
            className="text-center text-muted my-4",
        ),
    ])


def create_recent_trades_list(trades: List[Dict[str, Any]]) -> dbc.ListGroup:
    """
    Create a recent trades list with data.

    Args:
        trades: List of trade dictionaries with keys:
            - symbol: Contract symbol
            - side: 'BUY' or 'SELL'
            - quantity: Trade size
            - price: Execution price
            - pnl: Realized P&L (if closed)
            - timestamp: Trade timestamp
            - outcome: 'WIN', 'LOSS', or 'PENDING'

    Returns:
        Populated trades list
    """
    if not trades:
        return _create_empty_trades_list()

    items = []
    for trade in trades:
        pnl = trade.get('pnl', 0)
        outcome = trade.get('outcome', 'PENDING')

        if outcome == 'WIN':
            outcome_color = 'success'
        elif outcome == 'LOSS':
            outcome_color = 'danger'
        else:
            outcome_color = 'secondary'

        side_icon = "bi-arrow-up-circle" if trade.get('side') == 'BUY' else "bi-arrow-down-circle"
        side_color = "text-success" if trade.get('side') == 'BUY' else "text-danger"

        timestamp = trade.get('timestamp', '')
        if isinstance(timestamp, datetime):
            timestamp = timestamp.strftime('%H:%M:%S')

        items.append(
            dbc.ListGroupItem([
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.I(className=f"{side_icon} {side_color} me-2"),
                            html.Strong(trade.get('symbol', '--')),
                            html.Span(
                                f" x{trade.get('quantity', 0)}",
                                className="text-muted small",
                            ),
                        ]),
                        html.Small([
                            f"@ ${trade.get('price', 0):,.2f}",
                            f" | {timestamp}",
                        ], className="text-muted"),
                    ], md=8),
                    dbc.Col([
                        html.Div([
                            html.Div([
                                html.Small(f"${pnl:+,.2f}", className="fw-bold"),
                            ], className="text-end"),
                            dbc.Badge(
                                outcome,
                                color=outcome_color,
                                className="float-end mt-1",
                            ),
                        ]),
                    ], md=4, className="text-end"),
                ]),
            ])
        )

    return dbc.ListGroup(items, flush=True)
