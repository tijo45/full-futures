# Specification: Autonomous Futures Trading Bot

## Overview

Build a fully autonomous futures trading bot using Interactive Brokers Python API that operates exclusively with live market data on a demo account. The system must be completely self-discovering, self-calibrating, and self-adapting—with zero hard-coded values, fixed strategies, or static thresholds. The explicit goal is ultra-aggressive capital growth, optimizing for rapid profitability discovery and Sharpe ratio improvement while maintaining strict data integrity and session awareness. The bot will leverage both Level 1 and Level 2 market data, implement continuous online learning from trading results (including losses as feedback), and provide a real-time dashboard for monitoring all aspects of the trading system.

## Workflow Type

**Type**: feature

**Rationale**: This is a greenfield implementation of a complete trading system with multiple interconnected components (IB integration, prediction engine, learning system, dashboard). It requires building from scratch with careful architectural planning for real-time performance and reliability.

## Task Scope

### Services Involved
- **trading-bot** (primary) - Core autonomous trading engine with IB integration
- **dashboard** (primary) - Real-time web visualization using Dash
- **prediction-engine** (primary) - Online ML with River for continuous learning
- **health-monitor** (primary) - System and data integrity watchdog

### This Task Will:
- [ ] Build IB API integration layer using ib_async for connection management, data feeds, and order execution
- [ ] Implement automatic futures contract discovery on startup
- [ ] Create real-time prediction engine with confidence scoring using River online ML
- [ ] Build continuous learning system that adapts from trading results
- [ ] Implement session and calendar management using pandas_market_calendars
- [ ] Create health monitoring for connectivity, data integrity, and system stability
- [ ] Build real-time Dash dashboard displaying all required metrics
- [ ] Implement confidence-gated trade execution with position management
- [ ] Create authoritative order/position/exposure tracking with IB reconciliation

### Out of Scope:
- Backtesting/offline simulation mode
- Trading of stocks, options, forex, or crypto
- Docker containerization
- Cloud deployment
- Multi-account support

## Service Context

### Trading Bot Core

**Tech Stack:**
- Language: Python 3.10+
- IB Integration: ib_async (maintained successor to ib_insync)
- Online ML: River
- Calendars: pandas_market_calendars
- Key directories: `src/`, `src/core/`, `src/trading/`, `src/data/`

**Entry Point:** `src/main.py`

**How to Run:**
```bash
python src/main.py
```

**Port:** N/A (connects to TWS on 7497)

### Dashboard Service

**Tech Stack:**
- Language: Python 3.10+
- Framework: Dash (Plotly)
- Key directories: `src/dashboard/`

**Entry Point:** `src/dashboard/app.py`

**How to Run:**
```bash
python src/dashboard/app.py
```

**Port:** 8050

## Files to Create

| File | Service | What to Create |
|------|---------|----------------|
| `src/main.py` | trading-bot | Application entry point and orchestration |
| `src/core/ib_client.py` | trading-bot | IB connection manager using ib_async |
| `src/core/contract_discovery.py` | trading-bot | Auto-discover tradeable futures contracts |
| `src/core/session_manager.py` | trading-bot | Exchange calendar and session handling |
| `src/core/health_monitor.py` | trading-bot | Connectivity and data integrity watchdog |
| `src/data/market_data.py` | trading-bot | Level 1 and Level 2 data handlers |
| `src/data/feature_engine.py` | trading-bot | Real-time feature extraction |
| `src/trading/predictor.py` | trading-bot | River-based online prediction engine |
| `src/trading/confidence.py` | trading-bot | Confidence scoring and tracking |
| `src/trading/executor.py` | trading-bot | Order execution with confidence gates |
| `src/trading/position_manager.py` | trading-bot | Position and exposure tracking |
| `src/trading/risk_manager.py` | trading-bot | Risk and Sharpe optimization |
| `src/learning/online_learner.py` | trading-bot | River continuous learning system |
| `src/learning/drift_detector.py` | trading-bot | Market regime change detection |
| `src/dashboard/app.py` | dashboard | Dash application entry point |
| `src/dashboard/layouts.py` | dashboard | Dashboard UI layouts |
| `src/dashboard/callbacks.py` | dashboard | Real-time update callbacks |
| `requirements.txt` | all | Python dependencies |
| `config.py` | all | Dynamic configuration (no hard-coded values) |

## Files to Reference

These files show patterns to follow:

| File | Pattern to Copy |
|------|----------------|
| ib_async documentation | Connection patterns, async handling |
| River documentation | Online learning with `predict_one`/`learn_one` |
| Dash documentation | `dcc.Interval` for real-time updates |
| pandas_market_calendars docs | Exchange schedule queries |

## Patterns to Follow

### IB Async Connection Pattern

From ib_async library:

```python
from ib_async import IB, Future, util

class IBClient:
    def __init__(self):
        self.ib = IB()

    async def connect(self, host='127.0.0.1', port=7497, client_id=1):
        await self.ib.connectAsync(host, port, clientId=client_id)

    async def subscribe_market_data(self, contract):
        ticker = self.ib.reqMktData(contract)
        return ticker

    async def subscribe_depth(self, contract):
        ticker = self.ib.reqMktDepth(contract)
        return ticker  # Access ticker.domBids, ticker.domAsks
```

**Key Points:**
- Use async patterns for concurrent operations
- Each connection needs unique `clientId`
- ContFuture is for historical data ONLY, not real-time

### River Online Learning Pattern

From River library:

```python
from river import linear_model, preprocessing, compose, metrics, drift

class OnlineLearner:
    def __init__(self):
        self.model = compose.Pipeline(
            preprocessing.StandardScaler(),
            linear_model.LogisticRegression()
        )
        self.metric = metrics.Accuracy()
        self.drift_detector = drift.ADWIN()

    def predict_and_learn(self, x: dict, y_true=None):
        # x must be dict: {'feature1': value, 'feature2': value}
        y_pred = self.model.predict_one(x)

        if y_true is not None:
            self.model.learn_one(x, y_true)  # NOT fit_one()
            self.metric.update(y_true, y_pred)
            self.drift_detector.update(int(y_pred != y_true))

        return y_pred
```

**Key Points:**
- Features must be dict, not numpy arrays
- Use `learn_one()` not `fit_one()`
- Implement drift detection for regime changes

### Dash Real-time Update Pattern

From Dash library:

```python
import dash
from dash import Dash, dcc, html, callback, Input, Output, State

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Interval(id='interval', interval=1000),  # 1 second
    html.Div(id='live-data')
])

@callback(
    Output('live-data', 'children'),
    Input('interval', 'n_intervals')
)
def update_live_data(n):
    # Fetch and return current data
    return f"Updated at interval {n}"
```

**Key Points:**
- Use `dcc.Interval` for polling updates
- Consider 1-second intervals for dashboard
- Downsample data if performance issues occur

### Session Management Pattern

```python
import pandas_market_calendars as mcal

class SessionManager:
    def __init__(self):
        self.calendars = {
            'CME': mcal.get_calendar('CME'),
            'CBOT': mcal.get_calendar('CBOT'),
            'COMEX': mcal.get_calendar('COMEX'),
            'NYMEX': mcal.get_calendar('NYMEX'),
        }

    def is_market_open(self, exchange: str) -> bool:
        cal = self.calendars.get(exchange)
        schedule = cal.schedule(start_date='today', end_date='today')
        # Check if current time within session

    def minutes_to_close(self, exchange: str) -> int:
        # Calculate time remaining in session
```

## Requirements

### Functional Requirements

1. **IB Connection Management**
   - Description: Establish and maintain connection to TWS/Gateway
   - Acceptance: Connection established, reconnects on disconnect, unique clientId

2. **Contract Auto-Discovery**
   - Description: Automatically discover all tradeable futures contracts on startup
   - Acceptance: System identifies available futures without manual configuration

3. **Real-time Market Data**
   - Description: Subscribe to Level 1 and Level 2 data for all monitored contracts
   - Acceptance: Live ticks and order book data flowing into system

4. **Online Prediction Engine**
   - Description: Generate real-time predictions with confidence scores
   - Acceptance: Predictions generated within milliseconds, confidence tracked

5. **Continuous Learning**
   - Description: Adapt model from every trade outcome, especially losses
   - Acceptance: Model updates after each trade, drift detection active

6. **Confidence-Gated Trading**
   - Description: Only execute trades when confidence exceeds dynamic threshold
   - Acceptance: Low-confidence signals rejected, high-confidence trades executed

7. **Session Awareness**
   - Description: Automatically handle exchange sessions and calendars
   - Acceptance: No entries near session close, positions exited before close

8. **Health Monitoring**
   - Description: Continuously monitor connectivity, data freshness, system health
   - Acceptance: Trading halts on any degradation, resumes on restoration

9. **Position Management**
   - Description: Track all orders, positions, fills, and exposure
   - Acceptance: Complete reconciliation with IB at all times

10. **Real-time Dashboard**
    - Description: Web interface showing all system metrics
    - Acceptance: Dashboard updates in real-time, displays all required metrics

### Edge Cases

1. **IB Disconnection** - Immediately halt trading, attempt reconnection, alert user
2. **Stale Market Data** - Detect tick staleness (>30s), halt trading until fresh
3. **Session Close Approaching** - Exit all positions 5+ minutes before close
4. **Zero Liquidity** - Skip contracts with insufficient order book depth
5. **Rapid Drawdown** - Reduce position size, increase confidence threshold
6. **Model Drift Detected** - Reset learning, increase caution until re-calibrated
7. **Multiple High-Confidence Signals** - Select highest confidence, single position focus
8. **Fill Mismatch** - Reconcile with IB, adjust position tracking

## Implementation Notes

### DO
- Use async/await patterns throughout for concurrent IB operations
- Implement proper connection heartbeat and health checks
- Use River's dict-based features for online learning
- Track confidence at multiple granularities (prediction, contract, regime, time)
- Use pandas_market_calendars for all session/calendar logic
- Implement exponential backoff for reconnection attempts
- Log all trading decisions with full context for post-analysis
- Use `reqMarketDataType(1)` for real-time data (subscription required)

### DON'T
- Use ContFuture for real-time data or orders (historical only!)
- Use numpy arrays with River (use dicts)
- Use `fit_one()` with River (use `learn_one()`)
- Hard-code any thresholds, strategies, or contract symbols
- Ignore market data type configuration
- Skip drift detection (markets exhibit concept drift)
- Use blocking calls in async context

## Development Environment

### Start Services

```bash
# Ensure TWS/Gateway is running with API enabled on port 7497

# Install dependencies
pip install -r requirements.txt

# Start trading bot
python src/main.py

# Start dashboard (separate terminal)
python src/dashboard/app.py
```

### Service URLs
- Dashboard: http://localhost:8050
- TWS API: localhost:7497 (must be running externally)

### Required Environment Variables
- `IB_HOST`: TWS/Gateway host (default: 127.0.0.1)
- `IB_PORT`: TWS/Gateway port (default: 7497)
- `IB_CLIENT_ID`: Unique client ID (default: 1)
- `DASHBOARD_PORT`: Dashboard port (default: 8050)

### Required Python Packages
```
ib_async>=2.1.0
river>=0.23.0
dash>=3.3.0
plotly>=5.18.0
pandas>=2.0.0
pandas_market_calendars>=5.2.4
numpy>=1.26.0
aiohttp>=3.9.0
dash-bootstrap-components>=1.5.0
```

## Success Criteria

The task is complete when:

1. [ ] Bot connects to IB TWS and maintains stable connection
2. [ ] Auto-discovers tradeable futures contracts on startup
3. [ ] Subscribes to Level 1 and Level 2 market data
4. [ ] Generates real-time predictions with confidence scores
5. [ ] Executes trades only when confidence threshold met
6. [ ] Continuously learns from trade outcomes
7. [ ] Properly manages exchange sessions and calendars
8. [ ] Halts trading on any data/connectivity degradation
9. [ ] Dashboard displays all required metrics in real-time
10. [ ] No console errors during normal operation
11. [ ] Full position/order reconciliation with IB

## QA Acceptance Criteria

**CRITICAL**: These criteria must be verified by the QA Agent before sign-off.

### Unit Tests
| Test | File | What to Verify |
|------|------|----------------|
| IB Connection | `tests/test_ib_client.py` | Connect, disconnect, reconnect handling |
| Contract Discovery | `tests/test_contract_discovery.py` | Futures contracts identified correctly |
| Session Manager | `tests/test_session_manager.py` | Calendar queries, session detection |
| Online Learner | `tests/test_online_learner.py` | Predict, learn, drift detection |
| Confidence Tracker | `tests/test_confidence.py` | Confidence calculation, thresholds |
| Position Manager | `tests/test_position_manager.py` | Order tracking, reconciliation |

### Integration Tests
| Test | Services | What to Verify |
|------|----------|----------------|
| Market Data Flow | ib_client ↔ market_data | Live ticks received and processed |
| Prediction Pipeline | market_data ↔ predictor | Features extracted, predictions made |
| Trade Execution | predictor ↔ executor ↔ ib_client | Orders placed when confidence high |
| Learning Loop | executor ↔ online_learner | Model updates after fill |
| Health Check | health_monitor ↔ all | Degradation detected, trading halted |

### End-to-End Tests
| Flow | Steps | Expected Outcome |
|------|-------|------------------|
| Startup Discovery | 1. Start bot 2. Wait for discovery | Contracts listed, data subscribed |
| Prediction Generation | 1. Receive ticks 2. Extract features 3. Predict | Predictions with confidence scores |
| Trade Lifecycle | 1. High confidence 2. Place order 3. Fill 4. Learn | Order filled, model updated |
| Session Close | 1. Approach close 2. Monitor | Positions exited, no new entries |
| Disconnect Recovery | 1. Simulate disconnect 2. Reconnect | Trading halts, then resumes |

### Browser Verification (Dashboard)
| Page/Component | URL | Checks |
|----------------|-----|--------|
| Main Dashboard | `http://localhost:8050/` | All panels render, data updates |
| Confidence Panel | `http://localhost:8050/` | Current and historical confidence shown |
| Positions Panel | `http://localhost:8050/` | Open positions, exposure displayed |
| P&L Panel | `http://localhost:8050/` | Session P&L, Sharpe ratio shown |
| Contracts Panel | `http://localhost:8050/` | Active contracts listed |

### System Health Verification
| Check | Method | Expected |
|-------|--------|----------|
| IB Connection | Check connection status | Connected, heartbeat active |
| Data Freshness | Check last tick timestamp | Within 30 seconds |
| Memory Usage | Monitor process | Below 80% of system memory |
| CPU Usage | Monitor process | Below 80% of system CPU |
| Model Drift | Check drift detector | No drift or handled appropriately |

### QA Sign-off Requirements
- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] All E2E tests pass
- [ ] Browser verification complete
- [ ] System health checks pass
- [ ] No regressions in existing functionality
- [ ] Code follows established patterns
- [ ] No security vulnerabilities introduced
- [ ] Resource usage within 80% target
- [ ] Logging captures all trading decisions

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Main Orchestrator                         │
│                         (src/main.py)                           │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐   ┌─────────────────┐   ┌─────────────────┐
│   IB Client   │   │ Health Monitor  │   │    Dashboard    │
│ (ib_async)    │   │                 │   │    (Dash)       │
└───────────────┘   └─────────────────┘   └─────────────────┘
        │
        ├── Contract Discovery
        ├── Market Data (L1 + L2)
        └── Order Execution
                │
        ┌───────┴───────┐
        │               │
        ▼               ▼
┌───────────────┐ ┌─────────────────┐
│   Predictor   │ │Position Manager │
│   (River)     │ │                 │
└───────────────┘ └─────────────────┘
        │
        ▼
┌───────────────┐
│   Executor    │
│(Confidence-   │
│   Gated)      │
└───────────────┘
        │
        ▼
┌───────────────┐
│Online Learner │
│ + Drift Det.  │
└───────────────┘
```

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| IB API limits (100 data lines) | Prioritize highest-volume contracts |
| Stale data causing bad trades | Tick freshness monitoring with hard gate |
| Model overfitting to recent data | Drift detection, adaptive learning rate |
| Session close position risk | Exit 5+ minutes before close |
| Rapid drawdown | Dynamic confidence threshold increase |
| Connection instability | Exponential backoff reconnection |
