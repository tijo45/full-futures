# Gotchas & Pitfalls

Things to watch out for in this codebase.

## [2026-01-12 17:10]
ib_async package requires Python 3.10+. For Python 3.9, use ib_insync instead. The IBClient code supports both via try/except imports.

_Context: Installing IB API library for Interactive Brokers connection_

## [2026-01-12 17:13]
CRITICAL: Use Future() NOT ContFuture() for real-time market data and orders. ContFuture is for historical data ONLY. Using ContFuture for live trading will fail.

_Context: Contract discovery and market data subscription in IB API_

## [2026-01-12 17:18]
pandas_market_calendars 5.x uses different calendar names than older versions. CME, CBOT, COMEX, NYMEX are NOT valid calendar names. Use: CME_Equity, CBOT_Equity, CMEGlobex_Metals, CMEGlobex_Energy (or product-specific calendars like CMEGlobex_Crude, CMEGlobex_Gold)

_Context: Loading exchange calendars with pandas_market_calendars 5.x_

## [2026-01-12 17:31]
River optimizers are in `river.optim` module, NOT `river.linear_model.optim`. Import with: `from river import optim` then use `optim.SGD(learning_rate)`.

_Context: Setting up River LogisticRegression with custom optimizer in OnlineLearner_
