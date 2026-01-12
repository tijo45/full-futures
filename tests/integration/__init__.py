"""
Integration tests for the autonomous futures trading system.

Tests component interactions:
- test_market_data_flow.py: IBClient ↔ MarketDataHandler
- test_prediction_pipeline.py: MarketData ↔ FeatureEngine ↔ Predictor
- test_trade_execution.py: Predictor ↔ Executor ↔ IBClient
- test_e2e_integration.py: Full end-to-end verification
"""
