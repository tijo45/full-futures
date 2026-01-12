"""
Unit tests for SessionManager - Exchange calendar and session handling.

Tests cover:
- Market open/close detection
- Session status queries
- Time remaining calculations
- Entry/exit decision logic
- Calendar mapping
"""

import pytest
import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch
import pandas as pd

sys.path.insert(0, '.')


@pytest.fixture
def mock_config():
    """Mock configuration for session manager."""
    config = MagicMock()
    config.SESSION_CLOSE_BUFFER_MINUTES = 5
    return config


@pytest.fixture
def mock_calendar():
    """Create a mock market calendar."""
    calendar = MagicMock()
    return calendar


@pytest.fixture
def sample_schedule():
    """Create a sample schedule DataFrame."""
    now = pd.Timestamp.now(tz='UTC')
    market_open = now - timedelta(hours=4)
    market_close = now + timedelta(hours=4)

    schedule = pd.DataFrame({
        'market_open': [market_open],
        'market_close': [market_close]
    })
    return schedule


@pytest.fixture
def closed_schedule():
    """Create a schedule where market is closed."""
    now = pd.Timestamp.now(tz='UTC')
    market_open = now - timedelta(hours=8)
    market_close = now - timedelta(hours=4)

    schedule = pd.DataFrame({
        'market_open': [market_open],
        'market_close': [market_close]
    })
    return schedule


class TestSessionManagerInitialization:
    """Tests for SessionManager initialization."""

    def test_initialization_loads_calendars(self, mock_config):
        """Test SessionManager loads exchange calendars on init."""
        with patch('src.core.session_manager.get_config', return_value=mock_config):
            with patch('src.core.session_manager.mcal') as mock_mcal:
                mock_mcal.get_calendar.return_value = MagicMock()

                from src.core.session_manager import SessionManager

                sm = SessionManager()

                # Should have loaded calendars for supported exchanges
                assert len(sm.calendars) > 0
                assert mock_mcal.get_calendar.called

    def test_supported_exchanges(self, mock_config):
        """Test supported exchanges are defined."""
        with patch('src.core.session_manager.get_config', return_value=mock_config):
            with patch('src.core.session_manager.mcal') as mock_mcal:
                mock_mcal.get_calendar.return_value = MagicMock()

                from src.core.session_manager import SessionManager

                expected = ['CME', 'CBOT', 'COMEX', 'NYMEX']
                assert SessionManager.SUPPORTED_EXCHANGES == expected

    def test_calendar_mapping(self, mock_config):
        """Test calendar mapping for PMC 5.x naming."""
        with patch('src.core.session_manager.get_config', return_value=mock_config):
            with patch('src.core.session_manager.mcal') as mock_mcal:
                mock_mcal.get_calendar.return_value = MagicMock()

                from src.core.session_manager import SessionManager

                mapping = SessionManager.CALENDAR_MAPPING
                assert mapping['CME'] == 'CME_Equity'
                assert mapping['CBOT'] == 'CBOT_Equity'
                assert mapping['COMEX'] == 'CMEGlobex_Metals'
                assert mapping['NYMEX'] == 'CMEGlobex_Energy'


class TestSessionManagerMarketStatus:
    """Tests for market open/close detection."""

    def test_is_market_open_true(self, mock_config, sample_schedule):
        """Test is_market_open returns True during trading hours."""
        with patch('src.core.session_manager.get_config', return_value=mock_config):
            with patch('src.core.session_manager.mcal') as mock_mcal:
                mock_calendar = MagicMock()
                mock_calendar.schedule.return_value = sample_schedule
                mock_mcal.get_calendar.return_value = mock_calendar

                from src.core.session_manager import SessionManager

                sm = SessionManager()
                result = sm.is_market_open('CME')

                assert result is True

    def test_is_market_open_false_when_closed(self, mock_config, closed_schedule):
        """Test is_market_open returns False after trading hours."""
        with patch('src.core.session_manager.get_config', return_value=mock_config):
            with patch('src.core.session_manager.mcal') as mock_mcal:
                mock_calendar = MagicMock()
                mock_calendar.schedule.return_value = closed_schedule
                mock_mcal.get_calendar.return_value = mock_calendar

                from src.core.session_manager import SessionManager

                sm = SessionManager()
                result = sm.is_market_open('CME')

                assert result is False

    def test_is_market_open_holiday(self, mock_config):
        """Test is_market_open returns False on holiday."""
        with patch('src.core.session_manager.get_config', return_value=mock_config):
            with patch('src.core.session_manager.mcal') as mock_mcal:
                mock_calendar = MagicMock()
                mock_calendar.schedule.return_value = pd.DataFrame()  # Empty = holiday
                mock_mcal.get_calendar.return_value = mock_calendar

                from src.core.session_manager import SessionManager

                sm = SessionManager()
                result = sm.is_market_open('CME')

                assert result is False

    def test_is_market_open_unknown_exchange(self, mock_config):
        """Test is_market_open returns False for unknown exchange."""
        with patch('src.core.session_manager.get_config', return_value=mock_config):
            with patch('src.core.session_manager.mcal') as mock_mcal:
                mock_mcal.get_calendar.return_value = MagicMock()

                from src.core.session_manager import SessionManager

                sm = SessionManager()
                sm.calendars = {}  # Clear calendars

                result = sm.is_market_open('UNKNOWN')

                assert result is False


class TestSessionManagerSessionStatus:
    """Tests for session status queries."""

    def test_get_session_status_open(self, mock_config, sample_schedule):
        """Test session status is OPEN during trading hours."""
        with patch('src.core.session_manager.get_config', return_value=mock_config):
            with patch('src.core.session_manager.mcal') as mock_mcal:
                mock_calendar = MagicMock()
                mock_calendar.schedule.return_value = sample_schedule
                mock_mcal.get_calendar.return_value = mock_calendar

                from src.core.session_manager import SessionManager, SessionStatus

                sm = SessionManager()
                status = sm.get_session_status('CME')

                assert status == SessionStatus.OPEN

    def test_get_session_status_post_market(self, mock_config, closed_schedule):
        """Test session status is POST_MARKET after close."""
        with patch('src.core.session_manager.get_config', return_value=mock_config):
            with patch('src.core.session_manager.mcal') as mock_mcal:
                mock_calendar = MagicMock()
                mock_calendar.schedule.return_value = closed_schedule
                mock_mcal.get_calendar.return_value = mock_calendar

                from src.core.session_manager import SessionManager, SessionStatus

                sm = SessionManager()
                status = sm.get_session_status('CME')

                assert status == SessionStatus.POST_MARKET

    def test_get_session_status_holiday(self, mock_config):
        """Test session status is HOLIDAY when no schedule."""
        with patch('src.core.session_manager.get_config', return_value=mock_config):
            with patch('src.core.session_manager.mcal') as mock_mcal:
                mock_calendar = MagicMock()
                mock_calendar.schedule.return_value = pd.DataFrame()
                mock_mcal.get_calendar.return_value = mock_calendar

                from src.core.session_manager import SessionManager, SessionStatus

                sm = SessionManager()
                # Patch datetime to be a weekday
                with patch('src.core.session_manager.datetime') as mock_dt:
                    mock_dt.now.return_value = datetime(2024, 12, 25)  # Weekday holiday

                    status = sm.get_session_status('CME')

                    assert status == SessionStatus.HOLIDAY


class TestSessionManagerTimeCalculations:
    """Tests for time remaining calculations."""

    def test_minutes_to_close_positive(self, mock_config, sample_schedule):
        """Test minutes_to_close returns positive value during session."""
        with patch('src.core.session_manager.get_config', return_value=mock_config):
            with patch('src.core.session_manager.mcal') as mock_mcal:
                mock_calendar = MagicMock()
                mock_calendar.schedule.return_value = sample_schedule
                mock_mcal.get_calendar.return_value = mock_calendar

                from src.core.session_manager import SessionManager

                sm = SessionManager()
                minutes = sm.minutes_to_close('CME')

                assert minutes > 0
                assert minutes <= 240  # 4 hours = 240 minutes

    def test_minutes_to_close_negative_when_closed(self, mock_config, closed_schedule):
        """Test minutes_to_close returns -1 when market closed."""
        with patch('src.core.session_manager.get_config', return_value=mock_config):
            with patch('src.core.session_manager.mcal') as mock_mcal:
                mock_calendar = MagicMock()
                mock_calendar.schedule.return_value = closed_schedule
                mock_mcal.get_calendar.return_value = mock_calendar

                from src.core.session_manager import SessionManager

                sm = SessionManager()
                minutes = sm.minutes_to_close('CME')

                assert minutes == -1

    def test_minutes_since_open_positive(self, mock_config, sample_schedule):
        """Test minutes_since_open returns positive value during session."""
        with patch('src.core.session_manager.get_config', return_value=mock_config):
            with patch('src.core.session_manager.mcal') as mock_mcal:
                mock_calendar = MagicMock()
                mock_calendar.schedule.return_value = sample_schedule
                mock_mcal.get_calendar.return_value = mock_calendar

                from src.core.session_manager import SessionManager

                sm = SessionManager()
                minutes = sm.minutes_since_open('CME')

                assert minutes >= 0
                assert minutes <= 240  # 4 hours = 240 minutes


class TestSessionManagerEntryExitDecisions:
    """Tests for entry and exit decision methods."""

    def test_should_allow_entry_true_early_in_session(self, mock_config, sample_schedule):
        """Test entry allowed early in session."""
        with patch('src.core.session_manager.get_config', return_value=mock_config):
            with patch('src.core.session_manager.mcal') as mock_mcal:
                mock_calendar = MagicMock()
                mock_calendar.schedule.return_value = sample_schedule
                mock_mcal.get_calendar.return_value = mock_calendar

                from src.core.session_manager import SessionManager

                sm = SessionManager()
                result = sm.should_allow_entry('CME')

                # Should allow entry when > 5 minutes to close
                assert result is True

    def test_should_allow_entry_false_when_closed(self, mock_config, closed_schedule):
        """Test entry not allowed when market closed."""
        with patch('src.core.session_manager.get_config', return_value=mock_config):
            with patch('src.core.session_manager.mcal') as mock_mcal:
                mock_calendar = MagicMock()
                mock_calendar.schedule.return_value = closed_schedule
                mock_mcal.get_calendar.return_value = mock_calendar

                from src.core.session_manager import SessionManager

                sm = SessionManager()
                result = sm.should_allow_entry('CME')

                assert result is False

    def test_should_exit_positions_near_close(self, mock_config):
        """Test exit signal when near close."""
        # Create schedule where we're within 5 minutes of close
        now = pd.Timestamp.now(tz='UTC')
        market_open = now - timedelta(hours=8)
        market_close = now + timedelta(minutes=3)  # 3 minutes to close

        near_close_schedule = pd.DataFrame({
            'market_open': [market_open],
            'market_close': [market_close]
        })

        with patch('src.core.session_manager.get_config', return_value=mock_config):
            with patch('src.core.session_manager.mcal') as mock_mcal:
                mock_calendar = MagicMock()
                mock_calendar.schedule.return_value = near_close_schedule
                mock_mcal.get_calendar.return_value = mock_calendar

                from src.core.session_manager import SessionManager

                sm = SessionManager()
                result = sm.should_exit_positions('CME')

                # Should signal exit when <= 5 minutes to close
                assert result is True

    def test_should_exit_positions_when_closed(self, mock_config, closed_schedule):
        """Test exit signal when already closed."""
        with patch('src.core.session_manager.get_config', return_value=mock_config):
            with patch('src.core.session_manager.mcal') as mock_mcal:
                mock_calendar = MagicMock()
                mock_calendar.schedule.return_value = closed_schedule
                mock_mcal.get_calendar.return_value = mock_calendar

                from src.core.session_manager import SessionManager

                sm = SessionManager()
                result = sm.should_exit_positions('CME')

                # Should always signal exit when market is closed
                assert result is True


class TestSessionManagerSessionInfo:
    """Tests for session info methods."""

    def test_get_session_info(self, mock_config, sample_schedule):
        """Test getting comprehensive session info."""
        with patch('src.core.session_manager.get_config', return_value=mock_config):
            with patch('src.core.session_manager.mcal') as mock_mcal:
                mock_calendar = MagicMock()
                mock_calendar.schedule.return_value = sample_schedule
                mock_mcal.get_calendar.return_value = mock_calendar

                from src.core.session_manager import SessionManager

                sm = SessionManager()
                info = sm.get_session_info('CME')

                assert 'exchange' in info
                assert info['exchange'] == 'CME'
                assert 'status' in info
                assert 'is_open' in info
                assert 'is_trading_day' in info
                assert 'minutes_to_close' in info
                assert 'allow_entry' in info
                assert 'should_exit' in info
                assert 'close_buffer_minutes' in info

    def test_get_all_sessions_info(self, mock_config, sample_schedule):
        """Test getting info for all supported exchanges."""
        with patch('src.core.session_manager.get_config', return_value=mock_config):
            with patch('src.core.session_manager.mcal') as mock_mcal:
                mock_calendar = MagicMock()
                mock_calendar.schedule.return_value = sample_schedule
                mock_mcal.get_calendar.return_value = mock_calendar

                from src.core.session_manager import SessionManager

                sm = SessionManager()
                all_info = sm.get_all_sessions_info()

                # Should have info for each supported exchange
                for exchange in SessionManager.SUPPORTED_EXCHANGES:
                    assert exchange in all_info
                    assert 'exchange' in all_info[exchange]

    def test_is_trading_day_true(self, mock_config, sample_schedule):
        """Test is_trading_day returns True on trading day."""
        with patch('src.core.session_manager.get_config', return_value=mock_config):
            with patch('src.core.session_manager.mcal') as mock_mcal:
                mock_calendar = MagicMock()
                mock_calendar.schedule.return_value = sample_schedule
                mock_mcal.get_calendar.return_value = mock_calendar

                from src.core.session_manager import SessionManager

                sm = SessionManager()
                result = sm.is_trading_day('CME')

                assert result is True

    def test_is_trading_day_false_on_holiday(self, mock_config):
        """Test is_trading_day returns False on holiday."""
        with patch('src.core.session_manager.get_config', return_value=mock_config):
            with patch('src.core.session_manager.mcal') as mock_mcal:
                mock_calendar = MagicMock()
                mock_calendar.schedule.return_value = pd.DataFrame()  # Empty = holiday
                mock_mcal.get_calendar.return_value = mock_calendar

                from src.core.session_manager import SessionManager

                sm = SessionManager()
                result = sm.is_trading_day('CME')

                assert result is False


class TestSessionManagerSessionTimes:
    """Tests for session time retrieval."""

    def test_get_session_times(self, mock_config, sample_schedule):
        """Test getting session open/close times."""
        with patch('src.core.session_manager.get_config', return_value=mock_config):
            with patch('src.core.session_manager.mcal') as mock_mcal:
                mock_calendar = MagicMock()
                mock_calendar.schedule.return_value = sample_schedule
                mock_mcal.get_calendar.return_value = mock_calendar

                from src.core.session_manager import SessionManager

                sm = SessionManager()
                times = sm.get_session_times('CME')

                assert times is not None
                assert len(times) == 2  # (open, close)
                assert times[0] < times[1]  # open before close

    def test_get_session_times_holiday(self, mock_config):
        """Test get_session_times returns None on holiday."""
        with patch('src.core.session_manager.get_config', return_value=mock_config):
            with patch('src.core.session_manager.mcal') as mock_mcal:
                mock_calendar = MagicMock()
                mock_calendar.schedule.return_value = pd.DataFrame()
                mock_mcal.get_calendar.return_value = mock_calendar

                from src.core.session_manager import SessionManager

                sm = SessionManager()
                times = sm.get_session_times('CME')

                assert times is None


class TestSessionManagerCaching:
    """Tests for schedule caching."""

    def test_schedule_cache_used(self, mock_config, sample_schedule):
        """Test that schedule cache is used for repeated queries."""
        with patch('src.core.session_manager.get_config', return_value=mock_config):
            with patch('src.core.session_manager.mcal') as mock_mcal:
                mock_calendar = MagicMock()
                mock_calendar.schedule.return_value = sample_schedule
                mock_mcal.get_calendar.return_value = mock_calendar

                from src.core.session_manager import SessionManager

                sm = SessionManager()

                # First call
                sm.is_market_open('CME')
                call_count_after_first = mock_calendar.schedule.call_count

                # Second call should use cache
                sm.is_market_open('CME')
                call_count_after_second = mock_calendar.schedule.call_count

                # Schedule should only be called once due to caching
                assert call_count_after_second == call_count_after_first


class TestSessionStatus:
    """Tests for SessionStatus enum."""

    def test_session_status_values(self):
        """Test SessionStatus enum values."""
        from src.core.session_manager import SessionStatus

        assert SessionStatus.OPEN.value == 'open'
        assert SessionStatus.CLOSED.value == 'closed'
        assert SessionStatus.PRE_MARKET.value == 'pre_market'
        assert SessionStatus.POST_MARKET.value == 'post_market'
        assert SessionStatus.HOLIDAY.value == 'holiday'
        assert SessionStatus.UNKNOWN.value == 'unknown'


class TestCloseBufferProperty:
    """Tests for close_buffer_minutes property."""

    def test_close_buffer_minutes_property(self, mock_config):
        """Test close_buffer_minutes returns config value."""
        mock_config.SESSION_CLOSE_BUFFER_MINUTES = 10

        with patch('src.core.session_manager.get_config', return_value=mock_config):
            with patch('src.core.session_manager.mcal') as mock_mcal:
                mock_mcal.get_calendar.return_value = MagicMock()

                from src.core.session_manager import SessionManager

                sm = SessionManager()

                assert sm.close_buffer_minutes == 10
