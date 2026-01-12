"""
Session Manager - Exchange calendar and session handling.

Uses pandas_market_calendars to manage exchange schedules, determine market
hours, and provide session awareness for trading decisions.

Provides:
- Market open/close detection
- Time remaining until session close
- Entry prevention near session close
- Position exit signals before session end
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from enum import Enum

import pandas as pd
import pandas_market_calendars as mcal

from config import get_config


class SessionStatus(Enum):
    """Session status enumeration."""
    OPEN = "open"
    CLOSED = "closed"
    PRE_MARKET = "pre_market"
    POST_MARKET = "post_market"
    HOLIDAY = "holiday"
    UNKNOWN = "unknown"


class SessionManager:
    """
    Exchange calendar and session handling manager.

    Uses pandas_market_calendars for accurate exchange schedules.
    Provides session awareness for trading decisions including:
    - Market open/close detection
    - Time remaining calculations
    - Entry prevention near close
    - Position exit signals

    Supported Exchanges:
    - CME (Chicago Mercantile Exchange)
    - CBOT (Chicago Board of Trade)
    - COMEX (Commodity Exchange)
    - NYMEX (New York Mercantile Exchange)

    Usage:
        sm = SessionManager()
        if sm.is_market_open('CME'):
            if sm.should_allow_entry('CME'):
                # Safe to enter new positions
            if sm.should_exit_positions('CME'):
                # Time to exit positions
    """

    # Supported exchanges with their calendar names
    SUPPORTED_EXCHANGES = ['CME', 'CBOT', 'COMEX', 'NYMEX']

    # Mapping from exchange names to pandas_market_calendars calendar names
    # Note: pandas_market_calendars uses different naming convention for CME Group exchanges
    CALENDAR_MAPPING = {
        'CME': 'CME_Equity',          # CME equity/index futures calendar
        'CBOT': 'CBOT_Equity',         # CBOT equity/treasury futures calendar
        'COMEX': 'CMEGlobex_Metals',   # COMEX metals futures (Gold, Silver, Copper)
        'NYMEX': 'CMEGlobex_Energy',   # NYMEX energy futures (Crude, NatGas)
    }

    def __init__(self):
        """
        Initialize SessionManager with exchange calendars.

        Loads calendars for all supported futures exchanges:
        CME, CBOT, COMEX, and NYMEX.
        """
        self._config = get_config()
        self._logger = logging.getLogger(__name__)

        # Load exchange calendars using the mapping
        self.calendars: Dict[str, mcal.MarketCalendar] = {}
        for exchange in self.SUPPORTED_EXCHANGES:
            calendar_name = self.CALENDAR_MAPPING.get(exchange, exchange)
            try:
                self.calendars[exchange] = mcal.get_calendar(calendar_name)
                self._logger.debug(f"Loaded calendar '{calendar_name}' for {exchange}")
            except Exception as e:
                self._logger.error(f"Failed to load calendar for {exchange}: {e}")

        # Cache for today's schedules (refreshed daily)
        self._schedule_cache: Dict[str, pd.DataFrame] = {}
        self._cache_date: Optional[datetime] = None

        self._logger.info(
            f"SessionManager initialized with {len(self.calendars)} exchanges"
        )

    @property
    def close_buffer_minutes(self) -> int:
        """Get the configured buffer time before session close (in minutes)."""
        return self._config.SESSION_CLOSE_BUFFER_MINUTES

    def _get_schedule(
        self,
        exchange: str,
        date: Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        """
        Get trading schedule for an exchange on a given date.

        Uses caching to avoid repeated calendar queries for the same day.

        Args:
            exchange: Exchange name (CME, CBOT, COMEX, NYMEX)
            date: Date to check (default: today)

        Returns:
            DataFrame with market_open and market_close columns, or None if holiday
        """
        if exchange not in self.calendars:
            self._logger.warning(f"Unknown exchange: {exchange}")
            return None

        if date is None:
            date = datetime.now()

        date_str = date.strftime('%Y-%m-%d')

        # Check if we need to refresh cache (new day)
        cache_key = f"{exchange}_{date_str}"
        today = datetime.now().date()

        if self._cache_date != today:
            self._schedule_cache.clear()
            self._cache_date = today

        # Return cached schedule if available
        if cache_key in self._schedule_cache:
            return self._schedule_cache[cache_key]

        # Query calendar
        try:
            calendar = self.calendars[exchange]
            schedule = calendar.schedule(
                start_date=date_str,
                end_date=date_str
            )

            if schedule.empty:
                # No trading on this day (holiday or weekend)
                self._schedule_cache[cache_key] = None
                return None

            self._schedule_cache[cache_key] = schedule
            return schedule

        except Exception as e:
            self._logger.error(f"Error getting schedule for {exchange}: {e}")
            return None

    def is_market_open(self, exchange: str) -> bool:
        """
        Check if the market is currently open for an exchange.

        Args:
            exchange: Exchange name (CME, CBOT, COMEX, NYMEX)

        Returns:
            True if market is currently open, False otherwise
        """
        schedule = self._get_schedule(exchange)

        if schedule is None or schedule.empty:
            return False

        now = pd.Timestamp.now(tz='UTC')

        try:
            market_open = schedule.iloc[0]['market_open']
            market_close = schedule.iloc[0]['market_close']

            # Ensure timezone awareness
            if market_open.tzinfo is None:
                market_open = market_open.tz_localize('UTC')
            if market_close.tzinfo is None:
                market_close = market_close.tz_localize('UTC')

            return market_open <= now <= market_close

        except Exception as e:
            self._logger.error(f"Error checking market open for {exchange}: {e}")
            return False

    def get_session_status(self, exchange: str) -> SessionStatus:
        """
        Get the current session status for an exchange.

        Args:
            exchange: Exchange name (CME, CBOT, COMEX, NYMEX)

        Returns:
            SessionStatus enum value
        """
        schedule = self._get_schedule(exchange)

        if schedule is None or schedule.empty:
            # Check if it's a holiday or weekend
            today = datetime.now()
            if today.weekday() >= 5:  # Saturday or Sunday
                return SessionStatus.CLOSED
            return SessionStatus.HOLIDAY

        now = pd.Timestamp.now(tz='UTC')

        try:
            market_open = schedule.iloc[0]['market_open']
            market_close = schedule.iloc[0]['market_close']

            # Ensure timezone awareness
            if market_open.tzinfo is None:
                market_open = market_open.tz_localize('UTC')
            if market_close.tzinfo is None:
                market_close = market_close.tz_localize('UTC')

            if now < market_open:
                return SessionStatus.PRE_MARKET
            elif now > market_close:
                return SessionStatus.POST_MARKET
            else:
                return SessionStatus.OPEN

        except Exception as e:
            self._logger.error(f"Error getting session status for {exchange}: {e}")
            return SessionStatus.UNKNOWN

    def minutes_to_close(self, exchange: str) -> int:
        """
        Calculate minutes remaining until session close.

        Args:
            exchange: Exchange name (CME, CBOT, COMEX, NYMEX)

        Returns:
            Minutes until close, or -1 if market is closed/error
        """
        schedule = self._get_schedule(exchange)

        if schedule is None or schedule.empty:
            return -1

        now = pd.Timestamp.now(tz='UTC')

        try:
            market_close = schedule.iloc[0]['market_close']

            # Ensure timezone awareness
            if market_close.tzinfo is None:
                market_close = market_close.tz_localize('UTC')

            if now > market_close:
                return -1

            time_remaining = market_close - now
            minutes_remaining = int(time_remaining.total_seconds() / 60)

            return max(0, minutes_remaining)

        except Exception as e:
            self._logger.error(f"Error calculating minutes to close for {exchange}: {e}")
            return -1

    def minutes_since_open(self, exchange: str) -> int:
        """
        Calculate minutes since session opened.

        Args:
            exchange: Exchange name (CME, CBOT, COMEX, NYMEX)

        Returns:
            Minutes since open, or -1 if market is closed/error
        """
        schedule = self._get_schedule(exchange)

        if schedule is None or schedule.empty:
            return -1

        now = pd.Timestamp.now(tz='UTC')

        try:
            market_open = schedule.iloc[0]['market_open']

            # Ensure timezone awareness
            if market_open.tzinfo is None:
                market_open = market_open.tz_localize('UTC')

            if now < market_open:
                return -1

            time_elapsed = now - market_open
            minutes_elapsed = int(time_elapsed.total_seconds() / 60)

            return max(0, minutes_elapsed)

        except Exception as e:
            self._logger.error(f"Error calculating minutes since open for {exchange}: {e}")
            return -1

    def should_allow_entry(self, exchange: str) -> bool:
        """
        Check if new position entries should be allowed.

        Prevents entries when too close to session close, based on the
        configured close buffer time (default 5 minutes).

        Args:
            exchange: Exchange name (CME, CBOT, COMEX, NYMEX)

        Returns:
            True if entries are allowed, False otherwise
        """
        if not self.is_market_open(exchange):
            return False

        minutes_remaining = self.minutes_to_close(exchange)

        if minutes_remaining < 0:
            return False

        # Don't allow entries within the buffer period
        return minutes_remaining > self.close_buffer_minutes

    def should_exit_positions(self, exchange: str) -> bool:
        """
        Check if positions should be exited due to approaching session close.

        Signals position exit when within the close buffer time (default 5 minutes).

        Args:
            exchange: Exchange name (CME, CBOT, COMEX, NYMEX)

        Returns:
            True if positions should be exited, False otherwise
        """
        if not self.is_market_open(exchange):
            # Already closed - should have exited
            return True

        minutes_remaining = self.minutes_to_close(exchange)

        if minutes_remaining < 0:
            return True

        # Signal exit when within the buffer period
        return minutes_remaining <= self.close_buffer_minutes

    def get_session_times(
        self,
        exchange: str,
        date: Optional[datetime] = None
    ) -> Optional[Tuple[datetime, datetime]]:
        """
        Get the open and close times for a session.

        Args:
            exchange: Exchange name (CME, CBOT, COMEX, NYMEX)
            date: Date to check (default: today)

        Returns:
            Tuple of (open_time, close_time) as datetime objects, or None if closed
        """
        schedule = self._get_schedule(exchange, date)

        if schedule is None or schedule.empty:
            return None

        try:
            market_open = schedule.iloc[0]['market_open']
            market_close = schedule.iloc[0]['market_close']

            return (
                market_open.to_pydatetime(),
                market_close.to_pydatetime()
            )

        except Exception as e:
            self._logger.error(f"Error getting session times for {exchange}: {e}")
            return None

    def get_next_session(
        self,
        exchange: str,
        from_date: Optional[datetime] = None
    ) -> Optional[Tuple[datetime, datetime]]:
        """
        Get the next trading session times.

        Args:
            exchange: Exchange name (CME, CBOT, COMEX, NYMEX)
            from_date: Start searching from this date (default: now)

        Returns:
            Tuple of (open_time, close_time) for next session, or None
        """
        if exchange not in self.calendars:
            return None

        if from_date is None:
            from_date = datetime.now()

        # Check up to 7 days ahead (to handle weekends and holidays)
        for days_ahead in range(7):
            check_date = from_date + timedelta(days=days_ahead)
            schedule = self._get_schedule(exchange, check_date)

            if schedule is not None and not schedule.empty:
                try:
                    market_open = schedule.iloc[0]['market_open']
                    market_close = schedule.iloc[0]['market_close']

                    # Skip if this session has already closed
                    now = pd.Timestamp.now(tz='UTC')
                    if market_close.tzinfo is None:
                        market_close_tz = market_close.tz_localize('UTC')
                    else:
                        market_close_tz = market_close

                    if market_close_tz > now:
                        return (
                            market_open.to_pydatetime(),
                            market_close.to_pydatetime()
                        )
                except Exception:
                    continue

        return None

    def is_trading_day(
        self,
        exchange: str,
        date: Optional[datetime] = None
    ) -> bool:
        """
        Check if a given date is a trading day for an exchange.

        Args:
            exchange: Exchange name (CME, CBOT, COMEX, NYMEX)
            date: Date to check (default: today)

        Returns:
            True if it's a trading day, False otherwise
        """
        schedule = self._get_schedule(exchange, date)
        return schedule is not None and not schedule.empty

    def get_session_info(self, exchange: str) -> Dict:
        """
        Get comprehensive session information for an exchange.

        Returns a dictionary with all session-related data useful for
        trading decisions and dashboard display.

        Args:
            exchange: Exchange name (CME, CBOT, COMEX, NYMEX)

        Returns:
            Dictionary with session info including status, times, and flags
        """
        status = self.get_session_status(exchange)
        is_open = status == SessionStatus.OPEN
        session_times = self.get_session_times(exchange)

        info = {
            'exchange': exchange,
            'status': status.value,
            'is_open': is_open,
            'is_trading_day': self.is_trading_day(exchange),
            'minutes_to_close': self.minutes_to_close(exchange) if is_open else None,
            'minutes_since_open': self.minutes_since_open(exchange) if is_open else None,
            'allow_entry': self.should_allow_entry(exchange),
            'should_exit': self.should_exit_positions(exchange) if is_open else False,
            'close_buffer_minutes': self.close_buffer_minutes,
            'session_open': session_times[0] if session_times else None,
            'session_close': session_times[1] if session_times else None,
        }

        return info

    def get_all_sessions_info(self) -> Dict[str, Dict]:
        """
        Get session information for all supported exchanges.

        Returns:
            Dictionary mapping exchange names to their session info
        """
        return {
            exchange: self.get_session_info(exchange)
            for exchange in self.SUPPORTED_EXCHANGES
        }
