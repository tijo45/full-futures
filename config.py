"""
Dynamic configuration module with environment variable support.

All configuration values can be overridden via environment variables.
No hard-coded values - everything is configurable.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """
    Configuration class for the autonomous futures trading bot.

    All values can be overridden via environment variables.
    Instance attributes are set dynamically from environment or defaults.
    """

    # IB Connection Settings
    IB_HOST: str = None
    IB_PORT: int = None
    IB_CLIENT_ID: int = None

    # Dashboard Settings
    DASHBOARD_PORT: int = None
    DASHBOARD_HOST: str = None
    DASHBOARD_DEBUG: bool = None

    # Data Integrity Settings
    DATA_STALENESS_THRESHOLD_SECONDS: int = None

    # Session Management Settings
    SESSION_CLOSE_BUFFER_MINUTES: int = None

    # Health Monitoring Settings
    HEALTH_CHECK_INTERVAL_SECONDS: int = None
    RECONNECT_MAX_ATTEMPTS: int = None
    RECONNECT_BASE_DELAY_SECONDS: float = None

    # Logging Settings
    LOG_LEVEL: str = None
    LOG_FORMAT: str = None

    def __post_init__(self):
        """Load configuration from environment variables with defaults."""

        # IB Connection Settings
        self.IB_HOST = os.environ.get('IB_HOST', '127.0.0.1')
        self.IB_PORT = int(os.environ.get('IB_PORT', '7497'))
        self.IB_CLIENT_ID = int(os.environ.get('IB_CLIENT_ID', '1'))

        # Dashboard Settings
        self.DASHBOARD_PORT = int(os.environ.get('DASHBOARD_PORT', '8050'))
        self.DASHBOARD_HOST = os.environ.get('DASHBOARD_HOST', '0.0.0.0')
        self.DASHBOARD_DEBUG = os.environ.get('DASHBOARD_DEBUG', 'false').lower() == 'true'

        # Data Integrity Settings
        # Default: 30 seconds before data is considered stale
        self.DATA_STALENESS_THRESHOLD_SECONDS = int(
            os.environ.get('DATA_STALENESS_THRESHOLD_SECONDS', '30')
        )

        # Session Management Settings
        # Default: Exit positions 5 minutes before session close
        self.SESSION_CLOSE_BUFFER_MINUTES = int(
            os.environ.get('SESSION_CLOSE_BUFFER_MINUTES', '5')
        )

        # Health Monitoring Settings
        self.HEALTH_CHECK_INTERVAL_SECONDS = int(
            os.environ.get('HEALTH_CHECK_INTERVAL_SECONDS', '5')
        )
        self.RECONNECT_MAX_ATTEMPTS = int(
            os.environ.get('RECONNECT_MAX_ATTEMPTS', '10')
        )
        self.RECONNECT_BASE_DELAY_SECONDS = float(
            os.environ.get('RECONNECT_BASE_DELAY_SECONDS', '1.0')
        )

        # Logging Settings
        self.LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
        self.LOG_FORMAT = os.environ.get(
            'LOG_FORMAT',
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def to_dict(self) -> dict:
        """Return configuration as dictionary for logging/debugging."""
        return {
            'IB_HOST': self.IB_HOST,
            'IB_PORT': self.IB_PORT,
            'IB_CLIENT_ID': self.IB_CLIENT_ID,
            'DASHBOARD_PORT': self.DASHBOARD_PORT,
            'DASHBOARD_HOST': self.DASHBOARD_HOST,
            'DASHBOARD_DEBUG': self.DASHBOARD_DEBUG,
            'DATA_STALENESS_THRESHOLD_SECONDS': self.DATA_STALENESS_THRESHOLD_SECONDS,
            'SESSION_CLOSE_BUFFER_MINUTES': self.SESSION_CLOSE_BUFFER_MINUTES,
            'HEALTH_CHECK_INTERVAL_SECONDS': self.HEALTH_CHECK_INTERVAL_SECONDS,
            'RECONNECT_MAX_ATTEMPTS': self.RECONNECT_MAX_ATTEMPTS,
            'RECONNECT_BASE_DELAY_SECONDS': self.RECONNECT_BASE_DELAY_SECONDS,
            'LOG_LEVEL': self.LOG_LEVEL,
            'LOG_FORMAT': self.LOG_FORMAT,
        }

    @classmethod
    def from_env(cls) -> 'Config':
        """Factory method to create Config from environment."""
        return cls()


# Singleton instance for global access
_config: Optional[Config] = None


def get_config() -> Config:
    """
    Get the global configuration instance.

    Creates a new Config instance if one doesn't exist.
    Configuration is loaded from environment variables.
    """
    global _config
    if _config is None:
        _config = Config()
    return _config


def reset_config() -> None:
    """Reset the global configuration (useful for testing)."""
    global _config
    _config = None
