# app/config.py
from pydantic_settings import BaseSettings
from typing import Optional, List
from functools import lru_cache

class Settings(BaseSettings):
    # Kite API Configuration
    KITE_API_KEY: str
    KITE_API_SECRET: str
    KITE_ACCESS_TOKEN: Optional[str] = None
    
    # Database
    DATABASE_URL: str = "postgresql://user:pass@localhost/kite_trading"
    
    # Redis (for caching and real-time updates)
    REDIS_URL: str = "redis://localhost:6379"
    
    # Performance Settings
    MAX_CONCURRENT_STRATEGIES: int = 10
    WEBSOCKET_RECONNECT_INTERVAL: int = 5
    HISTORICAL_DATA_CACHE_SIZE: int = 1000000  # Number of candles
    
    # Backtesting
    DEFAULT_SLIPPAGE_PCT: float = 0.05
    DEFAULT_COMMISSION_PCT: float = 0.02
    
    # Paper Trading
    PAPER_TRADING_INITIAL_CAPITAL: float = 1000000.0
    MAX_POSITION_SIZE_PCT: float = 20.0  # Max % of capital per position
    
    # API Settings
    API_V1_PREFIX: str = "/api/v1"
    PROJECT_NAME: str = "Kite Trading System"
    VERSION: str = "1.0.0"
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()