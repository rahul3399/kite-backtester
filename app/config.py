from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    kite_api_key: str
    kite_api_secret: str
    kite_access_token: str
    log_level: str = "INFO"
    redis_url: Optional[str] = None
    
    # Trading parameters
    default_spread_threshold: float = 2.0
    default_stop_loss_multiplier: float = 2.0
    position_size: int = 75  # lot size for Nifty options
    
    class Config:
        env_file = ".env"

settings = Settings()