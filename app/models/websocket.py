from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum

class TickMode(str, Enum):
    """Kite WebSocket tick modes"""
    LTP = "ltp"
    QUOTE = "quote"
    FULL = "full"

class TickData(BaseModel):
    """Model for tick data received from Kite WebSocket"""
    instrument_token: int
    mode: str
    tradable: bool
    last_price: float
    last_traded_quantity: Optional[int] = None
    average_traded_price: Optional[float] = None
    volume_traded: Optional[int] = None
    total_buy_quantity: Optional[int] = None
    total_sell_quantity: Optional[int] = None
    ohlc: Optional[Dict[str, float]] = None
    change: Optional[float] = None
    last_trade_time: Optional[datetime] = None
    oi: Optional[int] = None  # Open Interest
    oi_day_high: Optional[int] = None
    oi_day_low: Optional[int] = None
    exchange_timestamp: Optional[datetime] = None
    
    # Quote and Full mode fields
    depth: Optional[Dict[str, List[Dict[str, Any]]]] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class WebSocketMessage(BaseModel):
    """WebSocket message wrapper"""
    type: str
    data: Any
    timestamp: datetime = Field(default_factory=datetime.now)

class SubscriptionRequest(BaseModel):
    """Model for subscription requests"""
    instruments: List[int]
    mode: TickMode = TickMode.FULL

class ConnectionStatus(BaseModel):
    """WebSocket connection status"""
    is_connected: bool
    last_ping: Optional[datetime] = None
    reconnect_count: int = 0
    error_count: int = 0
    last_error: Optional[str] = None

class MarketDepth(BaseModel):
    """Market depth data structure"""
    price: float
    quantity: int
    orders: int

class DepthData(BaseModel):
    """Full depth data for an instrument"""
    buy: List[MarketDepth]
    sell: List[MarketDepth]

class WebSocketConfig(BaseModel):
    """Configuration for WebSocket connection"""
    api_key: str
    access_token: str
    debug: bool = False
    reconnect: bool = True
    max_reconnect_attempts: int = 50
    reconnect_interval: int = 5  # seconds
    ping_interval: int = 2.5  # seconds
    
class TickProcessor(BaseModel):
    """Configuration for tick processing"""
    buffer_size: int = 1000
    process_interval: float = 0.1  # seconds
    enable_logging: bool = True
    
class StreamingData(BaseModel):
    """Aggregated streaming data for instruments"""
    instrument_token: int
    symbol: str
    ltp: float
    change_percent: float
    volume: int
    oi: Optional[int] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_qty: Optional[int] = None
    ask_qty: Optional[int] = None
    last_update: datetime = Field(default_factory=datetime.now)
    
    @property
    def spread(self) -> Optional[float]:
        """Calculate bid-ask spread"""
        if self.bid and self.ask:
            return self.ask - self.bid
        return None
    
    @property
    def spread_percentage(self) -> Optional[float]:
        """Calculate bid-ask spread percentage"""
        if self.bid and self.ask and self.bid > 0:
            return ((self.ask - self.bid) / self.bid) * 100
        return None