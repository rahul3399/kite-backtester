from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum

class TradeAction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    CLOSE = "CLOSE"
    NONE = "NONE"

class OptionInstrument(BaseModel):
    instrument_token: int
    trading_symbol: str
    strike_price: float
    option_type: str  # CE or PE
    expiry: datetime

class OHLCData(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    oi: Optional[int] = None  # Open Interest

class SpreadData(BaseModel):
    timestamp: datetime
    instrument1: str
    instrument2: str
    price1: float
    price2: float
    spread: float
    spread_percentage: float
    ma_20: Optional[float] = None
    ma_50: Optional[float] = None
    z_score: Optional[float] = None

class TradeSignal(BaseModel):
    timestamp: datetime
    action: TradeAction
    instrument1: str
    instrument2: str
    spread: float
    entry_reason: str
    
class Trade(BaseModel):
    trade_id: str
    entry_time: datetime
    exit_time: Optional[datetime] = None
    instrument1: str
    instrument2: str
    entry_spread: float
    exit_spread: Optional[float] = None
    pnl: Optional[float] = None
    status: str = "OPEN"  # OPEN, CLOSED
    trade_type: str  # LONG_SPREAD, SHORT_SPREAD

class StrategyStatus(BaseModel):
    is_active: bool
    current_pair: Optional[Dict[str, str]] = None
    current_spread: Optional[float] = None
    open_trades: List[Trade] = []
    total_pnl: float = 0.0
    trades_count: int = 0