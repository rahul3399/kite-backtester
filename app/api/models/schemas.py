# app/api/models/schemas.py
from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum

# Database schemas that correspond to SQLAlchemy models

class BacktestBase(BaseModel):
    """Base backtest schema"""
    strategy_name: str
    start_date: datetime
    end_date: datetime
    symbols: List[str]
    initial_capital: float
    parameters: Dict[str, Any]

class BacktestCreate(BacktestBase):
    """Schema for creating a backtest"""
    commission: float = 0.0002
    slippage: float = 0.0001

class BacktestUpdate(BaseModel):
    """Schema for updating a backtest"""
    status: Optional[str] = None
    progress: Optional[float] = None
    error: Optional[str] = None
    results: Optional[Dict[str, Any]] = None

class BacktestInDB(BacktestBase):
    """Schema for backtest in database"""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    backtest_id: str
    status: str
    progress: Optional[float]
    error: Optional[str]
    results: Optional[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime

class StrategyInstanceBase(BaseModel):
    """Base strategy instance schema"""
    strategy_name: str
    symbols: List[str]
    parameters: Dict[str, Any]
    is_paper_trading: bool = True

class StrategyInstanceCreate(StrategyInstanceBase):
    """Schema for creating a strategy instance"""
    capital_allocation: Optional[float] = None
    risk_per_trade: Optional[float] = 0.02
    max_positions: Optional[int] = 5

class StrategyInstanceUpdate(BaseModel):
    """Schema for updating a strategy instance"""
    is_active: Optional[bool] = None
    parameters: Optional[Dict[str, Any]] = None
    capital_allocation: Optional[float] = None
    risk_per_trade: Optional[float] = None
    max_positions: Optional[int] = None

class StrategyInstanceInDB(StrategyInstanceBase):
    """Schema for strategy instance in database"""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    strategy_id: str
    is_active: bool
    capital_allocation: Optional[float]
    risk_per_trade: float
    max_positions: int
    created_at: datetime
    updated_at: datetime
    last_signal_at: Optional[datetime]
    total_trades: int
    open_positions: int

class TradeBase(BaseModel):
    """Base trade schema"""
    symbol: str
    side: str
    quantity: int
    entry_price: float
    entry_time: datetime

class TradeCreate(TradeBase):
    """Schema for creating a trade"""
    strategy_id: Optional[str] = None
    order_type: str = "MARKET"
    commission: float = 0.0

class TradeUpdate(BaseModel):
    """Schema for updating a trade"""
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: Optional[float] = None
    commission: Optional[float] = None
    status: Optional[str] = None

class TradeInDB(TradeBase):
    """Schema for trade in database"""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    trade_id: str
    strategy_id: Optional[str]
    exit_price: Optional[float]
    exit_time: Optional[datetime]
    pnl: Optional[float]
    pnl_percentage: Optional[float]
    commission: float
    status: str
    created_at: datetime
    updated_at: datetime

class PositionBase(BaseModel):
    """Base position schema"""
    symbol: str
    quantity: int
    avg_price: float
    strategy_id: Optional[str] = None

class PositionCreate(PositionBase):
    """Schema for creating a position"""
    entry_time: datetime = Field(default_factory=datetime.utcnow)

class PositionUpdate(BaseModel):
    """Schema for updating a position"""
    quantity: Optional[int] = None
    avg_price: Optional[float] = None
    current_price: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    realized_pnl: Optional[float] = None

class PositionInDB(PositionBase):
    """Schema for position in database"""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    position_id: str
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_percentage: float
    realized_pnl: float
    entry_time: datetime
    updated_at: datetime

class OrderBase(BaseModel):
    """Base order schema"""
    symbol: str
    side: str
    quantity: int
    order_type: str
    price: Optional[float] = None
    trigger_price: Optional[float] = None

class OrderCreate(OrderBase):
    """Schema for creating an order"""
    strategy_id: Optional[str] = None

class OrderUpdate(BaseModel):
    """Schema for updating an order"""
    quantity: Optional[int] = None
    price: Optional[float] = None
    trigger_price: Optional[float] = None
    status: Optional[str] = None
    filled_quantity: Optional[int] = None
    avg_fill_price: Optional[float] = None

class OrderInDB(OrderBase):
    """Schema for order in database"""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    order_id: str
    strategy_id: Optional[str]
    status: str
    filled_quantity: int
    avg_fill_price: Optional[float]
    placed_time: datetime
    updated_time: datetime

class PerformanceRecordBase(BaseModel):
    """Base performance record schema"""
    strategy_id: Optional[str] = None
    date: datetime
    portfolio_value: float
    capital: float
    positions_value: float

class PerformanceRecordCreate(PerformanceRecordBase):
    """Schema for creating a performance record"""
    daily_pnl: float = 0.0
    trades_count: int = 0

class PerformanceRecordInDB(PerformanceRecordBase):
    """Schema for performance record in database"""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    daily_pnl: float
    cumulative_pnl: float
    trades_count: int
    win_rate: Optional[float]
    sharpe_ratio: Optional[float]
    created_at: datetime

# Pagination schemas
class PaginationParams(BaseModel):
    """Pagination parameters"""
    skip: int = Field(0, ge=0, description="Number of items to skip")
    limit: int = Field(100, ge=1, le=1000, description="Number of items to return")

class PaginatedResponse(BaseModel):
    """Generic paginated response"""
    items: List[Any]
    total: int
    skip: int
    limit: int
    has_more: bool

# Filter schemas
class TradeFilters(BaseModel):
    """Trade filter parameters"""
    strategy_id: Optional[str] = None
    symbol: Optional[str] = None
    side: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    min_pnl: Optional[float] = None
    max_pnl: Optional[float] = None

class BacktestFilters(BaseModel):
    """Backtest filter parameters"""
    strategy_name: Optional[str] = None
    status: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None