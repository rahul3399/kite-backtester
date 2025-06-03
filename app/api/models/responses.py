# app/api/models/responses.py
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum

class StatusEnum(str, Enum):
    """Common status values"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class BaseResponse(BaseModel):
    """Base response model"""
    success: bool = Field(..., description="Whether the request was successful")
    message: str = Field(..., description="Response message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")

class BacktestResponse(BaseModel):
    """Response model for backtest initiation"""
    backtest_id: str = Field(..., description="Unique backtest identifier")
    status: StatusEnum = Field(..., description="Current status")
    message: str = Field(..., description="Status message")
    estimated_time: Optional[int] = Field(None, description="Estimated completion time in seconds")

class BacktestStatusResponse(BaseModel):
    """Response model for backtest status"""
    backtest_id: str
    status: StatusEnum
    progress: Optional[float] = Field(None, ge=0, le=100, description="Progress percentage")
    current_date: Optional[datetime] = None
    total_trades: Optional[int] = None
    error: Optional[str] = None
    timestamp: datetime

class BacktestResultResponse(BaseModel):
    """Response model for backtest results"""
    backtest_id: str
    strategy_name: str
    symbols: List[str]
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float = Field(..., description="Total return percentage")
    metrics: Dict[str, float] = Field(..., description="Performance metrics")
    summary: Dict[str, Any] = Field(..., description="Summary statistics")

class TradeResponse(BaseModel):
    """Response model for trade information"""
    trade_id: str
    symbol: str
    side: str
    quantity: int
    entry_price: float
    exit_price: Optional[float] = None
    entry_time: datetime
    exit_time: Optional[datetime] = None
    pnl: Optional[float] = None
    pnl_percentage: Optional[float] = None
    commission: float
    status: str

class PositionResponse(BaseModel):
    """Response model for position information"""
    symbol: str
    quantity: int
    avg_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_percentage: float
    realized_pnl: float
    entry_time: datetime
    strategy_id: Optional[str] = None

class StrategyResponse(BaseModel):
    """Response model for strategy operations"""
    strategy_id: str
    status: StatusEnum
    message: str
    strategy_name: Optional[str] = None
    symbols: Optional[List[str]] = None

class StrategyInfoResponse(BaseModel):
    """Response model for strategy information"""
    name: str
    class_name: str
    module: str
    description: Optional[str]
    parameters: Dict[str, Any]
    required_indicators: List[str]
    is_active: bool = False
    instances: int = 0

class StrategyListResponse(BaseModel):
    """Response model for strategy list"""
    strategies: List[StrategyInfoResponse]
    total: int

class OrderResponse(BaseModel):
    """Response model for order information"""
    order_id: str
    symbol: str
    side: str
    quantity: int
    order_type: str
    price: Optional[float]
    trigger_price: Optional[float]
    status: str
    filled_quantity: int = 0
    avg_fill_price: Optional[float] = None
    placed_time: datetime
    updated_time: datetime
    strategy_id: Optional[str] = None

class PerformanceMetrics(BaseModel):
    """Performance metrics model"""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    expectancy: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    avg_holding_period: float

class PerformanceResponse(BaseModel):
    """Response model for performance data"""
    initial_capital: float
    current_capital: float
    total_value: float
    total_pnl: float
    total_pnl_percentage: float
    metrics: PerformanceMetrics
    daily_returns: Optional[List[Dict[str, Any]]] = None
    monthly_returns: Optional[Dict[str, float]] = None
    positions_count: int
    active_strategies: int

class EquityCurvePoint(BaseModel):
    """Single point in equity curve"""
    timestamp: datetime
    portfolio_value: float
    capital: float
    positions_value: float
    daily_pnl: Optional[float] = None
    cumulative_pnl: float

class EquityCurveResponse(BaseModel):
    """Response model for equity curve data"""
    data: List[EquityCurvePoint]
    start_date: datetime
    end_date: datetime
    initial_value: float
    final_value: float
    max_value: float
    min_value: float

class ErrorResponse(BaseModel):
    """Response model for errors"""
    error: str = Field(..., description="Error message")
    detail: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    status_code: int = Field(..., description="HTTP status code")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class HealthCheckResponse(BaseModel):
    """Response model for health check"""
    status: str
    version: str
    components: Dict[str, bool]
    timestamp: datetime

class WebSocketMessage(BaseModel):
    """WebSocket message format"""
    type: str = Field(..., description="Message type")
    data: Any = Field(..., description="Message data")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class MarketDataTick(BaseModel):
    """Market data tick model"""
    symbol: str
    last_price: float
    last_quantity: int
    buy_quantity: int
    sell_quantity: int
    volume: int
    bid_price: float
    ask_price: float
    open: float
    high: float
    low: float
    close: float
    timestamp: datetime