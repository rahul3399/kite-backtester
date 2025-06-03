# app/api/models/requests.py
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum

class TimeFrame(str, Enum):
    """Valid timeframes for historical data"""
    MINUTE_1 = "minute"
    MINUTE_3 = "3minute"
    MINUTE_5 = "5minute"
    MINUTE_10 = "10minute"
    MINUTE_15 = "15minute"
    MINUTE_30 = "30minute"
    MINUTE_60 = "60minute"
    DAY = "day"

class BacktestRequest(BaseModel):
    """Request model for running a backtest"""
    strategy_name: str = Field(..., description="Name of the strategy to backtest")
    start_date: datetime = Field(..., description="Start date for backtesting")
    end_date: datetime = Field(..., description="End date for backtesting")
    symbols: List[str] = Field(..., min_items=1, description="List of symbols to trade")
    initial_capital: float = Field(1000000, gt=0, description="Initial capital for backtesting")
    commission: float = Field(0.0002, ge=0, le=0.01, description="Commission rate (0-1%)")
    slippage: float = Field(0.0001, ge=0, le=0.01, description="Slippage rate (0-1%)")
    timeframe: TimeFrame = Field(TimeFrame.MINUTE_5, description="Data timeframe")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Strategy-specific parameters")
    
    @validator('end_date')
    def end_date_after_start_date(cls, v, values):
        if 'start_date' in values and v <= values['start_date']:
            raise ValueError('end_date must be after start_date')
        return v

class OptimizationRequest(BaseModel):
    """Request model for strategy optimization"""
    strategy_name: str = Field(..., description="Name of the strategy to optimize")
    start_date: datetime = Field(..., description="Start date for optimization")
    end_date: datetime = Field(..., description="End date for optimization")
    symbols: List[str] = Field(..., min_items=1, description="List of symbols to trade")
    parameter_grid: Dict[str, List[Any]] = Field(..., description="Parameter grid for optimization")
    optimization_metric: str = Field("sharpe_ratio", description="Metric to optimize")
    initial_capital: float = Field(1000000, gt=0, description="Initial capital")
    commission: float = Field(0.0002, ge=0, le=0.01, description="Commission rate")
    slippage: float = Field(0.0001, ge=0, le=0.01, description="Slippage rate")
    max_iterations: Optional[int] = Field(None, gt=0, le=1000, description="Maximum optimization iterations")

class StartStrategyRequest(BaseModel):
    """Request model for starting a paper trading strategy"""
    strategy_name: str = Field(..., description="Name of the strategy to start")
    symbols: List[str] = Field(..., min_items=1, description="List of symbols to trade")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Strategy parameters")
    capital_allocation: Optional[float] = Field(None, gt=0, description="Capital to allocate to this strategy")
    risk_per_trade: Optional[float] = Field(0.02, gt=0, le=0.1, description="Risk per trade (0-10%)")
    max_positions: Optional[int] = Field(5, gt=0, le=50, description="Maximum concurrent positions")

class StopStrategyRequest(BaseModel):
    """Request model for stopping a strategy"""
    strategy_id: str = Field(..., description="ID of the strategy to stop")
    close_positions: bool = Field(True, description="Whether to close all positions")
    cancel_orders: bool = Field(True, description="Whether to cancel pending orders")

class RegisterStrategyRequest(BaseModel):
    """Request model for registering a custom strategy"""
    strategy_name: str = Field(..., description="Name for the strategy")
    code: str = Field(..., description="Python code for the strategy")
    description: Optional[str] = Field(None, description="Strategy description")
    required_indicators: Optional[List[str]] = Field(None, description="Required technical indicators")

class HistoricalDataRequest(BaseModel):
    """Request model for fetching historical data"""
    symbol: str = Field(..., description="Trading symbol")
    start_date: datetime = Field(..., description="Start date")
    end_date: datetime = Field(..., description="End date")
    timeframe: TimeFrame = Field(TimeFrame.MINUTE_5, description="Data timeframe")
    include_indicators: bool = Field(True, description="Include technical indicators")

class OrderRequest(BaseModel):
    """Request model for placing an order (paper trading)"""
    symbol: str = Field(..., description="Trading symbol")
    side: str = Field(..., pattern="^(BUY|SELL)$", description="Order side")
    quantity: int = Field(..., gt=0, description="Order quantity")
    order_type: str = Field("MARKET", pattern="^(MARKET|LIMIT|STOP_LOSS|STOP_LOSS_LIMIT)$")
    price: Optional[float] = Field(None, gt=0, description="Limit price")
    trigger_price: Optional[float] = Field(None, gt=0, description="Stop loss trigger price")
    strategy_id: Optional[str] = Field(None, description="Associated strategy ID")

class ModifyOrderRequest(BaseModel):
    """Request model for modifying an order"""
    order_id: str = Field(..., description="Order ID to modify")
    quantity: Optional[int] = Field(None, gt=0, description="New quantity")
    price: Optional[float] = Field(None, gt=0, description="New price")
    trigger_price: Optional[float] = Field(None, gt=0, description="New trigger price")

class ReportRequest(BaseModel):
    """Request model for generating reports"""
    report_type: str = Field(..., pattern="^(performance|trades|positions|equity_curve)$")
    start_date: Optional[datetime] = Field(None, description="Report start date")
    end_date: Optional[datetime] = Field(None, description="Report end date")
    strategy_id: Optional[str] = Field(None, description="Filter by strategy ID")
    symbols: Optional[List[str]] = Field(None, description="Filter by symbols")
    format: str = Field("json", pattern="^(json|csv|excel)$", description="Output format")

class WebSocketSubscribeRequest(BaseModel):
    """Request model for WebSocket subscriptions"""
    symbols: List[str] = Field(..., min_items=1, description="Symbols to subscribe")
    data_types: List[str] = Field(["tick"], description="Data types to receive")
    mode: str = Field("full", pattern="^(ltp|quote|full)$", description="Subscription mode")