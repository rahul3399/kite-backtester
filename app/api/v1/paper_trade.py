# app/api/v1/paper_trade.py
from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from ...api.models.requests import (
    StartStrategyRequest, StopStrategyRequest, OrderRequest, ModifyOrderRequest
)
from ...api.models.responses import (
    StrategyResponse, PositionResponse, OrderResponse, 
    PerformanceResponse, StatusEnum
)
from ...strategies.registry import strategy_registry
from ...strategies.base_strategy import StrategyConfig
from ...dependencies import get_db
from sqlalchemy.orm import Session

router = APIRouter()
logger = logging.getLogger(__name__)

# Active strategies tracking
active_strategies: Dict[str, Dict[str, Any]] = {}

@router.post("/start", response_model=StrategyResponse)
async def start_strategy(
    request: StartStrategyRequest,
    db: Session = Depends(get_db)
):
    """
    Start a paper trading strategy
    
    - **strategy_name**: Name of the strategy to start
    - **symbols**: List of symbols to trade
    - **parameters**: Strategy-specific parameters
    - **capital_allocation**: Capital to allocate (optional)
    """
    
    # Validate strategy exists
    strategy_class = strategy_registry.get_strategy_class(request.strategy_name)
    if not strategy_class:
        raise HTTPException(
            status_code=404,
            detail=f"Strategy '{request.strategy_name}' not found"
        )
    
    from ...main import paper_engine
    
    # Check if paper engine is running
    if not paper_engine.is_running:
        raise HTTPException(
            status_code=503,
            detail="Paper trading engine is not running"
        )
    
    # Create strategy config
    config = StrategyConfig(
        name=request.strategy_name,
        symbols=request.symbols,
        parameters=request.parameters
    )
    
    # Create strategy instance
    strategy = strategy_class(config)
    
    # Set capital allocation if provided
    if request.capital_allocation:
        strategy.capital_allocation = request.capital_allocation
    
    # Add to paper trading engine
    strategy_id = await paper_engine.add_strategy(strategy)
    
    # Store in active strategies
    active_strategies[strategy_id] = {
        "name": request.strategy_name,
        "symbols": request.symbols,
        "parameters": request.parameters,
        "status": StatusEnum.RUNNING,
        "started_at": datetime.now(),
        "capital_allocation": request.capital_allocation,
        "risk_per_trade": request.risk_per_trade,
        "max_positions": request.max_positions
    }
    
    logger.info(f"Started strategy {request.strategy_name} with ID {strategy_id}")
    
    return StrategyResponse(
        strategy_id=strategy_id,
        status=StatusEnum.RUNNING,
        message=f"Strategy {request.strategy_name} started successfully",
        strategy_name=request.strategy_name,
        symbols=request.symbols
    )

@router.post("/stop/{strategy_id}", response_model=StrategyResponse)
async def stop_strategy(
    strategy_id: str,
    request: Optional[StopStrategyRequest] = None
):
    """Stop a paper trading strategy"""
    
    if strategy_id not in active_strategies:
        raise HTTPException(
            status_code=404,
            detail="Strategy not found"
        )
    
    from ...main import paper_engine
    
    # Get strategy info
    strategy_info = active_strategies[strategy_id]
    
    # Stop the strategy
    await paper_engine.stop_strategy(strategy_id)
    
    # Update status
    active_strategies[strategy_id]["status"] = StatusEnum.CANCELLED
    active_strategies[strategy_id]["stopped_at"] = datetime.now()
    
    # Handle position closing if requested
    if request and request.close_positions:
        # Close all positions for this strategy
        positions = paper_engine.virtual_broker.get_positions()
        for symbol, position in positions.items():
            if position.get("strategy_id") == strategy_id:
                # Create sell order to close position
                order = {
                    "symbol": symbol,
                    "side": "SELL",
                    "quantity": position["quantity"],
                    "order_type": "MARKET",
                    "strategy_id": strategy_id,
                    "timestamp": datetime.now()
                }
                paper_engine.virtual_broker.execute_order(
                    order, 
                    paper_engine.virtual_broker.market_prices.get(symbol, position["avg_price"])
                )
    
    logger.info(f"Stopped strategy {strategy_id}")
    
    return StrategyResponse(
        strategy_id=strategy_id,
        status=StatusEnum.CANCELLED,
        message=f"Strategy stopped successfully",
        strategy_name=strategy_info["name"]
    )

@router.get("/strategies", response_model=List[Dict[str, Any]])
async def list_active_strategies(
    include_stopped: bool = Query(False, description="Include stopped strategies")
):
    """List all paper trading strategies"""
    
    strategies = []
    for strategy_id, info in active_strategies.items():
        if include_stopped or info["status"] == StatusEnum.RUNNING:
            strategies.append({
                "strategy_id": strategy_id,
                "name": info["name"],
                "symbols": info["symbols"],
                "status": info["status"],
                "started_at": info["started_at"],
                "stopped_at": info.get("stopped_at"),
                "capital_allocation": info.get("capital_allocation"),
                "parameters": info["parameters"]
            })
    
    # Sort by start time (newest first)
    strategies.sort(key=lambda x: x["started_at"], reverse=True)
    
    return strategies

@router.get("/positions", response_model=List[PositionResponse])
async def get_positions(
    strategy_id: Optional[str] = Query(None, description="Filter by strategy ID")
):
    """Get current positions"""
    
    from ...main import paper_engine
    
    positions = paper_engine.virtual_broker.get_positions()
    position_list = []
    
    for symbol, position in positions.items():
        # Apply strategy filter if provided
        if strategy_id and position.get("strategy_id") != strategy_id:
            continue
        
        # Get current market price
        current_price = paper_engine.virtual_broker.market_prices.get(
            symbol, position["avg_price"]
        )
        market_value = current_price * position["quantity"]
        unrealized_pnl = (current_price - position["avg_price"]) * position["quantity"]
        unrealized_pnl_pct = (unrealized_pnl / (position["avg_price"] * position["quantity"])) * 100
        
        position_list.append(PositionResponse(
            symbol=symbol,
            quantity=position["quantity"],
            avg_price=position["avg_price"],
            current_price=current_price,
            market_value=market_value,
            unrealized_pnl=unrealized_pnl,
            unrealized_pnl_percentage=unrealized_pnl_pct,
            realized_pnl=position.get("realized_pnl", 0),
            entry_time=position.get("entry_time", datetime.now()),
            strategy_id=position.get("strategy_id")
        ))
    
    return position_list

@router.get("/performance", response_model=PerformanceResponse)
async def get_performance(
    strategy_id: Optional[str] = Query(None, description="Filter by strategy ID")
):
    """Get performance metrics"""
    
    from ...main import paper_engine
    
    # Get performance summary
    summary = paper_engine.virtual_broker.get_performance_summary()
    
    # Calculate detailed metrics
    trades = paper_engine.virtual_broker.trades
    
    # Filter trades by strategy if needed
    if strategy_id:
        trades = [t for t in trades if t.get("strategy_id") == strategy_id]
    
    # Calculate metrics
    from ...reporting.metrics_calculator import MetricsCalculator
    calculator = MetricsCalculator()
    
    # Create simple equity curve for metrics calculation
    equity_data = []
    current_value = summary["initial_capital"]
    
    for trade in trades:
        if trade.get("pnl"):
            current_value += trade["pnl"]
            equity_data.append({
                "timestamp": trade["timestamp"],
                "portfolio_value": current_value
            })
    
    if equity_data:
        equity_df = pd.DataFrame(equity_data)
        equity_df.set_index("timestamp", inplace=True)
    else:
        equity_df = pd.DataFrame()
    
    metrics = calculator.calculate_metrics(
        trades, equity_df, summary["initial_capital"]
    )
    
    # Build performance response
    return PerformanceResponse(
        initial_capital=summary["initial_capital"],
        current_capital=summary["current_capital"],
        total_value=summary["portfolio_value"],
        total_pnl=summary["total_pnl"],
        total_pnl_percentage=(summary["total_pnl"] / summary["initial_capital"]) * 100,
        metrics=PerformanceMetrics(**metrics),
        positions_count=summary["positions_count"],
        active_strategies=len([s for s in active_strategies.values() if s["status"] == StatusEnum.RUNNING])
    )

@router.get("/trades", response_model=List[TradeResponse])
async def get_trades(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    strategy_id: Optional[str] = Query(None, description="Filter by strategy ID"),
    symbol: Optional[str] = Query(None, description="Filter by symbol")
):
    """Get recent trades"""
    
    from ...main import paper_engine
    
    trades = paper_engine.virtual_broker.trades
    
    # Apply filters
    filtered_trades = trades
    if strategy_id:
        filtered_trades = [t for t in filtered_trades if t.get("strategy_id") == strategy_id]
    if symbol:
        filtered_trades = [t for t in filtered_trades if t["symbol"] == symbol]
    
    # Sort by timestamp (newest first)
    filtered_trades.sort(key=lambda x: x["timestamp"], reverse=True)
    
    # Apply pagination
    paginated_trades = filtered_trades[skip:skip + limit]
    
    # Convert to response format
    trade_responses = []
    for trade in paginated_trades:
        trade_responses.append(TradeResponse(
            trade_id=trade.get("order_id", ""),
            symbol=trade["symbol"],
            side=trade["side"],
            quantity=trade["quantity"],
            entry_price=trade["price"],
            exit_price=trade.get("exit_price"),
            entry_time=trade["timestamp"],
            exit_time=trade.get("exit_time"),
            pnl=trade.get("pnl"),
            pnl_percentage=trade.get("pnl_pct"),
            commission=trade.get("commission", 0),
            status="filled"
        ))
    
    return trade_responses

@router.post("/order", response_model=OrderResponse)
async def place_order(request: OrderRequest):
    """Place a manual order in paper trading"""
    
    from ...main import paper_engine
    
    # Validate order
    if request.order_type == "LIMIT" and request.price is None:
        raise HTTPException(
            status_code=400,
            detail="Limit orders require a price"
        )
    
    # Get current market price
    current_price = paper_engine.virtual_broker.market_prices.get(request.symbol)
    if not current_price:
        raise HTTPException(
            status_code=400,
            detail=f"No market data available for {request.symbol}"
        )
    
    # Create order
    order = {
        "symbol": request.symbol,
        "side": request.side,
        "quantity": request.quantity,
        "order_type": request.order_type,
        "price": request.price,
        "strategy_id": request.strategy_id,
        "timestamp": datetime.now()
    }
    
    # Execute order
    fill = paper_engine.virtual_broker.execute_order(order, current_price)
    
    if not fill:
        raise HTTPException(
            status_code=400,
            detail="Order execution failed. Check capital and position limits."
        )
    
    return OrderResponse(
        order_id=fill["order_id"],
        symbol=fill["symbol"],
        side=fill["side"],
        quantity=fill["quantity"],
        order_type=order["order_type"],
        price=fill["price"],
        status="filled",
        filled_quantity=fill["quantity"],
        avg_fill_price=fill["price"],
        placed_time=fill["timestamp"],
        updated_time=fill["timestamp"],
        strategy_id=fill.get("strategy_id")
    )

@router.get("/orders", response_model=List[OrderResponse])
async def get_orders(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    strategy_id: Optional[str] = Query(None, description="Filter by strategy ID")
):
    """Get order history"""
    
    from ...main import paper_engine
    
    orders = paper_engine.virtual_broker.orders
    
    # Apply filters
    if strategy_id:
        orders = [o for o in orders if o.get("strategy_id") == strategy_id]
    
    # Sort by timestamp (newest first)
    orders.sort(key=lambda x: x["timestamp"], reverse=True)
    
    # Apply pagination
    paginated_orders = orders[skip:skip + limit]
    
    # Convert to response format
    order_responses = []
    for order in paginated_orders:
        # Find corresponding fill
        fill = next((t for t in paper_engine.virtual_broker.trades 
                    if t.get("order_id") == order.get("order_id")), None)
        
        order_responses.append(OrderResponse(
            order_id=order.get("order_id", str(uuid.uuid4())),
            symbol=order["symbol"],
            side=order["side"],
            quantity=order["quantity"],
            order_type=order["order_type"],
            price=order.get("price"),
            status="filled" if fill else "pending",
            filled_quantity=fill["quantity"] if fill else 0,
            avg_fill_price=fill["price"] if fill else None,
            placed_time=order["timestamp"],
            updated_time=fill["timestamp"] if fill else order["timestamp"],
            strategy_id=order.get("strategy_id")
        ))
    
    return order_responses

@router.post("/reset")
async def reset_paper_trading():
    """Reset paper trading account to initial state"""
    
    from ...main import paper_engine
    
    # Stop all active strategies
    for strategy_id in list(active_strategies.keys()):
        if active_strategies[strategy_id]["status"] == StatusEnum.RUNNING:
            await paper_engine.stop_strategy(strategy_id)
    
    # Reset virtual broker
    settings = get_settings()
    paper_engine.virtual_broker.reset(settings.PAPER_TRADING_INITIAL_CAPITAL)
    
    # Clear active strategies
    active_strategies.clear()
    
    return {"message": "Paper trading account reset successfully"}