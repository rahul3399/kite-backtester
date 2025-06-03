# app/api/v1/backtest.py
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid
import logging

from ...api.models.requests import BacktestRequest, OptimizationRequest
from ...api.models.responses import (
    BacktestResponse, BacktestStatusResponse, BacktestResultResponse,
    TradeResponse, ErrorResponse, StatusEnum
)
from ...strategies.registry import strategy_registry
from ...strategies.base_strategy import StrategyConfig
from ...dependencies import get_db
from sqlalchemy.orm import Session

router = APIRouter()
logger = logging.getLogger(__name__)

# Store backtest results in memory (in production, use database)
backtest_results: Dict[str, Any] = {}

@router.post("/run", response_model=BacktestResponse)
async def run_backtest(
    request: BacktestRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Run a backtest for a strategy
    
    - **strategy_name**: Name of the strategy to backtest
    - **start_date**: Start date for the backtest period
    - **end_date**: End date for the backtest period
    - **symbols**: List of symbols to trade
    - **initial_capital**: Starting capital for the backtest
    - **parameters**: Strategy-specific parameters
    """
    
    # Validate strategy exists
    strategy_class = strategy_registry.get_strategy_class(request.strategy_name)
    if not strategy_class:
        raise HTTPException(
            status_code=404,
            detail=f"Strategy '{request.strategy_name}' not found"
        )
    
    # Create backtest ID
    backtest_id = str(uuid.uuid4())
    
    # Initialize result entry
    backtest_results[backtest_id] = {
        "status": StatusEnum.PENDING,
        "progress": 0.0,
        "timestamp": datetime.now()
    }
    
    # Run backtest in background
    background_tasks.add_task(
        _run_backtest_task,
        backtest_id,
        strategy_class,
        request
    )
    
    return BacktestResponse(
        backtest_id=backtest_id,
        status=StatusEnum.PENDING,
        message="Backtest queued for execution",
        estimated_time=60  # Rough estimate
    )

async def _run_backtest_task(backtest_id: str, strategy_class: type, request: BacktestRequest):
    """Background task to run backtest"""
    try:
        from ...main import backtest_engine
        
        # Update status
        backtest_results[backtest_id]["status"] = StatusEnum.RUNNING
        backtest_results[backtest_id]["start_time"] = datetime.now()
        
        # Create strategy config
        config = StrategyConfig(
            name=request.strategy_name,
            symbols=request.symbols,
            timeframe=request.timeframe,
            parameters=request.parameters
        )
        
        # Create strategy instance
        strategy = strategy_class(config)
        
        # Run backtest
        result = await backtest_engine.run_backtest(
            strategy=strategy,
            start_date=request.start_date,
            end_date=request.end_date,
            symbols=request.symbols,
            commission=request.commission,
            slippage=request.slippage
        )
        
        # Store result
        backtest_results[backtest_id] = {
            "status": StatusEnum.COMPLETED,
            "result": result,
            "timestamp": datetime.now(),
            "end_time": datetime.now()
        }
        
        logger.info(f"Backtest {backtest_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Backtest {backtest_id} failed: {str(e)}")
        backtest_results[backtest_id] = {
            "status": StatusEnum.FAILED,
            "error": str(e),
            "timestamp": datetime.now()
        }

@router.get("/status/{backtest_id}", response_model=BacktestStatusResponse)
async def get_backtest_status(backtest_id: str):
    """Get the status of a running backtest"""
    
    if backtest_id not in backtest_results:
        raise HTTPException(
            status_code=404,
            detail="Backtest not found"
        )
    
    result = backtest_results[backtest_id]
    
    # Calculate progress for running backtests
    progress = None
    if result["status"] == StatusEnum.RUNNING and "start_time" in result:
        # Simple time-based progress estimate
        elapsed = (datetime.now() - result["start_time"]).total_seconds()
        progress = min(elapsed / 60 * 100, 95)  # Cap at 95%
    
    return BacktestStatusResponse(
        backtest_id=backtest_id,
        status=result["status"],
        progress=progress,
        error=result.get("error"),
        timestamp=result["timestamp"],
        total_trades=result.get("result", {}).trades if "result" in result else None
    )

@router.get("/results/{backtest_id}", response_model=BacktestResultResponse)
async def get_backtest_results(backtest_id: str):
    """Get detailed results of a completed backtest"""
    
    if backtest_id not in backtest_results:
        raise HTTPException(
            status_code=404,
            detail="Backtest not found"
        )
    
    result = backtest_results[backtest_id]
    
    if result["status"] != StatusEnum.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Backtest is not completed. Current status: {result['status']}"
        )
    
    backtest_result = result["result"]
    
    # Calculate total return percentage
    total_return = ((backtest_result.final_capital - backtest_result.initial_capital) / 
                   backtest_result.initial_capital * 100)
    
    return BacktestResultResponse(
        backtest_id=backtest_id,
        strategy_name=backtest_result.strategy_name,
        symbols=backtest_result.symbols,
        start_date=backtest_result.start_date,
        end_date=backtest_result.end_date,
        initial_capital=backtest_result.initial_capital,
        final_capital=backtest_result.final_capital,
        total_return=total_return,
        metrics=backtest_result.metrics,
        summary={
            "total_trades": len(backtest_result.trades),
            "winning_trades": backtest_result.metrics.get("winning_trades", 0),
            "losing_trades": backtest_result.metrics.get("losing_trades", 0),
            "win_rate": backtest_result.metrics.get("win_rate", 0),
            "sharpe_ratio": backtest_result.metrics.get("sharpe_ratio", 0),
            "max_drawdown": backtest_result.metrics.get("max_drawdown", 0)
        }
    )

@router.get("/results/{backtest_id}/trades", response_model=List[TradeResponse])
async def get_backtest_trades(
    backtest_id: str,
    skip: int = Query(0, ge=0, description="Number of items to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Number of items to return")
):
    """Get trades from a backtest"""
    
    if backtest_id not in backtest_results:
        raise HTTPException(
            status_code=404,
            detail="Backtest not found"
        )
    
    result = backtest_results[backtest_id]
    
    if result["status"] != StatusEnum.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Backtest is not completed. Current status: {result['status']}"
        )
    
    trades = result["result"].trades
    
    # Apply pagination
    paginated_trades = trades[skip:skip + limit]
    
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
            status="closed"
        ))
    
    return trade_responses

@router.get("/results/{backtest_id}/equity-curve")
async def get_equity_curve(backtest_id: str):
    """Get equity curve data from a backtest"""
    
    if backtest_id not in backtest_results:
        raise HTTPException(
            status_code=404,
            detail="Backtest not found"
        )
    
    result = backtest_results[backtest_id]
    
    if result["status"] != StatusEnum.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Backtest is not completed. Current status: {result['status']}"
        )
    
    equity_curve = result["result"].equity_curve
    
    if equity_curve.empty:
        return {"data": []}
    
    # Convert DataFrame to list of dicts
    curve_data = equity_curve.reset_index().to_dict("records")
    
    return {"data": curve_data}

@router.post("/optimize", response_model=BacktestResponse)
async def optimize_strategy(
    request: OptimizationRequest,
    background_tasks: BackgroundTasks
):
    """
    Run strategy parameter optimization
    
    This will test multiple parameter combinations to find the optimal settings
    """
    
    # Validate strategy exists
    strategy_class = strategy_registry.get_strategy_class(request.strategy_name)
    if not strategy_class:
        raise HTTPException(
            status_code=404,
            detail=f"Strategy '{request.strategy_name}' not found"
        )
    
    # Validate parameter grid
    if not request.parameter_grid:
        raise HTTPException(
            status_code=400,
            detail="Parameter grid cannot be empty"
        )
    
    # Create optimization ID
    optimization_id = str(uuid.uuid4())
    
    # Initialize result entry
    backtest_results[optimization_id] = {
        "status": StatusEnum.PENDING,
        "progress": 0.0,
        "timestamp": datetime.now(),
        "type": "optimization"
    }
    
    # Run optimization in background
    background_tasks.add_task(
        _run_optimization_task,
        optimization_id,
        strategy_class,
        request
    )
    
    return BacktestResponse(
        backtest_id=optimization_id,
        status=StatusEnum.PENDING,
        message="Optimization queued for execution",
        estimated_time=300  # Optimization takes longer
    )

async def _run_optimization_task(optimization_id: str, strategy_class: type, request: OptimizationRequest):
    """Background task to run optimization"""
    try:
        from ...main import backtest_engine
        
        # Update status
        backtest_results[optimization_id]["status"] = StatusEnum.RUNNING
        backtest_results[optimization_id]["start_time"] = datetime.now()
        
        # Run optimization
        result = await backtest_engine.optimize_strategy(
            strategy_class=strategy_class,
            parameter_grid=request.parameter_grid,
            start_date=request.start_date,
            end_date=request.end_date,
            symbols=request.symbols,
            optimization_metric=request.optimization_metric
        )
        
        # Store result
        backtest_results[optimization_id] = {
            "status": StatusEnum.COMPLETED,
            "result": result,
            "timestamp": datetime.now(),
            "end_time": datetime.now(),
            "type": "optimization"
        }
        
        logger.info(f"Optimization {optimization_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Optimization {optimization_id} failed: {str(e)}")
        backtest_results[optimization_id] = {
            "status": StatusEnum.FAILED,
            "error": str(e),
            "timestamp": datetime.now(),
            "type": "optimization"
        }

@router.get("/list")
async def list_backtests(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    status: Optional[StatusEnum] = None
):
    """List all backtests with optional filtering"""
    
    # Filter backtests
    filtered_backtests = []
    for backtest_id, data in backtest_results.items():
        if status is None or data["status"] == status:
            filtered_backtests.append({
                "backtest_id": backtest_id,
                "status": data["status"],
                "timestamp": data["timestamp"],
                "type": data.get("type", "backtest")
            })
    
    # Sort by timestamp (newest first)
    filtered_backtests.sort(key=lambda x: x["timestamp"], reverse=True)
    
    # Apply pagination
    total = len(filtered_backtests)
    items = filtered_backtests[skip:skip + limit]
    
    return {
        "items": items,
        "total": total,
        "skip": skip,
        "limit": limit,
        "has_more": skip + limit < total
    }

@router.delete("/{backtest_id}")
async def delete_backtest(backtest_id: str):
    """Delete a backtest and its results"""
    
    if backtest_id not in backtest_results:
        raise HTTPException(
            status_code=404,
            detail="Backtest not found"
        )
    
    # Don't delete running backtests
    if backtest_results[backtest_id]["status"] == StatusEnum.RUNNING:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete a running backtest"
        )
    
    del backtest_results[backtest_id]
    
    return {"message": f"Backtest {backtest_id} deleted successfully"}