# app/api/v1/reports.py
from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import StreamingResponse, FileResponse
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import json
import io
import logging
from sqlalchemy.orm import Session

from ...api.models.requests import ReportRequest
from ...api.models.responses import EquityCurveResponse, EquityCurvePoint
from ...dependencies import get_db
from ...database import crud

router = APIRouter()
logger = logging.getLogger(__name__)

# In-memory storage for backtest results (should be replaced with database)
_backtest_results: Dict[str, Any] = {}
_active_strategies: Dict[str, Dict[str, Any]] = {}

def set_backtest_results(results: Dict[str, Any]):
    """Set backtest results (called from backtest module)"""
    global _backtest_results
    _backtest_results = results

def set_active_strategies(strategies: Dict[str, Dict[str, Any]]):
    """Set active strategies (called from paper_trade module)"""
    global _active_strategies
    _active_strategies = strategies

@router.get("/backtest/{backtest_id}/summary")
async def get_backtest_summary(backtest_id: str, db: Session = Depends(get_db)):
    """
    Get comprehensive backtest report
    
    Returns detailed analysis including:
    - Performance metrics
    - Monthly returns
    - Drawdown analysis
    - Trade statistics
    """
    
    # Try to get from database first
    backtest = crud.get_backtest(db, backtest_id)
    if backtest and backtest.status == "completed":
        return _generate_backtest_summary_from_db(backtest)
    
    # Fallback to in-memory results
    if backtest_id not in _backtest_results:
        raise HTTPException(
            status_code=404,
            detail="Backtest not found"
        )
    
    result = _backtest_results[backtest_id]
    if result["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Backtest is not completed. Current status: {result['status']}"
        )
    
    backtest_result = result["result"]
    
    # Generate comprehensive report
    report = {
        "summary": {
            "strategy": backtest_result.strategy_name if hasattr(backtest_result, 'strategy_name') else 'Unknown',
            "period": {
                "start": backtest_result.start_date.isoformat() if hasattr(backtest_result, 'start_date') else None,
                "end": backtest_result.end_date.isoformat() if hasattr(backtest_result, 'end_date') else None,
                "days": (backtest_result.end_date - backtest_result.start_date).days if hasattr(backtest_result, 'start_date') and hasattr(backtest_result, 'end_date') else 0
            },
            "symbols": backtest_result.symbols if hasattr(backtest_result, 'symbols') else [],
            "initial_capital": backtest_result.initial_capital if hasattr(backtest_result, 'initial_capital') else 0,
            "final_capital": backtest_result.final_capital if hasattr(backtest_result, 'final_capital') else 0,
            "total_return": ((backtest_result.final_capital - backtest_result.initial_capital) / 
                           backtest_result.initial_capital * 100) if hasattr(backtest_result, 'initial_capital') and hasattr(backtest_result, 'final_capital') and backtest_result.initial_capital > 0 else 0,
            "total_trades": len(backtest_result.trades) if hasattr(backtest_result, 'trades') else 0,
            "avg_trades_per_day": len(backtest_result.trades) / max(1, (backtest_result.end_date - backtest_result.start_date).days) if hasattr(backtest_result, 'trades') and hasattr(backtest_result, 'start_date') and hasattr(backtest_result, 'end_date') else 0
        },
        "metrics": backtest_result.metrics if hasattr(backtest_result, 'metrics') else {},
        "monthly_returns": _calculate_monthly_returns(backtest_result.equity_curve) if hasattr(backtest_result, 'equity_curve') else {},
        "yearly_returns": _calculate_yearly_returns(backtest_result.equity_curve) if hasattr(backtest_result, 'equity_curve') else {},
        "drawdown_analysis": _analyze_drawdowns(backtest_result.equity_curve) if hasattr(backtest_result, 'equity_curve') else {},
        "trade_analysis": _analyze_trades(backtest_result.trades) if hasattr(backtest_result, 'trades') else {},
        "symbol_performance": _analyze_symbol_performance(backtest_result.trades) if hasattr(backtest_result, 'trades') else {},
        "time_analysis": _analyze_time_patterns(backtest_result.trades) if hasattr(backtest_result, 'trades') else {}
    }
    
    return report

def _generate_backtest_summary_from_db(backtest) -> Dict[str, Any]:
    """Generate summary from database backtest record"""
    
    return {
        "summary": {
            "strategy": backtest.strategy.name if backtest.strategy else 'Unknown',
            "period": {
                "start": backtest.start_date.isoformat(),
                "end": backtest.end_date.isoformat(),
                "days": (backtest.end_date - backtest.start_date).days
            },
            "symbols": backtest.symbols,
            "initial_capital": backtest.initial_capital,
            "final_capital": backtest.final_capital or 0,
            "total_return": backtest.total_return or 0,
            "total_trades": backtest.total_trades or 0
        },
        "metrics": backtest.metrics or {},
        "execution_time": backtest.execution_time,
        "bars_processed": backtest.bars_processed
    }

@router.get("/backtest/{backtest_id}/export")
async def export_backtest_results(
    backtest_id: str,
    format: str = Query("csv", pattern="^(csv|json|excel)$", description="Export format"),
    db: Session = Depends(get_db)
):
    """Export backtest results in various formats"""
    
    # Try database first
    backtest = crud.get_backtest(db, backtest_id)
    if backtest and backtest.status == "completed":
        return _export_backtest_from_db(backtest, format)
    
    # Fallback to in-memory
    if backtest_id not in _backtest_results:
        raise HTTPException(
            status_code=404,
            detail="Backtest not found"
        )
    
    result = _backtest_results[backtest_id]
    if result["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Backtest is not completed. Current status: {result['status']}"
        )
    
    backtest_result = result["result"]
    
    if format == "csv":
        # Export trades as CSV
        trades_data = backtest_result.trades if hasattr(backtest_result, 'trades') else []
        df = pd.DataFrame(trades_data)
        
        # Add additional columns
        if not df.empty:
            df['strategy'] = backtest_result.strategy_name if hasattr(backtest_result, 'strategy_name') else 'Unknown'
            df['backtest_id'] = backtest_id
            
        stream = io.StringIO()
        df.to_csv(stream, index=False)
        stream.seek(0)
        
        return StreamingResponse(
            io.BytesIO(stream.getvalue().encode()),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=backtest_{backtest_id}.csv"}
        )
    
    elif format == "json":
        # Export as JSON with full details
        export_data = {
            "backtest_id": backtest_id,
            "summary": {
                "strategy": backtest_result.strategy_name if hasattr(backtest_result, 'strategy_name') else 'Unknown',
                "start_date": backtest_result.start_date.isoformat() if hasattr(backtest_result, 'start_date') else None,
                "end_date": backtest_result.end_date.isoformat() if hasattr(backtest_result, 'end_date') else None,
                "initial_capital": backtest_result.initial_capital if hasattr(backtest_result, 'initial_capital') else 0,
                "final_capital": backtest_result.final_capital if hasattr(backtest_result, 'final_capital') else 0,
                "total_return": ((backtest_result.final_capital - backtest_result.initial_capital) / 
                               backtest_result.initial_capital * 100) if hasattr(backtest_result, 'initial_capital') and hasattr(backtest_result, 'final_capital') and backtest_result.initial_capital > 0 else 0
            },
            "metrics": backtest_result.metrics if hasattr(backtest_result, 'metrics') else {},
            "trades": backtest_result.trades if hasattr(backtest_result, 'trades') else [],
            "equity_curve": backtest_result.equity_curve.to_dict() if hasattr(backtest_result, 'equity_curve') and not backtest_result.equity_curve.empty else {},
            "monthly_returns": _calculate_monthly_returns(backtest_result.equity_curve) if hasattr(backtest_result, 'equity_curve') else {}
        }
        
        return StreamingResponse(
            io.BytesIO(json.dumps(export_data, indent=2, default=str).encode()),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=backtest_{backtest_id}.json"}
        )
    
    elif format == "excel":
        # Export as Excel with multiple sheets
        output = io.BytesIO()
        
        try:
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                # Summary sheet
                summary_df = pd.DataFrame([{
                    'Strategy': backtest_result.strategy_name if hasattr(backtest_result, 'strategy_name') else 'Unknown',
                    'Start Date': backtest_result.start_date if hasattr(backtest_result, 'start_date') else None,
                    'End Date': backtest_result.end_date if hasattr(backtest_result, 'end_date') else None,
                    'Initial Capital': backtest_result.initial_capital if hasattr(backtest_result, 'initial_capital') else 0,
                    'Final Capital': backtest_result.final_capital if hasattr(backtest_result, 'final_capital') else 0,
                    'Total Return %': ((backtest_result.final_capital - backtest_result.initial_capital) / 
                                     backtest_result.initial_capital * 100) if hasattr(backtest_result, 'initial_capital') and hasattr(backtest_result, 'final_capital') and backtest_result.initial_capital > 0 else 0,
                    'Total Trades': len(backtest_result.trades) if hasattr(backtest_result, 'trades') else 0
                }])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Metrics sheet
                if hasattr(backtest_result, 'metrics') and backtest_result.metrics:
                    metrics_df = pd.DataFrame([backtest_result.metrics])
                    metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
                
                # Trades sheet
                if hasattr(backtest_result, 'trades') and backtest_result.trades:
                    trades_df = pd.DataFrame(backtest_result.trades)
                    trades_df.to_excel(writer, sheet_name='Trades', index=False)
                
                # Equity curve sheet
                if hasattr(backtest_result, 'equity_curve') and not backtest_result.equity_curve.empty:
                    equity_df = backtest_result.equity_curve.reset_index()
                    equity_df.to_excel(writer, sheet_name='Equity Curve', index=False)
        except Exception as e:
            logger.error(f"Error creating Excel file: {e}")
            raise HTTPException(status_code=500, detail="Error generating Excel report")
        
        output.seek(0)
        
        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename=backtest_{backtest_id}.xlsx"}
        )

def _export_backtest_from_db(backtest, format: str):
    """Export backtest from database record"""
    
    if format == "json":
        export_data = {
            "backtest_id": backtest.backtest_id,
            "summary": {
                "strategy": backtest.strategy.name if backtest.strategy else 'Unknown',
                "start_date": backtest.start_date.isoformat(),
                "end_date": backtest.end_date.isoformat(),
                "initial_capital": backtest.initial_capital,
                "final_capital": backtest.final_capital or 0,
                "total_return": backtest.total_return or 0
            },
            "metrics": backtest.metrics or {},
            "parameters": backtest.parameters or {}
        }
        
        return StreamingResponse(
            io.BytesIO(json.dumps(export_data, indent=2, default=str).encode()),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=backtest_{backtest.backtest_id}.json"}
        )
    
    # For CSV and Excel, would need to fetch related trade data
    raise HTTPException(status_code=501, detail=f"Export format {format} not implemented for database records")

@router.get("/paper-trading/performance")
async def get_paper_trading_performance_report(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    strategy_id: Optional[str] = None
):
    """Get paper trading performance report"""
    
    # This would need to be implemented with proper paper trading engine integration
    # For now, return a placeholder
    try:
        # Get paper trading data from main application
        from ...main import paper_engine
        
        # Get all trades
        trades = paper_engine.virtual_broker.trades if paper_engine else []
        
        # Apply filters
        if strategy_id:
            trades = [t for t in trades if t.get("strategy_id") == strategy_id]
        
        if start_date:
            trades = [t for t in trades if t["timestamp"] >= start_date]
        
        if end_date:
            trades = [t for t in trades if t["timestamp"] <= end_date]
        
        # Get current positions
        positions = paper_engine.virtual_broker.get_positions() if paper_engine else {}
        
        # Calculate metrics
        from ...reporting.metrics_calculator import MetricsCalculator
        calculator = MetricsCalculator()
        
        # Create equity curve from trades
        equity_data = []
        current_value = paper_engine.virtual_broker.initial_capital if paper_engine else 1000000
        
        for trade in sorted(trades, key=lambda x: x["timestamp"]):
            if trade.get("pnl"):
                current_value += trade["pnl"] - trade.get("commission", 0)
                equity_data.append({
                    "timestamp": trade["timestamp"],
                    "portfolio_value": current_value,
                    "capital": current_value - sum(p.get("market_value", 0) for p in positions.values())
                })
        
        equity_df = pd.DataFrame(equity_data) if equity_data else pd.DataFrame()
        if not equity_df.empty:
            equity_df.set_index("timestamp", inplace=True)
        
        metrics = calculator.calculate_metrics(
            trades, equity_df, paper_engine.virtual_broker.initial_capital if paper_engine else 1000000
        )
        
        # Build report
        report = {
            "period": {
                "start": start_date.isoformat() if start_date else None,
                "end": end_date.isoformat() if end_date else None
            },
            "summary": paper_engine.virtual_broker.get_performance_summary() if paper_engine else {},
            "metrics": metrics,
            "positions": [
                {
                    "symbol": symbol,
                    "quantity": pos["quantity"],
                    "avg_price": pos["avg_price"],
                    "current_value": pos["quantity"] * (paper_engine.virtual_broker.market_prices.get(symbol, pos["avg_price"]) if paper_engine else pos["avg_price"]),
                    "unrealized_pnl": pos.get("unrealized_pnl", 0)
                }
                for symbol, pos in positions.items()
            ],
            "active_strategies": len(_active_strategies),
            "trade_analysis": _analyze_trades(trades) if trades else {}
        }
        
        return report
        
    except ImportError:
        # Fallback if paper_engine not available
        return {
            "error": "Paper trading engine not available",
            "period": {
                "start": start_date.isoformat() if start_date else None,
                "end": end_date.isoformat() if end_date else None
            },
            "summary": {},
            "metrics": {},
            "positions": [],
            "active_strategies": 0,
            "trade_analysis": {}
        }

@router.get("/equity-curve")
async def get_equity_curve(
    source: str = Query(..., pattern="^(backtest|paper-trading)$"),
    id: Optional[str] = None,
    interval: str = Query("daily", pattern="^(tick|hourly|daily|weekly|monthly)$")
) -> EquityCurveResponse:
    """Get equity curve data"""
    
    if source == "backtest":
        if not id or id not in _backtest_results:
            raise HTTPException(
                status_code=404,
                detail="Backtest not found"
            )
        
        result = _backtest_results[id]
        if result["status"] != "completed":
            raise HTTPException(
                status_code=400,
                detail="Backtest not completed"
            )
        
        equity_curve = result["result"].equity_curve if hasattr(result["result"], 'equity_curve') else pd.DataFrame()
        
    else:  # paper-trading
        try:
            from ...main import paper_engine
            
            # Build equity curve from trades
            trades = paper_engine.virtual_broker.trades if paper_engine else []
            if id:  # Filter by strategy
                trades = [t for t in trades if t.get("strategy_id") == id]
            
            equity_data = []
            current_value = paper_engine.virtual_broker.initial_capital if paper_engine else 1000000
            
            for trade in sorted(trades, key=lambda x: x["timestamp"]):
                if trade.get("pnl"):
                    current_value += trade["pnl"] - trade.get("commission", 0)
                    equity_data.append({
                        "timestamp": trade["timestamp"],
                        "portfolio_value": current_value
                    })
            
            equity_curve = pd.DataFrame(equity_data)
            if not equity_curve.empty:
                equity_curve.set_index("timestamp", inplace=True)
                
        except ImportError:
            equity_curve = pd.DataFrame()
    
    if equity_curve.empty:
        return EquityCurveResponse(
            data=[],
            start_date=datetime.now(),
            end_date=datetime.now(),
            initial_value=0,
            final_value=0,
            max_value=0,
            min_value=0
        )
    
    # Resample based on interval
    if interval != "tick" and not equity_curve.empty:
        resample_map = {
            "hourly": "H",
            "daily": "D",
            "weekly": "W",
            "monthly": "M"
        }
        equity_curve = equity_curve.resample(resample_map[interval]).last().dropna()
    
    # Calculate daily PnL
    equity_curve["daily_pnl"] = equity_curve["portfolio_value"].diff()
    equity_curve["cumulative_pnl"] = equity_curve["portfolio_value"] - equity_curve["portfolio_value"].iloc[0]
    
    # Convert to response format
    data_points = []
    for timestamp, row in equity_curve.iterrows():
        data_points.append(EquityCurvePoint(
            timestamp=timestamp,
            portfolio_value=row["portfolio_value"],
            capital=row.get("capital", row["portfolio_value"]),
            positions_value=row.get("positions_value", 0),
            daily_pnl=row.get("daily_pnl"),
            cumulative_pnl=row["cumulative_pnl"]
        ))
    
    return EquityCurveResponse(
        data=data_points,
        start_date=equity_curve.index[0],
        end_date=equity_curve.index[-1],
        initial_value=equity_curve["portfolio_value"].iloc[0],
        final_value=equity_curve["portfolio_value"].iloc[-1],
        max_value=equity_curve["portfolio_value"].max(),
        min_value=equity_curve["portfolio_value"].min()
    )

@router.post("/generate", response_model=Dict[str, Any])
async def generate_custom_report(request: ReportRequest):
    """Generate custom reports based on criteria"""
    
    if request.report_type == "performance":
        return await get_paper_trading_performance_report(
            start_date=request.start_date,
            end_date=request.end_date,
            strategy_id=request.strategy_id
        )
    
    elif request.report_type == "trades":
        try:
            from ...main import paper_engine
            trades = paper_engine.virtual_broker.trades if paper_engine else []
            
            # Apply filters
            if request.strategy_id:
                trades = [t for t in trades if t.get("strategy_id") == request.strategy_id]
            if request.symbols:
                trades = [t for t in trades if t["symbol"] in request.symbols]
            if request.start_date:
                trades = [t for t in trades if t["timestamp"] >= request.start_date]
            if request.end_date:
                trades = [t for t in trades if t["timestamp"] <= request.end_date]
            
            return {
                "trades": trades,
                "total": len(trades),
                "filters_applied": {
                    "strategy_id": request.strategy_id,
                    "symbols": request.symbols,
                    "date_range": {
                        "start": request.start_date,
                        "end": request.end_date
                    }
                }
            }
        except ImportError:
            return {
                "trades": [],
                "total": 0,
                "error": "Paper trading engine not available"
            }
    
    elif request.report_type == "positions":
        try:
            from ...main import paper_engine
            positions = paper_engine.virtual_broker.get_positions() if paper_engine else {}
            
            # Filter by symbols if specified
            if request.symbols:
                positions = {s: p for s, p in positions.items() if s in request.symbols}
            
            return {
                "positions": positions,
                "total": len(positions),
                "total_value": sum(
                    p["quantity"] * (paper_engine.virtual_broker.market_prices.get(s, p["avg_price"]) if paper_engine else p["avg_price"])
                    for s, p in positions.items()
                )
            }
        except ImportError:
            return {
                "positions": {},
                "total": 0,
                "total_value": 0,
                "error": "Paper trading engine not available"
            }
    
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown report type: {request.report_type}"
        )

# Helper functions

def _calculate_monthly_returns(equity_curve: pd.DataFrame) -> Dict[str, float]:
    """Calculate monthly returns from equity curve"""
    if equity_curve.empty or 'portfolio_value' not in equity_curve.columns:
        return {}
    
    monthly = equity_curve['portfolio_value'].resample('M').last()
    monthly_returns = monthly.pct_change().dropna()
    
    return {
        str(date.date()): float(ret * 100) 
        for date, ret in monthly_returns.items()
    }

def _calculate_yearly_returns(equity_curve: pd.DataFrame) -> Dict[str, float]:
    """Calculate yearly returns from equity curve"""
    if equity_curve.empty or 'portfolio_value' not in equity_curve.columns:
        return {}
    
    yearly = equity_curve['portfolio_value'].resample('Y').last()
    yearly_returns = yearly.pct_change().dropna()
    
    return {
        str(date.year): float(ret * 100) 
        for date, ret in yearly_returns.items()
    }

def _analyze_drawdowns(equity_curve: pd.DataFrame) -> Dict[str, Any]:
    """Analyze drawdowns from equity curve"""
    if equity_curve.empty or 'portfolio_value' not in equity_curve.columns:
        return {}
    
    values = equity_curve['portfolio_value']
    cummax = values.cummax()
    drawdown = (values - cummax) / cummax
    
    # Find drawdown periods
    drawdown_start = None
    drawdown_periods = []
    
    for i in range(len(drawdown)):
        if drawdown.iloc[i] < 0 and drawdown_start is None:
            drawdown_start = i
        elif drawdown.iloc[i] >= 0 and drawdown_start is not None:
            period = {
                "start": equity_curve.index[drawdown_start],
                "end": equity_curve.index[i],
                "max_drawdown": float(drawdown.iloc[drawdown_start:i].min() * 100),
                "duration_days": (equity_curve.index[i] - equity_curve.index[drawdown_start]).days
            }
            drawdown_periods.append(period)
            drawdown_start = None
    
    return {
        "max_drawdown": float(drawdown.min() * 100),
        "current_drawdown": float(drawdown.iloc[-1] * 100),
        "drawdown_periods": drawdown_periods,
        "avg_drawdown": float(drawdown[drawdown < 0].mean() * 100) if len(drawdown[drawdown < 0]) > 0 else 0,
        "drawdown_frequency": len(drawdown_periods)
    }

def _analyze_trades(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze trade statistics"""
    if not trades:
        return {}
    
    df = pd.DataFrame(trades)
    
    # Filter completed trades
    completed_trades = df[df['side'] == 'SELL'] if 'side' in df.columns else df
    
    if completed_trades.empty:
        return {}
    
    # Calculate statistics
    winning_trades = completed_trades[completed_trades.get('pnl', 0) > 0]
    losing_trades = completed_trades[completed_trades.get('pnl', 0) < 0]
    
    # Calculate holding periods
    if 'exit_time' in completed_trades.columns and 'timestamp' in completed_trades.columns:
        completed_trades['holding_period'] = (
            pd.to_datetime(completed_trades['exit_time']) - 
            pd.to_datetime(completed_trades['timestamp'])
        ).dt.total_seconds() / 3600  # Convert to hours
        avg_holding_period = completed_trades['holding_period'].mean()
    else:
        avg_holding_period = None
    
    return {
        "total_trades": len(completed_trades),
        "winning_trades": len(winning_trades),
        "losing_trades": len(losing_trades),
        "win_rate": len(winning_trades) / len(completed_trades) * 100 if len(completed_trades) > 0 else 0,
        "avg_win": float(winning_trades['pnl'].mean()) if not winning_trades.empty else 0,
        "avg_loss": float(losing_trades['pnl'].mean()) if not losing_trades.empty else 0,
        "largest_win": float(winning_trades['pnl'].max()) if not winning_trades.empty else 0,
        "largest_loss": float(losing_trades['pnl'].min()) if not losing_trades.empty else 0,
        "profit_factor": abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if not losing_trades.empty and losing_trades['pnl'].sum() != 0 else 0,
        "expectancy": float(completed_trades['pnl'].mean()) if 'pnl' in completed_trades.columns else 0,
        "avg_holding_period_hours": float(avg_holding_period) if avg_holding_period else None
    }

def _analyze_symbol_performance(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze performance by symbol"""
    if not trades:
        return {}
    
    df = pd.DataFrame(trades)
    symbol_stats = {}
    
    for symbol in df['symbol'].unique():
        symbol_trades = df[df['symbol'] == symbol]
        completed = symbol_trades[symbol_trades['side'] == 'SELL'] if 'side' in symbol_trades.columns else symbol_trades
        
        if not completed.empty and 'pnl' in completed.columns:
            symbol_stats[symbol] = {
                "total_trades": len(completed),
                "total_pnl": float(completed['pnl'].sum()),
                "avg_pnl": float(completed['pnl'].mean()),
                "win_rate": len(completed[completed['pnl'] > 0]) / len(completed) * 100
            }
    
    return symbol_stats

def _analyze_time_patterns(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze trading patterns by time"""
    if not trades:
        return {}
    
    df = pd.DataFrame(trades)
    if 'timestamp' not in df.columns:
        return {}
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.day_name()
    
    # Hourly distribution
    hourly_dist = df.groupby('hour').size().to_dict()
    
    # Day of week distribution
    dow_dist = df.groupby('day_of_week').size().to_dict()
    
    return {
        "hourly_distribution": hourly_dist,
        "day_of_week_distribution": dow_dist,
        "most_active_hour": max(hourly_dist, key=hourly_dist.get) if hourly_dist else None,
        "most_active_day": max(dow_dist, key=dow_dist.get) if dow_dist else None
    }