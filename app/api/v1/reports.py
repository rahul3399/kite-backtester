# app/api/v1/reports.py
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse, FileResponse
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import json
import io
import logging

from ...api.models.requests import ReportRequest
from ...api.models.responses import EquityCurveResponse, EquityCurvePoint
from ..backtest import backtest_results
from ..paper_trade import active_strategies

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/backtest/{backtest_id}/summary")
async def get_backtest_summary(backtest_id: str):
    """
    Get comprehensive backtest report
    
    Returns detailed analysis including:
    - Performance metrics
    - Monthly returns
    - Drawdown analysis
    - Trade statistics
    """
    
    if backtest_id not in backtest_results:
        raise HTTPException(
            status_code=404,
            detail="Backtest not found"
        )
    
    result = backtest_results[backtest_id]
    if result["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Backtest is not completed. Current status: {result['status']}"
        )
    
    backtest_result = result["result"]
    
    # Generate comprehensive report
    report = {
        "summary": {
            "strategy": backtest_result.strategy_name,
            "period": {
                "start": backtest_result.start_date.isoformat(),
                "end": backtest_result.end_date.isoformat(),
                "days": (backtest_result.end_date - backtest_result.start_date).days
            },
            "symbols": backtest_result.symbols,
            "initial_capital": backtest_result.initial_capital,
            "final_capital": backtest_result.final_capital,
            "total_return": ((backtest_result.final_capital - backtest_result.initial_capital) / 
                           backtest_result.initial_capital * 100),
            "total_trades": len(backtest_result.trades),
            "avg_trades_per_day": len(backtest_result.trades) / max(1, (backtest_result.end_date - backtest_result.start_date).days)
        },
        "metrics": backtest_result.metrics,
        "monthly_returns": _calculate_monthly_returns(backtest_result.equity_curve),
        "yearly_returns": _calculate_yearly_returns(backtest_result.equity_curve),
        "drawdown_analysis": _analyze_drawdowns(backtest_result.equity_curve),
        "trade_analysis": _analyze_trades(backtest_result.trades),
        "symbol_performance": _analyze_symbol_performance(backtest_result.trades),
        "time_analysis": _analyze_time_patterns(backtest_result.trades)
    }
    
    return report

@router.get("/backtest/{backtest_id}/export")
async def export_backtest_results(
    backtest_id: str,
    format: str = Query("csv", pattern="^(csv|json|excel)$", description="Export format")
):
    """Export backtest results in various formats"""
    
    if backtest_id not in backtest_results:
        raise HTTPException(
            status_code=404,
            detail="Backtest not found"
        )
    
    result = backtest_results[backtest_id]
    if result["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Backtest is not completed. Current status: {result['status']}"
        )
    
    backtest_result = result["result"]
    
    if format == "csv":
        # Export trades as CSV
        df = pd.DataFrame(backtest_result.trades)
        
        # Add additional columns
        if not df.empty:
            df['strategy'] = backtest_result.strategy_name
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
                "strategy": backtest_result.strategy_name,
                "start_date": backtest_result.start_date.isoformat(),
                "end_date": backtest_result.end_date.isoformat(),
                "initial_capital": backtest_result.initial_capital,
                "final_capital": backtest_result.final_capital,
                "total_return": ((backtest_result.final_capital - backtest_result.initial_capital) / 
                               backtest_result.initial_capital * 100)
            },
            "metrics": backtest_result.metrics,
            "trades": backtest_result.trades,
            "equity_curve": backtest_result.equity_curve.to_dict() if not backtest_result.equity_curve.empty else {},
            "monthly_returns": _calculate_monthly_returns(backtest_result.equity_curve)
        }
        
        return StreamingResponse(
            io.BytesIO(json.dumps(export_data, indent=2, default=str).encode()),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=backtest_{backtest_id}.json"}
        )
    
    elif format == "excel":
        # Export as Excel with multiple sheets
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Summary sheet
            summary_df = pd.DataFrame([{
                'Strategy': backtest_result.strategy_name,
                'Start Date': backtest_result.start_date,
                'End Date': backtest_result.end_date,
                'Initial Capital': backtest_result.initial_capital,
                'Final Capital': backtest_result.final_capital,
                'Total Return %': ((backtest_result.final_capital - backtest_result.initial_capital) / 
                                 backtest_result.initial_capital * 100),
                'Total Trades': len(backtest_result.trades)
            }])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Metrics sheet
            metrics_df = pd.DataFrame([backtest_result.metrics])
            metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
            
            # Trades sheet
            trades_df = pd.DataFrame(backtest_result.trades)
            if not trades_df.empty:
                trades_df.to_excel(writer, sheet_name='Trades', index=False)
            
            # Equity curve sheet
            if not backtest_result.equity_curve.empty:
                equity_df = backtest_result.equity_curve.reset_index()
                equity_df.to_excel(writer, sheet_name='Equity Curve', index=False)
        
        output.seek(0)
        
        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename=backtest_{backtest_id}.xlsx"}
        )

@router.get("/paper-trading/performance")
async def get_paper_trading_performance_report(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    strategy_id: Optional[str] = None
):
    """Get paper trading performance report"""
    
    from ...main import paper_engine
    
    # Get all trades
    trades = paper_engine.virtual_broker.trades
    
    # Apply filters
    if strategy_id:
        trades = [t for t in trades if t.get("strategy_id") == strategy_id]
    
    if start_date:
        trades = [t for t in trades if t["timestamp"] >= start_date]
    
    if end_date:
        trades = [t for t in trades if t["timestamp"] <= end_date]
    
    # Get current positions
    positions = paper_engine.virtual_broker.get_positions()
    
    # Calculate metrics
    from ...reporting.metrics_calculator import MetricsCalculator
    calculator = MetricsCalculator()
    
    # Create equity curve from trades
    equity_data = []
    current_value = paper_engine.virtual_broker.initial_capital
    
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
        trades, equity_df, paper_engine.virtual_broker.initial_capital
    )
    
    # Build report
    report = {
        "period": {
            "start": start_date.isoformat() if start_date else None,
            "end": end_date.isoformat() if end_date else None
        },
        "summary": paper_engine.virtual_broker.get_performance_summary(),
        "metrics": metrics,
        "positions": [
            {
                "symbol": symbol,
                "quantity": pos["quantity"],
                "avg_price": pos["avg_price"],
                "current_value": pos["quantity"] * paper_engine.virtual_broker.market_prices.get(symbol, pos["avg_price"]),
                "unrealized_pnl": pos.get("unrealized_pnl", 0)
            }
            for symbol, pos in positions.items()
        ],
        "active_strategies": len([s for s in active_strategies.values() if s["status"] == "running"]),
        "trade_analysis": _analyze_trades(trades) if trades else {}
    }
    
    return report

@router.get("/equity-curve")
async def get_equity_curve(
    source: str = Query(..., pattern="^(backtest|paper-trading)$"),
    id: Optional[str] = None,
    interval: str = Query("daily", pattern="^(tick|hourly|daily|weekly|monthly)$")
) -> EquityCurveResponse:
    """Get equity curve data"""
    
    if source == "backtest":
        if not id or id not in backtest_results:
            raise HTTPException(
                status_code=404,
                detail="Backtest not found"
            )
        
        result = backtest_results[id]
        if result["status"] != "completed":
            raise HTTPException(
                status_code=400,
                detail="Backtest not completed"
            )
        
        equity_curve = result["result"].equity_curve
        
    else:  # paper-trading
        from ...main import paper_engine
        
        # Build equity curve from trades
        trades = paper_engine.virtual_broker.trades
        if id:  # Filter by strategy
            trades = [t for t in trades if t.get("strategy_id") == id]
        
        equity_data = []
        current_value = paper_engine.virtual_broker.initial_capital
        
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
        from ...main import paper_engine
        trades = paper_engine.virtual_broker.trades
        
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
    
    elif request.report_type == "positions":
        from ...main import paper_engine
        positions = paper_engine.virtual_broker.get_positions()
        
        # Filter by symbols if specified
        if request.symbols:
            positions = {s: p for s, p in positions.items() if s in request.symbols}
        
        return {
            "positions": positions,
            "total": len(positions),
            "total_value": sum(
                p["quantity"] * paper_engine.virtual_broker.market_prices.get(s, p["avg_price"])
                for s, p in positions.items()
            )
        }
    
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown report type: {request.report_type}"
        )

# Helper functions

def _calculate_monthly_returns(equity_curve: pd.DataFrame) -> Dict[str, float]:
    """Calculate monthly returns from equity curve"""
    if equity_curve.empty:
        return {}
    
    monthly = equity_curve['portfolio_value'].resample('M').last()
    monthly_returns = monthly.pct_change().dropna()
    
    return {
        str(date.date()): float(ret * 100) 
        for date, ret in monthly_returns.items()
    }

def _calculate_yearly_returns(equity_curve: pd.DataFrame) -> Dict[str, float]:
    """Calculate yearly returns from equity curve"""
    if equity_curve.empty:
        return {}
    
    yearly = equity_curve['portfolio_value'].resample('Y').last()
    yearly_returns = yearly.pct_change().dropna()
    
    return {
        str(date.year): float(ret * 100) 
        for date, ret in yearly_returns.items()
    }

def _analyze_drawdowns(equity_curve: pd.DataFrame) -> Dict[str, Any]:
    """Analyze drawdowns from equity curve"""
    if equity_curve.empty:
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