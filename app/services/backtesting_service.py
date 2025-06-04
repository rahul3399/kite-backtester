import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
import json
from ..services.kite_client import KiteClient
from ..services.spread_calculator import SpreadCalculator
from ..models.trading import Trade, TradeSignal, TradeAction, SpreadData

logger = logging.getLogger(__name__)

@dataclass
class BacktestTrade:
    """Trade record for backtesting"""
    trade_id: str
    entry_time: datetime
    exit_time: Optional[datetime]
    instrument1: str
    instrument2: str
    entry_price1: float
    entry_price2: float
    exit_price1: Optional[float]
    exit_price2: Optional[float]
    entry_spread: float
    exit_spread: Optional[float]
    entry_z_score: float
    exit_z_score: Optional[float]
    trade_type: str  # LONG_SPREAD or SHORT_SPREAD
    pnl: Optional[float]
    pnl_points: Optional[float]
    trade_duration_minutes: Optional[float]
    exit_reason: Optional[str]
    max_favorable_move: Optional[float]
    max_adverse_move: Optional[float]
    
    def to_dict(self):
        """Convert to dictionary with string timestamps"""
        data = asdict(self)
        data['entry_time'] = self.entry_time.isoformat()
        if self.exit_time:
            data['exit_time'] = self.exit_time.isoformat()
        return data

@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    entry_z_score: float = 2.0
    exit_z_score: float = 0.5
    stop_loss_z_score: float = 3.0
    lookback_period: int = 20  # For moving average
    lot_size: int = 75
    transaction_cost: float = 40  # Per lot round trip
    slippage_points: float = 0.5  # Points of slippage
    
@dataclass
class BacktestResults:
    """Results of backtesting"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    average_pnl: float
    average_win: float
    average_loss: float
    profit_factor: float
    max_drawdown: float
    max_drawdown_percent: float
    sharpe_ratio: float
    calmar_ratio: float
    average_trade_duration: float
    trades: List[BacktestTrade]
    equity_curve: List[float]
    daily_returns: List[float]
    monthly_returns: Dict[str, float]

class BacktestingService:
    def __init__(self, kite_client: KiteClient):
        self.kite_client = kite_client
        self.spread_calculator = SpreadCalculator()
        
    def fetch_historical_data(self, 
                            instrument_token: int,
                            from_date: datetime,
                            to_date: datetime,
                            interval: str = "minute") -> pd.DataFrame:
        """Fetch historical data from Kite"""
        try:
            data = self.kite_client.kite.historical_data(
                instrument_token=instrument_token,
                from_date=from_date,
                to_date=to_date,
                interval=interval
            )
            
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch historical data: {e}")
            raise
    
    def prepare_spread_data(self,
                          df1: pd.DataFrame,
                          df2: pd.DataFrame,
                          config: BacktestConfig) -> pd.DataFrame:
        """Prepare spread data with indicators"""
        # Align dataframes
        df = pd.DataFrame(index=df1.index)
        df['price1'] = df1['close']
        df['price2'] = df2['close']
        
        # Calculate spread
        df['spread'] = df['price1'] - df['price2']
        df['spread_pct'] = (df['spread'] / df['price2']) * 100
        
        # Calculate moving statistics
        df['spread_ma'] = df['spread'].rolling(window=config.lookback_period).mean()
        df['spread_std'] = df['spread'].rolling(window=config.lookback_period).std()
        
        # Calculate z-score
        df['z_score'] = (df['spread'] - df['spread_ma']) / df['spread_std']
        
        # Drop NaN values
        df.dropna(inplace=True)
        
        return df
    
    def simulate_trades(self,
                       spread_df: pd.DataFrame,
                       config: BacktestConfig,
                       instrument1: str,
                       instrument2: str) -> List[BacktestTrade]:
        """Simulate trades based on historical spread data"""
        trades = []
        current_trade = None
        equity = 100000  # Starting capital
        equity_curve = [equity]
        
        for idx, row in spread_df.iterrows():
            z_score = row['z_score']
            
            # Check if we have an open trade
            if current_trade is None:
                # Entry logic
                signal = None
                
                if z_score >= config.entry_z_score:
                    signal = "SHORT_SPREAD"
                elif z_score <= -config.entry_z_score:
                    signal = "LONG_SPREAD"
                
                if signal:
                    # Open new trade
                    current_trade = BacktestTrade(
                        trade_id=f"BT_{idx.strftime('%Y%m%d_%H%M%S')}",
                        entry_time=idx,
                        exit_time=None,
                        instrument1=instrument1,
                        instrument2=instrument2,
                        entry_price1=row['price1'],
                        entry_price2=row['price2'],
                        exit_price1=None,
                        exit_price2=None,
                        entry_spread=row['spread'],
                        exit_spread=None,
                        entry_z_score=z_score,
                        exit_z_score=None,
                        trade_type=signal,
                        pnl=None,
                        pnl_points=None,
                        trade_duration_minutes=None,
                        exit_reason=None,
                        max_favorable_move=0,
                        max_adverse_move=0
                    )
                    
            else:
                # Exit logic for open trade
                exit_signal = False
                exit_reason = None
                
                # Track max favorable/adverse moves
                current_pnl_points = self._calculate_pnl_points(
                    current_trade, row['spread'], row['spread']
                )
                
                if current_pnl_points > current_trade.max_favorable_move:
                    current_trade.max_favorable_move = current_pnl_points
                if current_pnl_points < current_trade.max_adverse_move:
                    current_trade.max_adverse_move = current_pnl_points
                
                # Check exit conditions
                if current_trade.trade_type == "LONG_SPREAD":
                    if z_score >= -config.exit_z_score:
                        exit_signal = True
                        exit_reason = "Target"
                    elif z_score <= -config.stop_loss_z_score:
                        exit_signal = True
                        exit_reason = "Stop Loss"
                        
                elif current_trade.trade_type == "SHORT_SPREAD":
                    if z_score <= config.exit_z_score:
                        exit_signal = True
                        exit_reason = "Target"
                    elif z_score >= config.stop_loss_z_score:
                        exit_signal = True
                        exit_reason = "Stop Loss"
                
                # Time-based exit (end of day)
                if idx.hour == 15 and idx.minute >= 25:
                    exit_signal = True
                    exit_reason = "EOD"
                
                if exit_signal:
                    # Close trade
                    current_trade.exit_time = idx
                    current_trade.exit_price1 = row['price1']
                    current_trade.exit_price2 = row['price2']
                    current_trade.exit_spread = row['spread']
                    current_trade.exit_z_score = z_score
                    current_trade.exit_reason = exit_reason
                    
                    # Calculate P&L
                    pnl_points = self._calculate_pnl_points(
                        current_trade, 
                        current_trade.entry_spread,
                        current_trade.exit_spread
                    )
                    
                    # Apply slippage and transaction costs
                    pnl_points -= config.slippage_points * 2  # Entry and exit
                    pnl = (pnl_points * config.lot_size) - config.transaction_cost
                    
                    current_trade.pnl_points = pnl_points
                    current_trade.pnl = pnl
                    current_trade.trade_duration_minutes = (
                        current_trade.exit_time - current_trade.entry_time
                    ).total_seconds() / 60
                    
                    trades.append(current_trade)
                    current_trade = None
                    
                    # Update equity
                    equity += pnl
                    equity_curve.append(equity)
        
        # Close any open trade at the end
        if current_trade:
            current_trade.exit_time = spread_df.index[-1]
            current_trade.exit_price1 = spread_df.iloc[-1]['price1']
            current_trade.exit_price2 = spread_df.iloc[-1]['price2']
            current_trade.exit_spread = spread_df.iloc[-1]['spread']
            current_trade.exit_z_score = spread_df.iloc[-1]['z_score']
            current_trade.exit_reason = "End of Data"
            
            pnl_points = self._calculate_pnl_points(
                current_trade,
                current_trade.entry_spread,
                current_trade.exit_spread
            )
            
            pnl_points -= config.slippage_points * 2
            pnl = (pnl_points * config.lot_size) - config.transaction_cost
            
            current_trade.pnl_points = pnl_points
            current_trade.pnl = pnl
            current_trade.trade_duration_minutes = (
                current_trade.exit_time - current_trade.entry_time
            ).total_seconds() / 60
            
            trades.append(current_trade)
            equity += pnl
            equity_curve.append(equity)
        
        return trades, equity_curve
    
    def _calculate_pnl_points(self, trade: BacktestTrade, 
                            entry_spread: float, exit_spread: float) -> float:
        """Calculate P&L in points based on trade type"""
        if trade.trade_type == "LONG_SPREAD":
            return exit_spread - entry_spread
        else:  # SHORT_SPREAD
            return entry_spread - exit_spread
    
    def calculate_metrics(self, trades: List[BacktestTrade], 
                        equity_curve: List[float],
                        initial_capital: float = 100000) -> BacktestResults:
        """Calculate performance metrics"""
        if not trades:
            return BacktestResults(
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0,
                total_pnl=0,
                average_pnl=0,
                average_win=0,
                average_loss=0,
                profit_factor=0,
                max_drawdown=0,
                max_drawdown_percent=0,
                sharpe_ratio=0,
                calmar_ratio=0,
                average_trade_duration=0,
                trades=trades,
                equity_curve=equity_curve,
                daily_returns=[],
                monthly_returns={}
            )
        
        # Basic metrics
        pnls = [t.pnl for t in trades]
        winning_trades = [p for p in pnls if p > 0]
        losing_trades = [p for p in pnls if p < 0]
        
        total_trades = len(trades)
        num_winning = len(winning_trades)
        num_losing = len(losing_trades)
        win_rate = (num_winning / total_trades * 100) if total_trades > 0 else 0
        
        total_pnl = sum(pnls)
        average_pnl = total_pnl / total_trades if total_trades > 0 else 0
        average_win = sum(winning_trades) / num_winning if num_winning > 0 else 0
        average_loss = sum(losing_trades) / num_losing if num_losing > 0 else 0
        
        # Profit factor
        gross_profit = sum(winning_trades) if winning_trades else 0
        gross_loss = abs(sum(losing_trades)) if losing_trades else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Drawdown
        max_drawdown, max_dd_percent = self._calculate_drawdown(equity_curve, initial_capital)
        
        # Returns for Sharpe ratio
        returns = pd.Series(equity_curve).pct_change().dropna()
        
        # Daily returns (assuming intraday trading)
        daily_returns = []
        trades_df = pd.DataFrame([t.to_dict() for t in trades])
        if not trades_df.empty:
            trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
            trades_df['date'] = trades_df['exit_time'].dt.date
            daily_pnl = trades_df.groupby('date')['pnl'].sum()
            daily_returns = (daily_pnl / initial_capital * 100).tolist()
        
        # Sharpe ratio (annualized)
        if len(returns) > 1:
            sharpe_ratio = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Calmar ratio
        annual_return = (equity_curve[-1] / initial_capital - 1) * 100
        calmar_ratio = annual_return / abs(max_dd_percent) if max_dd_percent != 0 else 0
        
        # Average trade duration
        durations = [t.trade_duration_minutes for t in trades if t.trade_duration_minutes]
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        # Monthly returns
        monthly_returns = {}
        if not trades_df.empty:
            trades_df['month'] = trades_df['exit_time'].dt.to_period('M')
            monthly_pnl = trades_df.groupby('month')['pnl'].sum()
            monthly_returns = {
                str(month): pnl 
                for month, pnl in monthly_pnl.items()
            }
        
        return BacktestResults(
            total_trades=total_trades,
            winning_trades=num_winning,
            losing_trades=num_losing,
            win_rate=win_rate,
            total_pnl=total_pnl,
            average_pnl=average_pnl,
            average_win=average_win,
            average_loss=average_loss,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            max_drawdown_percent=max_dd_percent,
            sharpe_ratio=sharpe_ratio,
            calmar_ratio=calmar_ratio,
            average_trade_duration=avg_duration,
            trades=trades,
            equity_curve=equity_curve,
            daily_returns=daily_returns,
            monthly_returns=monthly_returns
        )
    
    def _calculate_drawdown(self, equity_curve: List[float], 
                          initial_capital: float) -> Tuple[float, float]:
        """Calculate maximum drawdown"""
        if not equity_curve:
            return 0, 0
        
        peak = equity_curve[0]
        max_dd = 0
        max_dd_percent = 0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            
            dd = peak - value
            dd_percent = (dd / peak * 100) if peak > 0 else 0
            
            if dd > max_dd:
                max_dd = dd
                max_dd_percent = dd_percent
        
        return max_dd, max_dd_percent
    
    async def run_backtest(self,
                          instrument1: str,
                          instrument2: str,
                          token1: int,
                          token2: int,
                          from_date: datetime,
                          to_date: datetime,
                          config: Optional[BacktestConfig] = None) -> Dict:
        """Run complete backtest"""
        if config is None:
            config = BacktestConfig()
        
        try:
            logger.info(f"Starting backtest for {instrument1} - {instrument2}")
            
            # Fetch historical data
            df1 = self.fetch_historical_data(token1, from_date, to_date)
            df2 = self.fetch_historical_data(token2, from_date, to_date)
            
            logger.info(f"Fetched {len(df1)} candles for {instrument1}")
            logger.info(f"Fetched {len(df2)} candles for {instrument2}")
            
            # Prepare spread data
            spread_df = self.prepare_spread_data(df1, df2, config)
            logger.info(f"Prepared {len(spread_df)} spread data points")
            
            # Simulate trades
            trades, equity_curve = self.simulate_trades(
                spread_df, config, instrument1, instrument2
            )
            logger.info(f"Simulated {len(trades)} trades")
            
            # Calculate metrics
            results = self.calculate_metrics(trades, equity_curve)
            
            # Convert to dictionary for JSON serialization
            results_dict = {
                "config": asdict(config),
                "period": {
                    "from": from_date.isoformat(),
                    "to": to_date.isoformat(),
                    "days": (to_date - from_date).days
                },
                "instruments": {
                    "instrument1": instrument1,
                    "instrument2": instrument2
                },
                "metrics": {
                    "total_trades": results.total_trades,
                    "winning_trades": results.winning_trades,
                    "losing_trades": results.losing_trades,
                    "win_rate": round(results.win_rate, 2),
                    "total_pnl": round(results.total_pnl, 2),
                    "average_pnl": round(results.average_pnl, 2),
                    "average_win": round(results.average_win, 2),
                    "average_loss": round(results.average_loss, 2),
                    "profit_factor": round(results.profit_factor, 2),
                    "max_drawdown": round(results.max_drawdown, 2),
                    "max_drawdown_percent": round(results.max_drawdown_percent, 2),
                    "sharpe_ratio": round(results.sharpe_ratio, 2),
                    "calmar_ratio": round(results.calmar_ratio, 2),
                    "average_trade_duration": round(results.average_trade_duration, 2)
                },
                "monthly_returns": results.monthly_returns,
                "trades": [t.to_dict() for t in results.trades]
            }
            
            # Save results
            self._save_backtest_results(results_dict)
            
            return results_dict
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            raise
    
    def _save_backtest_results(self, results: Dict):
        """Save backtest results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"logs/backtest_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Backtest results saved to {filename}")