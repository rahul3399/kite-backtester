# app/core/metrics.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
from scipy import stats
from collections import defaultdict

logger = logging.getLogger(__name__)

class PerformanceMetrics:
    """
    Comprehensive performance metrics calculator for trading strategies
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize metrics calculator
        
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = 252
        
    def calculate_metrics(self, 
                         trades: List[Dict[str, Any]], 
                         equity_curve: pd.DataFrame,
                         initial_capital: float) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics
        
        Args:
            trades: List of trade dictionaries
            equity_curve: DataFrame with portfolio values over time
            initial_capital: Starting capital
            
        Returns:
            Dictionary of performance metrics
        """
        
        metrics = {}
        
        # Basic metrics
        metrics.update(self._calculate_basic_metrics(trades, equity_curve, initial_capital))
        
        # Risk metrics
        metrics.update(self._calculate_risk_metrics(equity_curve))
        
        # Trade metrics
        metrics.update(self._calculate_trade_metrics(trades))
        
        # Advanced metrics
        metrics.update(self._calculate_advanced_metrics(trades, equity_curve))
        
        return metrics
        
    def _calculate_basic_metrics(self, 
                               trades: List[Dict[str, Any]], 
                               equity_curve: pd.DataFrame,
                               initial_capital: float) -> Dict[str, float]:
        """Calculate basic performance metrics"""
        
        metrics = {}
        
        if equity_curve.empty:
            return metrics
            
        # Total return
        final_value = equity_curve['portfolio_value'].iloc[-1]
        total_return = (final_value - initial_capital) / initial_capital
        metrics['total_return'] = total_return * 100
        
        # Annualized return
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        if days > 0:
            years = days / 365
            annualized_return = (final_value / initial_capital) ** (1/years) - 1
            metrics['annualized_return'] = annualized_return * 100
        else:
            metrics['annualized_return'] = 0
            
        # Calculate daily returns
        if 'portfolio_value' in equity_curve.columns:
            daily_returns = equity_curve['portfolio_value'].pct_change().dropna()
            
            if len(daily_returns) > 0:
                metrics['avg_daily_return'] = daily_returns.mean() * 100
                metrics['daily_volatility'] = daily_returns.std() * 100
                metrics['annual_volatility'] = daily_returns.std() * np.sqrt(self.trading_days_per_year) * 100
            
        return metrics
        
    def _calculate_risk_metrics(self, equity_curve: pd.DataFrame) -> Dict[str, float]:
        """Calculate risk-related metrics"""
        
        metrics = {}
        
        if equity_curve.empty or 'portfolio_value' not in equity_curve.columns:
            return metrics
            
        values = equity_curve['portfolio_value']
        daily_returns = values.pct_change().dropna()
        
        if len(daily_returns) == 0:
            return metrics
            
        # Sharpe ratio
        if daily_returns.std() > 0:
            daily_risk_free = self.risk_free_rate / self.trading_days_per_year
            excess_returns = daily_returns - daily_risk_free
            sharpe_ratio = np.sqrt(self.trading_days_per_year) * excess_returns.mean() / daily_returns.std()
            metrics['sharpe_ratio'] = sharpe_ratio
        else:
            metrics['sharpe_ratio'] = 0
            
        # Sortino ratio
        downside_returns = daily_returns[daily_returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino_ratio = np.sqrt(self.trading_days_per_year) * daily_returns.mean() / downside_returns.std()
            metrics['sortino_ratio'] = sortino_ratio
        else:
            metrics['sortino_ratio'] = 0
            
        # Maximum drawdown
        cumulative = (1 + daily_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        
        metrics['max_drawdown'] = drawdown.min() * 100
        metrics['max_drawdown_duration'] = self._calculate_max_drawdown_duration(drawdown)
        
        # Calmar ratio
        if metrics['max_drawdown'] != 0 and 'annualized_return' in metrics:
            metrics['calmar_ratio'] = metrics['annualized_return'] / abs(metrics['max_drawdown'])
        else:
            metrics['calmar_ratio'] = 0
            
        # Value at Risk (VaR)
        metrics['var_95'] = np.percentile(daily_returns, 5) * 100  # 95% VaR
        metrics['var_99'] = np.percentile(daily_returns, 1) * 100  # 99% VaR
        
        # Conditional Value at Risk (CVaR)
        var_95_threshold = np.percentile(daily_returns, 5)
        cvar_returns = daily_returns[daily_returns <= var_95_threshold]
        metrics['cvar_95'] = cvar_returns.mean() * 100 if len(cvar_returns) > 0 else 0
        
        # Beta (assuming market return of 0 for simplicity)
        # In production, compare against actual market returns
        metrics['beta'] = 1.0
        
        # Information ratio (would need benchmark returns)
        metrics['information_ratio'] = 0
        
        return metrics
        
    def _calculate_trade_metrics(self, trades: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate trade-specific metrics"""
        
        metrics = {}
        
        if not trades:
            return metrics
            
        # Filter completed trades with PnL
        completed_trades = [t for t in trades if 'pnl' in t and t.get('side') == 'SELL']
        
        if not completed_trades:
            metrics['total_trades'] = 0
            return metrics
            
        # Basic trade stats
        metrics['total_trades'] = len(completed_trades)
        
        # Separate winning and losing trades
        winning_trades = [t for t in completed_trades if t['pnl'] > 0]
        losing_trades = [t for t in completed_trades if t['pnl'] < 0]
        breakeven_trades = [t for t in completed_trades if t['pnl'] == 0]
        
        metrics['winning_trades'] = len(winning_trades)
        metrics['losing_trades'] = len(losing_trades)
        metrics['breakeven_trades'] = len(breakeven_trades)
        
        # Win rate
        metrics['win_rate'] = (len(winning_trades) / len(completed_trades) * 100 
                              if completed_trades else 0)
        
        # Average win/loss
        if winning_trades:
            metrics['avg_win'] = np.mean([t['pnl'] for t in winning_trades])
            metrics['largest_win'] = max([t['pnl'] for t in winning_trades])
        else:
            metrics['avg_win'] = 0
            metrics['largest_win'] = 0
            
        if losing_trades:
            metrics['avg_loss'] = np.mean([t['pnl'] for t in losing_trades])
            metrics['largest_loss'] = min([t['pnl'] for t in losing_trades])
        else:
            metrics['avg_loss'] = 0
            metrics['largest_loss'] = 0
            
        # Profit factor
        if losing_trades and metrics['avg_loss'] != 0:
            gross_profit = sum([t['pnl'] for t in winning_trades])
            gross_loss = abs(sum([t['pnl'] for t in losing_trades]))
            metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else 0
        else:
            metrics['profit_factor'] = float('inf') if winning_trades else 0
            
        # Expectancy
        all_pnls = [t['pnl'] for t in completed_trades]
        metrics['expectancy'] = np.mean(all_pnls) if all_pnls else 0
        
        # Payoff ratio
        if metrics['avg_loss'] != 0:
            metrics['payoff_ratio'] = abs(metrics['avg_win'] / metrics['avg_loss'])
        else:
            metrics['payoff_ratio'] = float('inf') if metrics['avg_win'] > 0 else 0
            
        # Kelly Criterion
        if metrics['win_rate'] > 0 and metrics['payoff_ratio'] > 0:
            win_rate_decimal = metrics['win_rate'] / 100
            metrics['kelly_criterion'] = (win_rate_decimal * metrics['payoff_ratio'] - 
                                        (1 - win_rate_decimal)) / metrics['payoff_ratio']
        else:
            metrics['kelly_criterion'] = 0
            
        # Average trade duration
        durations = []
        for trade in completed_trades:
            if 'timestamp' in trade and 'exit_time' in trade:
                duration = (trade['exit_time'] - trade['timestamp']).total_seconds() / 3600
                durations.append(duration)
                
        if durations:
            metrics['avg_trade_duration_hours'] = np.mean(durations)
            
        # Commission analysis
        total_commission = sum(t.get('commission', 0) for t in trades)
        metrics['total_commission'] = total_commission
        
        return metrics
        
    def _calculate_advanced_metrics(self, 
                                  trades: List[Dict[str, Any]], 
                                  equity_curve: pd.DataFrame) -> Dict[str, float]:
        """Calculate advanced performance metrics"""
        
        metrics = {}
        
        # Recovery factor
        if 'max_drawdown' in metrics and metrics['max_drawdown'] != 0:
            if 'total_return' in metrics:
                metrics['recovery_factor'] = metrics['total_return'] / abs(metrics['max_drawdown'])
                
        # Ulcer Index
        if not equity_curve.empty and 'portfolio_value' in equity_curve.columns:
            ulcer_index = self._calculate_ulcer_index(equity_curve['portfolio_value'])
            metrics['ulcer_index'] = ulcer_index
            
        # Consecutive wins/losses
        consecutive_stats = self._calculate_consecutive_stats(trades)
        metrics.update(consecutive_stats)
        
        # Monthly returns analysis
        if not equity_curve.empty:
            monthly_stats = self._calculate_monthly_stats(equity_curve)
            metrics.update(monthly_stats)
            
        # Risk-adjusted metrics
        if 'sharpe_ratio' in metrics and metrics['sharpe_ratio'] > 0:
            # Omega ratio
            if not equity_curve.empty:
                daily_returns = equity_curve['portfolio_value'].pct_change().dropna()
                threshold = 0  # Can be adjusted
                gains = daily_returns[daily_returns > threshold] - threshold
                losses = threshold - daily_returns[daily_returns <= threshold]
                
                if len(losses) > 0 and losses.sum() > 0:
                    metrics['omega_ratio'] = gains.sum() / losses.sum()
                else:
                    metrics['omega_ratio'] = float('inf') if len(gains) > 0 else 0
                    
        # Stability metrics
        if not equity_curve.empty:
            # R-squared of equity curve
            x = np.arange(len(equity_curve))
            y = equity_curve['portfolio_value'].values
            
            if len(x) > 1:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                metrics['equity_curve_r_squared'] = r_value ** 2
                metrics['equity_curve_stability'] = r_value ** 2  # Alias
                
        # System Quality Number (SQN)
        if trades and 'expectancy' in metrics:
            trade_pnls = [t.get('pnl', 0) for t in trades if 'pnl' in t]
            if len(trade_pnls) > 1:
                trade_std = np.std(trade_pnls)
                if trade_std > 0:
                    metrics['sqn'] = (metrics['expectancy'] / trade_std) * np.sqrt(len(trade_pnls))
                    
        return metrics
        
    def _calculate_max_drawdown_duration(self, drawdown_series: pd.Series) -> int:
        """Calculate maximum drawdown duration in days"""
        
        if drawdown_series.empty:
            return 0
            
        # Find periods where drawdown < 0
        in_drawdown = drawdown_series < 0
        
        # Find drawdown periods
        drawdown_periods = []
        start_idx = None
        
        for i, is_dd in enumerate(in_drawdown):
            if is_dd and start_idx is None:
                start_idx = i
            elif not is_dd and start_idx is not None:
                # Calculate duration
                start_date = drawdown_series.index[start_idx]
                end_date = drawdown_series.index[i-1]
                duration = (end_date - start_date).days
                drawdown_periods.append(duration)
                start_idx = None
                
        # Handle case where we're still in drawdown
        if start_idx is not None:
            end_date = drawdown_series.index[-1]
            start_date = drawdown_series.index[start_idx]
            duration = (end_date - start_date).days
            drawdown_periods.append(duration)
            
        return max(drawdown_periods) if drawdown_periods else 0
        
    def _calculate_ulcer_index(self, values: pd.Series, period: int = 14) -> float:
        """Calculate Ulcer Index"""
        
        if len(values) < period:
            return 0
            
        # Calculate percentage drawdown
        rolling_max = values.rolling(period).max()
        drawdown = 100 * (values - rolling_max) / rolling_max
        
        # Square the drawdowns
        squared_drawdowns = drawdown ** 2
        
        # Calculate mean of squared drawdowns
        mean_squared = squared_drawdowns.rolling(period).mean()
        
        # Take square root
        ulcer_index = np.sqrt(mean_squared.iloc[-1])
        
        return ulcer_index
        
    def _calculate_consecutive_stats(self, trades: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate consecutive win/loss statistics"""
        
        stats = {
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'current_consecutive_wins': 0,
            'current_consecutive_losses': 0
        }
        
        if not trades:
            return stats
            
        consecutive_wins = 0
        consecutive_losses = 0
        
        for trade in trades:
            if 'pnl' not in trade:
                continue
                
            if trade['pnl'] > 0:
                consecutive_wins += 1
                consecutive_losses = 0
                stats['max_consecutive_wins'] = max(stats['max_consecutive_wins'], 
                                                  consecutive_wins)
            elif trade['pnl'] < 0:
                consecutive_losses += 1
                consecutive_wins = 0
                stats['max_consecutive_losses'] = max(stats['max_consecutive_losses'], 
                                                     consecutive_losses)
                                                     
        stats['current_consecutive_wins'] = consecutive_wins
        stats['current_consecutive_losses'] = consecutive_losses
        
        return stats
        
    def _calculate_monthly_stats(self, equity_curve: pd.DataFrame) -> Dict[str, float]:
        """Calculate monthly return statistics"""
        
        stats = {}
        
        if equity_curve.empty or 'portfolio_value' not in equity_curve.columns:
            return stats
            
        # Resample to monthly
        monthly = equity_curve['portfolio_value'].resample('M').last()
        monthly_returns = monthly.pct_change().dropna()
        
        if len(monthly_returns) > 0:
            stats['avg_monthly_return'] = monthly_returns.mean() * 100
            stats['monthly_volatility'] = monthly_returns.std() * 100
            stats['best_month'] = monthly_returns.max() * 100
            stats['worst_month'] = monthly_returns.min() * 100
            stats['positive_months'] = (monthly_returns > 0).sum()
            stats['negative_months'] = (monthly_returns < 0).sum()
            stats['monthly_win_rate'] = (stats['positive_months'] / 
                                       len(monthly_returns) * 100)
            
        return stats
        
    def calculate_rolling_metrics(self, 
                                equity_curve: pd.DataFrame,
                                window: int = 30) -> pd.DataFrame:
        """Calculate rolling performance metrics"""
        
        if equity_curve.empty or 'portfolio_value' not in equity_curve.columns:
            return pd.DataFrame()
            
        # Calculate rolling returns
        rolling_returns = equity_curve['portfolio_value'].pct_change()
        
        # Create results dataframe
        results = pd.DataFrame(index=equity_curve.index)
        
        # Rolling return
        results['rolling_return'] = rolling_returns.rolling(window).mean() * 100
        
        # Rolling volatility
        results['rolling_volatility'] = rolling_returns.rolling(window).std() * 100
        
        # Rolling Sharpe
        daily_rf = self.risk_free_rate / self.trading_days_per_year
        excess_returns = rolling_returns - daily_rf
        
        rolling_mean = excess_returns.rolling(window).mean()
        rolling_std = rolling_returns.rolling(window).std()
        
        results['rolling_sharpe'] = (np.sqrt(self.trading_days_per_year) * 
                                    rolling_mean / rolling_std)
        
        # Rolling max drawdown
        for i in range(window, len(equity_curve)):
            window_data = equity_curve['portfolio_value'].iloc[i-window:i]
            cummax = window_data.cummax()
            drawdown = (window_data - cummax) / cummax
            results.loc[equity_curve.index[i], 'rolling_max_drawdown'] = drawdown.min() * 100
            
        return results