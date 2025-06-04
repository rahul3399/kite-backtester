# app/reporting/metrics_calculator.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
from scipy import stats
from collections import defaultdict

logger = logging.getLogger(__name__)

class MetricsCalculator:
    """
    Comprehensive performance metrics calculator for trading strategies
    Calculates various risk and return metrics from trade and portfolio data
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize metrics calculator
        
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = 252
        self.trading_minutes_per_day = 390  # 6.5 hours
        
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
        
        # Time-based metrics
        metrics.update(self._calculate_time_metrics(trades, equity_curve))
        
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
        metrics['total_return_pct'] = total_return * 100
        
        # Annualized return
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        if days > 0:
            years = days / 365
            annualized_return = (final_value / initial_capital) ** (1/years) - 1
            metrics['annualized_return'] = annualized_return * 100
            metrics['cagr'] = annualized_return * 100  # Compound Annual Growth Rate
        else:
            metrics['annualized_return'] = 0
            metrics['cagr'] = 0
            
        # Calculate daily returns
        if 'portfolio_value' in equity_curve.columns:
            daily_returns = equity_curve['portfolio_value'].pct_change().dropna()
            
            if len(daily_returns) > 0:
                metrics['avg_daily_return'] = daily_returns.mean() * 100
                metrics['daily_volatility'] = daily_returns.std() * 100
                metrics['annual_volatility'] = daily_returns.std() * np.sqrt(self.trading_days_per_year) * 100
                
                # Skewness and Kurtosis
                metrics['skewness'] = float(stats.skew(daily_returns))
                metrics['kurtosis'] = float(stats.kurtosis(daily_returns))
            
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
            metrics['sharpe_ratio'] = float(sharpe_ratio)
        else:
            metrics['sharpe_ratio'] = 0
            
        # Sortino ratio
        downside_returns = daily_returns[daily_returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino_ratio = np.sqrt(self.trading_days_per_year) * daily_returns.mean() / downside_returns.std()
            metrics['sortino_ratio'] = float(sortino_ratio)
        else:
            metrics['sortino_ratio'] = float('inf') if daily_returns.mean() > 0 else 0
            
        # Maximum drawdown
        cumulative = (1 + daily_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        
        metrics['max_drawdown'] = float(drawdown.min() * 100)
        metrics['max_drawdown_pct'] = float(drawdown.min() * 100)
        
        # Drawdown duration
        metrics['max_drawdown_duration'] = self._calculate_max_drawdown_duration(drawdown)
        
        # Average drawdown
        drawdown_periods = self._get_drawdown_periods(drawdown)
        if drawdown_periods:
            avg_drawdown = np.mean([period['drawdown'] for period in drawdown_periods])
            metrics['avg_drawdown'] = float(avg_drawdown * 100)
            metrics['drawdown_frequency'] = len(drawdown_periods)
        else:
            metrics['avg_drawdown'] = 0
            metrics['drawdown_frequency'] = 0
            
        # Calmar ratio
        if metrics['max_drawdown'] != 0 and 'annualized_return' in metrics:
            metrics['calmar_ratio'] = abs(metrics['annualized_return'] / metrics['max_drawdown'])
        else:
            metrics['calmar_ratio'] = 0
            
        # Value at Risk (VaR)
        metrics['var_95'] = float(np.percentile(daily_returns, 5) * 100)  # 95% VaR
        metrics['var_99'] = float(np.percentile(daily_returns, 1) * 100)  # 99% VaR
        
        # Conditional Value at Risk (CVaR)
        var_95_threshold = np.percentile(daily_returns, 5)
        cvar_returns = daily_returns[daily_returns <= var_95_threshold]
        metrics['cvar_95'] = float(cvar_returns.mean() * 100) if len(cvar_returns) > 0 else 0
        
        # Ulcer Index
        metrics['ulcer_index'] = self._calculate_ulcer_index(values)
        
        # Recovery time
        metrics['avg_recovery_time'] = self._calculate_avg_recovery_time(drawdown_periods)
        
        return metrics
        
    def _calculate_trade_metrics(self, trades: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate trade-specific metrics"""
        
        metrics = {}
        
        if not trades:
            metrics['total_trades'] = 0
            return metrics
            
        # Filter completed trades with PnL
        completed_trades = [t for t in trades if 'pnl' in t]
        
        if not completed_trades:
            metrics['total_trades'] = len(trades)
            metrics['completed_trades'] = 0
            return metrics
            
        # Basic trade stats
        metrics['total_trades'] = len(completed_trades)
        metrics['completed_trades'] = len(completed_trades)
        
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
        metrics['win_rate_pct'] = metrics['win_rate']
        
        # PnL statistics
        all_pnls = [t['pnl'] for t in completed_trades]
        metrics['total_pnl'] = sum(all_pnls)
        metrics['avg_pnl'] = np.mean(all_pnls) if all_pnls else 0
        
        # Average win/loss
        if winning_trades:
            win_pnls = [t['pnl'] for t in winning_trades]
            metrics['avg_win'] = np.mean(win_pnls)
            metrics['largest_win'] = max(win_pnls)
            metrics['median_win'] = np.median(win_pnls)
        else:
            metrics['avg_win'] = 0
            metrics['largest_win'] = 0
            metrics['median_win'] = 0
            
        if losing_trades:
            loss_pnls = [t['pnl'] for t in losing_trades]
            metrics['avg_loss'] = np.mean(loss_pnls)
            metrics['largest_loss'] = min(loss_pnls)
            metrics['median_loss'] = np.median(loss_pnls)
        else:
            metrics['avg_loss'] = 0
            metrics['largest_loss'] = 0
            metrics['median_loss'] = 0
            
        # Profit factor
        if losing_trades and metrics['avg_loss'] != 0:
            gross_profit = sum([t['pnl'] for t in winning_trades]) if winning_trades else 0
            gross_loss = abs(sum([t['pnl'] for t in losing_trades]))
            metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        else:
            metrics['profit_factor'] = float('inf') if winning_trades else 0
            
        # Expectancy
        metrics['expectancy'] = metrics['avg_pnl']
        
        # Expectancy ratio
        if metrics['avg_loss'] != 0:
            metrics['expectancy_ratio'] = abs(metrics['expectancy'] / metrics['avg_loss'])
        else:
            metrics['expectancy_ratio'] = float('inf') if metrics['expectancy'] > 0 else 0
            
        # Payoff ratio (Risk/Reward)
        if metrics['avg_loss'] != 0:
            metrics['payoff_ratio'] = abs(metrics['avg_win'] / metrics['avg_loss'])
            metrics['risk_reward_ratio'] = metrics['payoff_ratio']
        else:
            metrics['payoff_ratio'] = float('inf') if metrics['avg_win'] > 0 else 0
            metrics['risk_reward_ratio'] = metrics['payoff_ratio']
            
        # Kelly Criterion
        if metrics['win_rate'] > 0 and metrics['payoff_ratio'] > 0:
            win_rate_decimal = metrics['win_rate'] / 100
            metrics['kelly_criterion'] = (win_rate_decimal * metrics['payoff_ratio'] - 
                                        (1 - win_rate_decimal)) / metrics['payoff_ratio']
            metrics['kelly_percentage'] = max(0, metrics['kelly_criterion'] * 100)
        else:
            metrics['kelly_criterion'] = 0
            metrics['kelly_percentage'] = 0
            
        # Trade duration analysis
        durations = self._calculate_trade_durations(completed_trades)
        if durations:
            metrics['avg_trade_duration_hours'] = np.mean(durations)
            metrics['median_trade_duration_hours'] = np.median(durations)
            metrics['min_trade_duration_hours'] = min(durations)
            metrics['max_trade_duration_hours'] = max(durations)
            
        # Commission and slippage analysis
        total_commission = sum(t.get('commission', 0) for t in trades)
        total_slippage = sum(t.get('slippage', 0) for t in trades)
        metrics['total_commission'] = total_commission
        metrics['total_slippage'] = total_slippage
        metrics['avg_commission_per_trade'] = total_commission / len(trades) if trades else 0
        
        # Consecutive wins/losses
        consecutive_stats = self._calculate_consecutive_stats(completed_trades)
        metrics.update(consecutive_stats)
        
        return metrics
        
    def _calculate_advanced_metrics(self, 
                                  trades: List[Dict[str, Any]], 
                                  equity_curve: pd.DataFrame) -> Dict[str, float]:
        """Calculate advanced performance metrics"""
        
        metrics = {}
        
        # Recovery factor
        if 'max_drawdown' in metrics and metrics['max_drawdown'] != 0:
            if 'total_return' in metrics:
                metrics['recovery_factor'] = abs(metrics['total_return'] / metrics['max_drawdown'])
        
        # System Quality Number (SQN)
        if trades:
            trade_pnls = [t.get('pnl', 0) for t in trades if 'pnl' in t]
            if len(trade_pnls) > 1:
                avg_pnl = np.mean(trade_pnls)
                std_pnl = np.std(trade_pnls)
                if std_pnl > 0:
                    metrics['sqn'] = (avg_pnl / std_pnl) * np.sqrt(len(trade_pnls))
                    metrics['system_quality_number'] = metrics['sqn']
                    
        # Omega ratio
        if not equity_curve.empty and 'portfolio_value' in equity_curve.columns:
            daily_returns = equity_curve['portfolio_value'].pct_change().dropna()
            if len(daily_returns) > 0:
                threshold = 0  # Can be adjusted
                gains = daily_returns[daily_returns > threshold] - threshold
                losses = threshold - daily_returns[daily_returns <= threshold]
                
                if len(losses) > 0 and losses.sum() > 0:
                    metrics['omega_ratio'] = float(gains.sum() / losses.sum())
                else:
                    metrics['omega_ratio'] = float('inf') if len(gains) > 0 else 0
                    
        # Stability of returns (R-squared)
        if not equity_curve.empty and len(equity_curve) > 10:
            try:
                x = np.arange(len(equity_curve))
                y = equity_curve['portfolio_value'].values
                
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                metrics['equity_curve_r_squared'] = float(r_value ** 2)
                metrics['equity_curve_stability'] = float(r_value ** 2)
                metrics['equity_curve_slope'] = float(slope)
            except:
                pass
                
        # Tail ratio
        if not equity_curve.empty and 'portfolio_value' in equity_curve.columns:
            returns = equity_curve['portfolio_value'].pct_change().dropna()
            if len(returns) > 20:
                # 95th percentile / abs(5th percentile)
                right_tail = np.percentile(returns, 95)
                left_tail = abs(np.percentile(returns, 5))
                if left_tail > 0:
                    metrics['tail_ratio'] = float(right_tail / left_tail)
                    
        # Probabilistic metrics
        metrics.update(self._calculate_probabilistic_metrics(trades, equity_curve))
        
        return metrics
        
    def _calculate_time_metrics(self, 
                              trades: List[Dict[str, Any]], 
                              equity_curve: pd.DataFrame) -> Dict[str, float]:
        """Calculate time-based performance metrics"""
        
        metrics = {}
        
        if not equity_curve.empty:
            # Trading days
            total_days = (equity_curve.index[-1] - equity_curve.index[0]).days
            metrics['total_trading_days'] = total_days
            
            # Active trading days (days with trades)
            if trades:
                trade_dates = set()
                for trade in trades:
                    if 'timestamp' in trade:
                        trade_dates.add(pd.to_datetime(trade['timestamp']).date())
                metrics['active_trading_days'] = len(trade_dates)
                metrics['trading_frequency'] = len(trade_dates) / max(1, total_days)
            
            # Monthly returns analysis
            monthly_stats = self._calculate_monthly_stats(equity_curve)
            metrics.update(monthly_stats)
            
            # Best/worst periods
            if 'portfolio_value' in equity_curve.columns:
                daily_returns = equity_curve['portfolio_value'].pct_change().dropna()
                if len(daily_returns) > 0:
                    metrics['best_day_return'] = float(daily_returns.max() * 100)
                    metrics['worst_day_return'] = float(daily_returns.min() * 100)
                    
                    # Rolling performance
                    if len(daily_returns) > 30:
                        rolling_30d_returns = daily_returns.rolling(30).mean() * 30
                        metrics['best_30d_return'] = float(rolling_30d_returns.max() * 100)
                        metrics['worst_30d_return'] = float(rolling_30d_returns.min() * 100)
        
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
        
    def _get_drawdown_periods(self, drawdown_series: pd.Series) -> List[Dict[str, Any]]:
        """Extract all drawdown periods"""
        
        periods = []
        if drawdown_series.empty:
            return periods
            
        in_drawdown = drawdown_series < 0
        start_idx = None
        
        for i, is_dd in enumerate(in_drawdown):
            if is_dd and start_idx is None:
                start_idx = i
            elif not is_dd and start_idx is not None:
                # Record period
                period_drawdown = drawdown_series.iloc[start_idx:i]
                periods.append({
                    'start': drawdown_series.index[start_idx],
                    'end': drawdown_series.index[i-1],
                    'duration_days': (drawdown_series.index[i-1] - drawdown_series.index[start_idx]).days,
                    'drawdown': float(period_drawdown.min()),
                    'max_drawdown_idx': period_drawdown.idxmin()
                })
                start_idx = None
                
        # Handle ongoing drawdown
        if start_idx is not None:
            period_drawdown = drawdown_series.iloc[start_idx:]
            periods.append({
                'start': drawdown_series.index[start_idx],
                'end': drawdown_series.index[-1],
                'duration_days': (drawdown_series.index[-1] - drawdown_series.index[start_idx]).days,
                'drawdown': float(period_drawdown.min()),
                'max_drawdown_idx': period_drawdown.idxmin(),
                'ongoing': True
            })
            
        return periods
        
    def _calculate_ulcer_index(self, values: pd.Series, period: int = 14) -> float:
        """Calculate Ulcer Index"""
        
        if len(values) < period:
            return 0
            
        # Calculate percentage drawdown from rolling max
        rolling_max = values.rolling(period).max()
        drawdown = 100 * (values - rolling_max) / rolling_max
        
        # Square the drawdowns
        squared_drawdowns = drawdown ** 2
        
        # Calculate mean of squared drawdowns
        mean_squared = squared_drawdowns.rolling(period).mean()
        
        # Take square root
        ulcer_index = float(np.sqrt(mean_squared.iloc[-1]))
        
        return ulcer_index
        
    def _calculate_avg_recovery_time(self, drawdown_periods: List[Dict[str, Any]]) -> float:
        """Calculate average recovery time from drawdowns"""
        
        if not drawdown_periods:
            return 0
            
        # Only consider recovered drawdowns
        recovered_periods = [p for p in drawdown_periods if not p.get('ongoing', False)]
        
        if not recovered_periods:
            return 0
            
        recovery_times = [p['duration_days'] for p in recovered_periods]
        return float(np.mean(recovery_times))
        
    def _calculate_trade_durations(self, trades: List[Dict[str, Any]]) -> List[float]:
        """Calculate trade durations in hours"""
        
        durations = []
        
        for trade in trades:
            if 'timestamp' in trade and 'exit_time' in trade and trade['exit_time']:
                entry = pd.to_datetime(trade['timestamp'])
                exit = pd.to_datetime(trade['exit_time'])
                duration_hours = (exit - entry).total_seconds() / 3600
                durations.append(duration_hours)
            elif 'entry_time' in trade and 'exit_time' in trade and trade['exit_time']:
                entry = pd.to_datetime(trade['entry_time'])
                exit = pd.to_datetime(trade['exit_time'])
                duration_hours = (exit - entry).total_seconds() / 3600
                durations.append(duration_hours)
                
        return durations
        
    def _calculate_consecutive_stats(self, trades: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate consecutive win/loss statistics"""
        
        stats = {
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'current_consecutive_wins': 0,
            'current_consecutive_losses': 0,
            'avg_consecutive_wins': 0,
            'avg_consecutive_losses': 0
        }
        
        if not trades:
            return stats
            
        consecutive_wins = 0
        consecutive_losses = 0
        win_streaks = []
        loss_streaks = []
        
        for trade in trades:
            if 'pnl' not in trade:
                continue
                
            if trade['pnl'] > 0:
                consecutive_wins += 1
                if consecutive_losses > 0:
                    loss_streaks.append(consecutive_losses)
                consecutive_losses = 0
            elif trade['pnl'] < 0:
                consecutive_losses += 1
                if consecutive_wins > 0:
                    win_streaks.append(consecutive_wins)
                consecutive_wins = 0
                
        # Add final streaks
        if consecutive_wins > 0:
            win_streaks.append(consecutive_wins)
        if consecutive_losses > 0:
            loss_streaks.append(consecutive_losses)
            
        # Calculate statistics
        stats['max_consecutive_wins'] = max(win_streaks) if win_streaks else 0
        stats['max_consecutive_losses'] = max(loss_streaks) if loss_streaks else 0
        stats['current_consecutive_wins'] = consecutive_wins
        stats['current_consecutive_losses'] = consecutive_losses
        stats['avg_consecutive_wins'] = float(np.mean(win_streaks)) if win_streaks else 0
        stats['avg_consecutive_losses'] = float(np.mean(loss_streaks)) if loss_streaks else 0
        
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
            stats['avg_monthly_return'] = float(monthly_returns.mean() * 100)
            stats['monthly_volatility'] = float(monthly_returns.std() * 100)
            stats['best_month'] = float(monthly_returns.max() * 100)
            stats['worst_month'] = float(monthly_returns.min() * 100)
            stats['positive_months'] = int((monthly_returns > 0).sum())
            stats['negative_months'] = int((monthly_returns < 0).sum())
            stats['monthly_win_rate'] = float(stats['positive_months'] / len(monthly_returns) * 100)
            
            # Monthly Sharpe
            if monthly_returns.std() > 0:
                monthly_rf = self.risk_free_rate / 12
                monthly_sharpe = (monthly_returns.mean() - monthly_rf) / monthly_returns.std() * np.sqrt(12)
                stats['monthly_sharpe_ratio'] = float(monthly_sharpe)
            
        return stats
        
    def _calculate_probabilistic_metrics(self, 
                                       trades: List[Dict[str, Any]], 
                                       equity_curve: pd.DataFrame) -> Dict[str, float]:
        """Calculate probability-based metrics"""
        
        metrics = {}
        
        if not trades:
            return metrics
            
        # Monte Carlo metrics
        trade_pnls = [t.get('pnl', 0) for t in trades if 'pnl' in t]
        
        if len(trade_pnls) > 10:
            # Probability of profit
            metrics['probability_of_profit'] = float((np.array(trade_pnls) > 0).mean() * 100)
            
            # Expected shortfall (CVaR alternative)
            sorted_pnls = sorted(trade_pnls)
            worst_5_pct = int(len(sorted_pnls) * 0.05)
            if worst_5_pct > 0:
                metrics['expected_shortfall'] = float(np.mean(sorted_pnls[:worst_5_pct]))
            
            # Risk of ruin estimation (simplified)
            if len(trade_pnls) > 30:
                avg_loss = np.mean([p for p in trade_pnls if p < 0]) if any(p < 0 for p in trade_pnls) else 0
                if avg_loss < 0 and 'win_rate' in metrics:
                    win_rate = metrics.get('win_rate', 50) / 100
                    # Simplified risk of ruin formula
                    if win_rate < 1 and win_rate > 0:
                        risk_of_ruin = ((1 - win_rate) / win_rate) ** (100 / abs(avg_loss))
                        metrics['risk_of_ruin_pct'] = float(min(risk_of_ruin * 100, 100))
                        
        return metrics
        
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
        results['rolling_return'] = rolling_returns.rolling(window).sum() * 100
        
        # Rolling volatility
        results['rolling_volatility'] = rolling_returns.rolling(window).std() * 100 * np.sqrt(252)
        
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
        
    def generate_metrics_summary(self, metrics: Dict[str, float]) -> str:
        """Generate a formatted summary of metrics"""
        
        summary = []
        summary.append("=== Performance Metrics Summary ===\n")
        
        # Returns
        summary.append("Returns:")
        summary.append(f"  Total Return: {metrics.get('total_return', 0):.2f}%")
        summary.append(f"  Annualized Return: {metrics.get('annualized_return', 0):.2f}%")
        summary.append(f"  Average Daily Return: {metrics.get('avg_daily_return', 0):.3f}%")
        
        # Risk
        summary.append("\nRisk Metrics:")
        summary.append(f"  Annual Volatility: {metrics.get('annual_volatility', 0):.2f}%")
        summary.append(f"  Maximum Drawdown: {metrics.get('max_drawdown', 0):.2f}%")
        summary.append(f"  Value at Risk (95%): {metrics.get('var_95', 0):.2f}%")
        
        # Risk-Adjusted Returns
        summary.append("\nRisk-Adjusted Returns:")
        summary.append(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
        summary.append(f"  Sortino Ratio: {metrics.get('sortino_ratio', 0):.3f}")
        summary.append(f"  Calmar Ratio: {metrics.get('calmar_ratio', 0):.3f}")
        
        # Trading Statistics
        summary.append("\nTrading Statistics:")
        summary.append(f"  Total Trades: {metrics.get('total_trades', 0)}")
        summary.append(f"  Win Rate: {metrics.get('win_rate', 0):.1f}%")
        summary.append(f"  Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        summary.append(f"  Average Win: ${metrics.get('avg_win', 0):.2f}")
        summary.append(f"  Average Loss: ${metrics.get('avg_loss', 0):.2f}")
        summary.append(f"  Expectancy: ${metrics.get('expectancy', 0):.2f}")
        
        # Advanced Metrics
        summary.append("\nAdvanced Metrics:")
        summary.append(f"  System Quality Number: {metrics.get('sqn', 0):.2f}")
        summary.append(f"  Kelly Criterion: {metrics.get('kelly_percentage', 0):.1f}%")
        summary.append(f"  Omega Ratio: {metrics.get('omega_ratio', 0):.2f}")
        
        return "\n".join(summary)