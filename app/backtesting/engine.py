# app/backtesting/engine.py
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import logging
from dataclasses import dataclass, field

from ..strategies.base_strategy import BaseStrategy, Signal, OrderSide, OrderType
from ..core.data_manager import DataManager
from .broker_simulator import BrokerSimulator
from .data_feed import DataFeed
from ..reporting.metrics_calculator import MetricsCalculator

logger = logging.getLogger(__name__)

@dataclass
class BacktestResult:
    """Container for backtest results"""
    trades: List[Dict[str, Any]] = field(default_factory=list)
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    metrics: Dict[str, float] = field(default_factory=dict)
    positions: Dict[str, Any] = field(default_factory=dict)
    orders: List[Dict[str, Any]] = field(default_factory=list)
    strategy_name: str = ""
    symbols: List[str] = field(default_factory=list)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    initial_capital: float = 0
    final_capital: float = 0
    execution_time: float = 0
    total_bars_processed: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

class BacktestingEngine:
    """High-performance backtesting engine with support for multiple strategies and symbols"""
    
    def __init__(self, data_manager: DataManager, initial_capital: float = 1000000):
        self.data_manager = data_manager
        self.initial_capital = initial_capital
        self.broker_simulator = BrokerSimulator(initial_capital)
        self.data_feed = DataFeed(data_manager)
        self.metrics_calculator = MetricsCalculator()
        self.max_workers = 4  # For parallel processing
        
    async def run_backtest(self,
                          strategy: BaseStrategy,
                          start_date: datetime,
                          end_date: datetime,
                          symbols: Optional[List[str]] = None,
                          commission: float = 0.0002,
                          slippage: float = 0.0001,
                          data_frequency: str = "1T") -> BacktestResult:
        """
        Run backtest for a strategy
        
        Args:
            strategy: Strategy instance to backtest
            start_date: Start date for backtesting
            end_date: End date for backtesting
            symbols: List of symbols to trade (uses strategy symbols if None)
            commission: Commission rate (default 0.02%)
            slippage: Slippage rate (default 0.01%)
            data_frequency: Data frequency for resampling
            
        Returns:
            BacktestResult object with complete results
        """
        
        start_time = datetime.now()
        
        result = BacktestResult()
        result.strategy_name = strategy.name
        result.initial_capital = self.initial_capital
        result.start_date = start_date
        result.end_date = end_date
        
        # Use strategy symbols if not provided
        if symbols is None:
            symbols = strategy.symbols
        result.symbols = symbols
        
        # Reset broker
        self.broker_simulator.reset(self.initial_capital)
        self.broker_simulator.set_commission(commission)
        self.broker_simulator.set_slippage(slippage)
        
        # Initialize strategy
        strategy.initialize()
        
        # Get instrument tokens for symbols
        instrument_tokens = await self._get_instrument_tokens(symbols)
        
        # Load historical data for all symbols
        logger.info(f"Loading historical data for {len(symbols)} symbols...")
        data_dict = await self.data_feed.load_data(
            instrument_tokens, 
            strategy.config.timeframe,
            start_date, 
            end_date,
            strategy.config.lookback_periods
        )
        
        if not data_dict:
            raise ValueError("No data loaded for backtesting")
        
        # Run backtest
        logger.info("Running backtest...")
        bars_processed = await self._run_backtest_loop(strategy, data_dict, result)
        
        # Calculate final metrics
        result.final_capital = self.broker_simulator.get_portfolio_value()
        result.trades = self.broker_simulator.get_trades()
        result.equity_curve = self.broker_simulator.get_equity_curve()
        result.positions = self.broker_simulator.get_positions()
        result.orders = self.broker_simulator.get_orders()
        
        # Calculate performance metrics
        result.metrics = self.metrics_calculator.calculate_metrics(
            result.trades,
            result.equity_curve,
            result.initial_capital
        )
        
        # Add execution metadata
        result.execution_time = (datetime.now() - start_time).total_seconds()
        result.total_bars_processed = bars_processed
        result.metadata = {
            "commission": commission,
            "slippage": slippage,
            "data_frequency": data_frequency,
            "strategy_parameters": strategy.config.parameters
        }
        
        logger.info(f"Backtest completed in {result.execution_time:.2f} seconds")
        logger.info(f"Processed {bars_processed} bars, generated {len(result.trades)} trades")
        
        return result
    
    async def run_multiple_backtests(self,
                                   strategies: List[BaseStrategy],
                                   start_date: datetime,
                                   end_date: datetime,
                                   symbols: Optional[List[str]] = None,
                                   parallel: bool = True) -> List[BacktestResult]:
        """
        Run multiple backtests
        
        Args:
            strategies: List of strategy instances
            start_date: Start date
            end_date: End date
            symbols: Symbols to trade
            parallel: Run backtests in parallel
            
        Returns:
            List of BacktestResult objects
        """
        
        if parallel and len(strategies) > 1:
            # Run backtests in parallel
            tasks = []
            for strategy in strategies:
                task = self.run_backtest(strategy, start_date, end_date, symbols)
                tasks.append(task)
                
            results = await asyncio.gather(*tasks)
        else:
            # Run backtests sequentially
            results = []
            for strategy in strategies:
                result = await self.run_backtest(strategy, start_date, end_date, symbols)
                results.append(result)
                
        return results
    
    async def optimize_strategy(self,
                              strategy_class: type,
                              parameter_grid: Dict[str, List[Any]],
                              start_date: datetime,
                              end_date: datetime,
                              symbols: List[str],
                              optimization_metric: str = "sharpe_ratio",
                              n_jobs: int = -1) -> Dict[str, Any]:
        """
        Optimize strategy parameters
        
        Args:
            strategy_class: Strategy class to optimize
            parameter_grid: Parameter grid for optimization
            start_date: Start date
            end_date: End date
            symbols: Symbols to trade
            optimization_metric: Metric to optimize
            n_jobs: Number of parallel jobs (-1 for all cores)
            
        Returns:
            Optimization results with best parameters
        """
        
        # Generate parameter combinations
        param_combinations = self._generate_parameter_combinations(parameter_grid)
        
        logger.info(f"Testing {len(param_combinations)} parameter combinations...")
        
        # Determine number of workers
        if n_jobs == -1:
            n_jobs = self.max_workers
        
        best_result = None
        best_metric = -float('inf')
        all_results = []
        
        # Run backtests with different parameters
        batch_size = max(1, len(param_combinations) // n_jobs)
        
        for i in range(0, len(param_combinations), batch_size):
            batch = param_combinations[i:i + batch_size]
            batch_results = await self._run_optimization_batch(
                strategy_class, batch, start_date, end_date, symbols
            )
            
            for params, result in zip(batch, batch_results):
                if result:
                    metric_value = result.metrics.get(optimization_metric, -float('inf'))
                    
                    all_results.append({
                        "parameters": params,
                        "metrics": result.metrics,
                        "metric_value": metric_value,
                        "total_trades": len(result.trades),
                        "final_capital": result.final_capital
                    })
                    
                    if metric_value > best_metric:
                        best_metric = metric_value
                        best_result = {
                            "parameters": params,
                            "result": result,
                            "metric_value": metric_value
                        }
                        
                    logger.info(f"Tested parameters {params}: {optimization_metric}={metric_value:.4f}")
        
        # Sort results by optimization metric
        all_results.sort(key=lambda x: x["metric_value"], reverse=True)
        
        return {
            "best": best_result,
            "all_results": all_results,
            "optimization_metric": optimization_metric,
            "total_combinations_tested": len(param_combinations),
            "parameter_importance": self._analyze_parameter_importance(all_results, parameter_grid)
        }
    
    async def _run_backtest_loop(self, 
                               strategy: BaseStrategy,
                               data_dict: Dict[str, pd.DataFrame],
                               result: BacktestResult) -> int:
        """Main backtest loop"""
        
        # Find common timestamp range
        all_timestamps = []
        for df in data_dict.values():
            if not df.empty:
                all_timestamps.extend(df.index.tolist())
        
        if not all_timestamps:
            logger.warning("No data available for backtesting")
            return 0
        
        unique_timestamps = sorted(set(all_timestamps))
        bars_processed = 0
        
        # Process each timestamp
        for timestamp in unique_timestamps:
            # Update broker with current timestamp
            self.broker_simulator.set_current_time(timestamp)
            
            # Process each symbol
            for symbol, df in data_dict.items():
                if timestamp in df.index:
                    # Get data up to current timestamp
                    current_data = df.loc[:timestamp]
                    
                    if current_data.empty:
                        continue
                    
                    # Update market prices
                    current_bar = current_data.iloc[-1]
                    self.broker_simulator.update_market_price(symbol, current_bar['close'])
                    
                    # Generate signal
                    try:
                        signal = strategy.on_data(symbol, current_data)
                        
                        if signal:
                            # Validate signal
                            is_valid, error = strategy.validate_signal(signal)
                            if not is_valid:
                                logger.warning(f"Invalid signal: {error}")
                                continue
                            
                            # Calculate position size
                            capital = self.broker_simulator.get_available_capital()
                            position_size = strategy.calculate_position_size(
                                symbol, signal, capital, current_bar['close']
                            )
                            
                            if position_size > 0:
                                signal.quantity = position_size
                                
                                # Execute order
                                order = self._create_order_from_signal(signal, timestamp)
                                fill = self.broker_simulator.execute_order(order)
                                
                                if fill:
                                    strategy.on_order_fill(fill)
                                    
                    except Exception as e:
                        logger.error(f"Error processing {symbol} at {timestamp}: {e}")
                        
            # Update positions
            positions = self.broker_simulator.get_positions()
            for position in positions.values():
                try:
                    strategy.on_position_update(position)
                except Exception as e:
                    logger.error(f"Error updating position: {e}")
            
            # Record equity
            self.broker_simulator.record_equity()
            bars_processed += 1
            
        return bars_processed
    
    async def _run_optimization_batch(self,
                                    strategy_class: type,
                                    param_batch: List[Dict[str, Any]],
                                    start_date: datetime,
                                    end_date: datetime,
                                    symbols: List[str]) -> List[Optional[BacktestResult]]:
        """Run a batch of optimization backtests"""
        
        results = []
        
        for params in param_batch:
            try:
                # Create strategy instance with parameters
                from ..strategies.base_strategy import StrategyConfig
                config = StrategyConfig(
                    name=f"{strategy_class.__name__}_opt",
                    symbols=symbols,
                    parameters=params
                )
                strategy = strategy_class(config)
                
                # Run backtest
                result = await self.run_backtest(
                    strategy, start_date, end_date, symbols
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error in optimization with params {params}: {e}")
                results.append(None)
                
        return results
    
    def _create_order_from_signal(self, signal: Signal, timestamp: datetime) -> Dict[str, Any]:
        """Create order from signal"""
        return {
            "symbol": signal.symbol,
            "side": signal.side.value,
            "quantity": signal.quantity,
            "order_type": signal.order_type.value,
            "price": signal.price,
            "stop_loss": signal.stop_loss,
            "take_profit": signal.take_profit,
            "timestamp": timestamp,
            "metadata": signal.metadata
        }
    
    async def _get_instrument_tokens(self, symbols: List[str]) -> Dict[str, int]:
        """Get instrument tokens for symbols"""
        # In production, this would map symbols to actual Kite instrument tokens
        # For now, using placeholder tokens
        tokens = {}
        for i, symbol in enumerate(symbols):
            tokens[symbol] = 1000 + i
        return tokens
    
    def _generate_parameter_combinations(self, 
                                       parameter_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Generate all parameter combinations from grid"""
        import itertools
        
        keys = list(parameter_grid.keys())
        values = [parameter_grid[key] for key in keys]
        
        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))
        
        return combinations
    
    def _analyze_parameter_importance(self,
                                    results: List[Dict[str, Any]],
                                    parameter_grid: Dict[str, List[Any]]) -> Dict[str, float]:
        """Analyze parameter importance based on results"""
        
        if not results:
            return {}
        
        importance = {}
        
        # Calculate correlation between each parameter and the metric
        for param_name in parameter_grid.keys():
            param_values = [r["parameters"][param_name] for r in results]
            metric_values = [r["metric_value"] for r in results]
            
            # Convert to numeric if possible
            try:
                param_values = [float(v) for v in param_values]
                
                # Calculate correlation
                if len(set(param_values)) > 1:  # Need variation
                    correlation = np.corrcoef(param_values, metric_values)[0, 1]
                    importance[param_name] = abs(correlation)
                else:
                    importance[param_name] = 0.0
                    
            except (ValueError, TypeError):
                # For non-numeric parameters, use variance analysis
                unique_values = list(set(param_values))
                if len(unique_values) > 1:
                    # Calculate mean metric for each unique value
                    value_metrics = {}
                    for val in unique_values:
                        val_results = [r["metric_value"] for r in results 
                                     if r["parameters"][param_name] == val]
                        value_metrics[val] = np.mean(val_results)
                    
                    # Use variance of means as importance
                    importance[param_name] = np.std(list(value_metrics.values()))
                else:
                    importance[param_name] = 0.0
        
        # Normalize importance scores
        total_importance = sum(importance.values())
        if total_importance > 0:
            importance = {k: v / total_importance for k, v in importance.items()}
        
        return importance