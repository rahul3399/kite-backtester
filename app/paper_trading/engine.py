# app/paper_trading/engine.py
import asyncio
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
import logging
from collections import defaultdict
import uuid
import pandas as pd
import numpy as np

from ..strategies.base_strategy import BaseStrategy, Signal, OrderSide, OrderType
from ..core.websocket_manager import WebSocketManager
from ..core.kite_client import KiteClient
from ..core.data_manager import DataManager
from ..core.order_manager import OrderManager, OrderStatus
from ..core.position_manager import PositionManager
from .virtual_broker import VirtualBroker
from .live_feed import LiveDataFeed
from .risk_manager import RiskManager
from ..reporting.metrics_calculator import MetricsCalculator
from ..database import crud, models
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

class PaperTradingEngine:
    """
    Real-time paper trading engine with WebSocket integration
    Manages multiple strategies running concurrently
    """
    
    def __init__(self, 
                 kite_client: KiteClient,
                 websocket_manager: WebSocketManager,
                 data_manager: DataManager,
                 db: Optional[Session] = None,
                 initial_capital: float = 1000000):
        
        self.kite_client = kite_client
        self.ws_manager = websocket_manager
        self.data_manager = data_manager
        self.db = db
        
        # Initialize components
        self.virtual_broker = VirtualBroker(initial_capital)
        self.order_manager = OrderManager(mode="paper")
        self.position_manager = PositionManager(initial_capital)
        self.live_feed = LiveDataFeed(websocket_manager, data_manager)
        self.risk_manager = RiskManager(initial_capital)
        self.metrics_calculator = MetricsCalculator()
        
        # Strategy management
        self.active_strategies: Dict[str, BaseStrategy] = {}
        self.strategy_tasks: Dict[str, asyncio.Task] = {}
        self.strategy_configs: Dict[str, Dict[str, Any]] = {}
        self.strategy_performance: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Engine state
        self.is_running = False
        self._main_task: Optional[asyncio.Task] = None
        self._monitoring_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self._start_time = datetime.now()
        self._tick_count = 0
        self._signal_count = 0
        self._order_count = 0
        
        # Data buffering for strategies
        self._strategy_data_buffer: Dict[str, Dict[str, pd.DataFrame]] = defaultdict(dict)
        self._buffer_size = 1000  # Keep last N bars per symbol
        
    async def start(self):
        """Start paper trading engine"""
        if self.is_running:
            logger.warning("Paper trading engine already running")
            return
            
        logger.info("Starting paper trading engine...")
        self.is_running = True
        self._start_time = datetime.now()
        
        # Connect WebSocket if not connected
        if not self.ws_manager.is_connected():
            await self.ws_manager.connect()
            
        # Register callbacks
        self.ws_manager.register_callback("tick", self._on_tick)
        self.ws_manager.register_callback("order_update", self._on_order_update)
        
        # Start main processing loop
        self._main_task = asyncio.create_task(self._main_loop())
        
        # Start monitoring task
        self._monitoring_task = asyncio.create_task(self._monitor_strategies())
        
        # Start risk monitoring
        asyncio.create_task(self.risk_manager.start_monitoring(
            self.position_manager,
            self.virtual_broker
        ))
        
        logger.info("Paper trading engine started successfully")
        
    async def stop(self):
        """Stop paper trading engine"""
        logger.info("Stopping paper trading engine...")
        self.is_running = False
        
        # Stop all strategies
        for strategy_id in list(self.active_strategies.keys()):
            await self.stop_strategy(strategy_id)
            
        # Cancel tasks
        if self._main_task:
            self._main_task.cancel()
        if self._monitoring_task:
            self._monitoring_task.cancel()
            
        # Unregister callbacks
        self.ws_manager.unregister_callback("tick", self._on_tick)
        self.ws_manager.unregister_callback("order_update", self._on_order_update)
        
        # Stop risk manager
        self.risk_manager.stop_monitoring()
        
        # Generate final report
        await self._generate_final_report()
        
        logger.info("Paper trading engine stopped")
        
    async def add_strategy(self, 
                          strategy: BaseStrategy,
                          capital_allocation: Optional[float] = None,
                          risk_per_trade: float = 0.02,
                          max_positions: int = 5) -> str:
        """
        Add and start a strategy
        
        Args:
            strategy: Strategy instance
            capital_allocation: Capital allocated to strategy
            risk_per_trade: Risk per trade as fraction
            max_positions: Maximum concurrent positions
            
        Returns:
            Strategy ID
        """
        
        strategy_id = str(uuid.uuid4())
        
        # Initialize strategy
        try:
            strategy.initialize()
            logger.info(f"Initialized strategy {strategy.name} ({strategy_id})")
        except Exception as e:
            logger.error(f"Failed to initialize strategy: {e}")
            raise
            
        # Store strategy config
        self.strategy_configs[strategy_id] = {
            'name': strategy.name,
            'symbols': strategy.symbols,
            'capital_allocation': capital_allocation,
            'risk_per_trade': risk_per_trade,
            'max_positions': max_positions,
            'started_at': datetime.now()
        }
        
        # Get instrument tokens for symbols
        instruments = await self._get_instruments_for_symbols(strategy.symbols)
        
        # Subscribe to market data
        tokens = [inst['instrument_token'] for inst in instruments]
        self.ws_manager.subscribe(tokens, "full")
        
        # Initialize data buffers for strategy
        for symbol in strategy.symbols:
            # Load initial historical data
            history = await self._load_initial_data(symbol, strategy.config.timeframe)
            if history is not None:
                self._strategy_data_buffer[strategy_id][symbol] = history
                
        # Store strategy
        self.active_strategies[strategy_id] = strategy
        
        # Start strategy task
        task = asyncio.create_task(self._run_strategy(strategy_id, strategy))
        self.strategy_tasks[strategy_id] = task
        
        # Initialize performance tracking
        self.strategy_performance[strategy_id] = {
            'total_signals': 0,
            'total_orders': 0,
            'total_trades': 0,
            'total_pnl': 0.0,
            'open_positions': 0
        }
        
        # Save to database if available
        if self.db:
            db_instance = crud.create_strategy_instance(
                self.db,
                strategy_id=1,  # Need to map strategy name to ID
                name=f"{strategy.name}_{strategy_id[:8]}",
                symbols=strategy.symbols,
                parameters=strategy.config.parameters,
                mode="paper",
                capital_allocation=capital_allocation,
                risk_per_trade=risk_per_trade,
                max_positions=max_positions
            )
            self.strategy_configs[strategy_id]['db_id'] = db_instance.id
            
        logger.info(f"Added strategy {strategy.name} with ID {strategy_id}")
        return strategy_id
        
    async def stop_strategy(self, strategy_id: str):
        """Stop and remove a strategy"""
        
        if strategy_id not in self.active_strategies:
            logger.warning(f"Strategy {strategy_id} not found")
            return
            
        logger.info(f"Stopping strategy {strategy_id}")
        
        # Cancel strategy task
        if strategy_id in self.strategy_tasks:
            self.strategy_tasks[strategy_id].cancel()
            try:
                await self.strategy_tasks[strategy_id]
            except asyncio.CancelledError:
                pass
            del self.strategy_tasks[strategy_id]
            
        # Get strategy
        strategy = self.active_strategies[strategy_id]
        
        # Close all positions for this strategy
        positions = self.position_manager.get_strategy_positions(strategy_id)
        for position in positions:
            symbol = position['symbol']
            current_price = self.virtual_broker.get_current_price(symbol)
            if current_price:
                await self._close_position(strategy_id, symbol, current_price)
                
        # Unsubscribe from symbols (if no other strategy uses them)
        await self._unsubscribe_unused_symbols(strategy.symbols)
        
        # Remove strategy
        del self.active_strategies[strategy_id]
        
        # Clean up buffers
        if strategy_id in self._strategy_data_buffer:
            del self._strategy_data_buffer[strategy_id]
            
        # Update database if available
        if self.db and 'db_id' in self.strategy_configs[strategy_id]:
            crud.stop_strategy_instance(
                self.db,
                self.strategy_configs[strategy_id]['db_id']
            )
            
        # Generate strategy report
        await self._generate_strategy_report(strategy_id)
        
        logger.info(f"Strategy {strategy_id} stopped")
        
    async def _main_loop(self):
        """Main processing loop"""
        
        logger.info("Paper trading main loop started")
        
        while self.is_running:
            try:
                # Process any pending orders
                await self._process_pending_orders()
                
                # Check stop loss / take profit
                await self._check_exit_conditions()
                
                # Update performance metrics
                self.position_manager.record_equity_snapshot()
                
                # Small delay to prevent CPU spinning
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                
        logger.info("Paper trading main loop stopped")
        
    async def _run_strategy(self, strategy_id: str, strategy: BaseStrategy):
        """Run strategy processing loop"""
        
        logger.info(f"Strategy {strategy_id} processing started")
        
        try:
            while self.is_running and strategy_id in self.active_strategies:
                # Process each symbol
                for symbol in strategy.symbols:
                    try:
                        # Get latest data
                        data = self._get_strategy_data(strategy_id, symbol)
                        if data is None or data.empty:
                            continue
                            
                        # Generate signal
                        signal = strategy.on_data(symbol, data)
                        
                        if signal:
                            self._signal_count += 1
                            self.strategy_performance[strategy_id]['total_signals'] += 1
                            
                            # Process signal
                            await self._process_signal(strategy_id, strategy, signal)
                            
                    except Exception as e:
                        logger.error(f"Error processing {symbol} for strategy {strategy_id}: {e}")
                        
                # Update positions
                positions = self.position_manager.get_strategy_positions(strategy_id)
                for position in positions:
                    try:
                        strategy.on_position_update(position)
                    except Exception as e:
                        logger.error(f"Error updating position: {e}")
                        
                # Small delay between iterations
                await asyncio.sleep(1)
                
        except asyncio.CancelledError:
            logger.info(f"Strategy {strategy_id} processing cancelled")
        except Exception as e:
            logger.error(f"Fatal error in strategy {strategy_id}: {e}")
            # Create alert
            if self.db:
                crud.create_alert(
                    self.db,
                    type="strategy_error",
                    severity="error",
                    title=f"Strategy {strategy.name} crashed",
                    message=str(e),
                    strategy_instance_id=self.strategy_configs[strategy_id].get('db_id')
                )
                
        logger.info(f"Strategy {strategy_id} processing stopped")
        
    async def _process_signal(self, 
                            strategy_id: str, 
                            strategy: BaseStrategy, 
                            signal: Signal):
        """Process trading signal from strategy"""
        
        # Validate signal
        is_valid, error = strategy.validate_signal(signal)
        if not is_valid:
            logger.warning(f"Invalid signal from {strategy_id}: {error}")
            return
            
        # Apply risk management
        risk_check = await self.risk_manager.check_signal(
            signal,
            self.position_manager,
            self.strategy_configs[strategy_id]
        )
        
        if not risk_check['approved']:
            logger.warning(f"Signal rejected by risk manager: {risk_check['reason']}")
            return
            
        # Calculate position size
        current_price = self.virtual_broker.get_current_price(signal.symbol)
        if not current_price:
            logger.warning(f"No price available for {signal.symbol}")
            return
            
        capital = self.strategy_configs[strategy_id].get('capital_allocation') or self.virtual_broker.get_available_capital()
        position_size = strategy.calculate_position_size(
            signal.symbol, signal, capital, current_price
        )
        
        if position_size <= 0:
            logger.warning(f"Invalid position size calculated: {position_size}")
            return
            
        signal.quantity = position_size
        
        # Place order
        success, order_id = await self.order_manager.place_order(
            symbol=signal.symbol,
            side=signal.side,
            quantity=signal.quantity,
            order_type=signal.order_type,
            price=signal.price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            strategy_id=strategy_id,
            metadata=signal.metadata
        )
        
        if success:
            self._order_count += 1
            self.strategy_performance[strategy_id]['total_orders'] += 1
            
            # Execute order immediately for market orders
            if signal.order_type == OrderType.MARKET:
                await self._execute_order(order_id, current_price)
                
            logger.info(f"Order placed: {order_id} for signal from {strategy_id}")
        else:
            logger.error(f"Failed to place order: {order_id}")
            
    async def _execute_order(self, order_id: str, execution_price: float):
        """Execute a paper trading order"""
        
        order = self.order_manager.get_order(order_id)
        if not order:
            return
            
        # Simulate order fill through virtual broker
        fill = self.virtual_broker.execute_order(order, execution_price)
        
        if fill:
            # Update order status
            self.order_manager.update_order_status(
                order_id,
                status=OrderStatus.COMPLETE,
                fill_data=fill
            )
            
            # Update position
            if order['side'] == 'BUY':
                self.position_manager.open_position(
                    symbol=order['symbol'],
                    quantity=order['quantity'],
                    price=fill['price'],
                    side="LONG",
                    strategy_id=order.get('strategy_id'),
                    commission=fill['commission']
                )
            else:
                self.position_manager.close_position(
                    symbol=order['symbol'],
                    quantity=order['quantity'],
                    price=fill['price'],
                    commission=fill['commission']
                )
                
            # Notify strategy
            strategy_id = order.get('strategy_id')
            if strategy_id and strategy_id in self.active_strategies:
                strategy = self.active_strategies[strategy_id]
                try:
                    strategy.on_order_fill(fill)
                except Exception as e:
                    logger.error(f"Error in strategy on_order_fill: {e}")
                    
            # Update performance
            if 'pnl' in fill:
                self.strategy_performance[strategy_id]['total_pnl'] += fill['pnl']
                self.strategy_performance[strategy_id]['total_trades'] += 1
                
            # Save to database
            if self.db:
                crud.create_trade(
                    self.db,
                    order_id=order['id'],
                    symbol=order['symbol'],
                    side=order['side'],
                    quantity=order['quantity'],
                    price=fill['price'],
                    strategy_instance_id=self.strategy_configs[strategy_id].get('db_id'),
                    pnl=fill.get('pnl', 0),
                    commission=fill['commission']
                )
                
    def _on_tick(self, ticks: List[Dict[str, Any]]):
        """Handle incoming market data ticks"""
        
        self._tick_count += len(ticks)
        
        # Update virtual broker prices
        for tick in ticks:
            symbol = self._get_symbol_from_tick(tick)
            if symbol:
                price = tick.get('last_price', 0)
                self.virtual_broker.update_market_price(symbol, price)
                
                # Update position manager prices
                self.position_manager.update_market_prices({symbol: price})
                
                # Update data buffers
                self._update_data_buffers(symbol, tick)
                
    def _on_order_update(self, order_update: Dict[str, Any]):
        """Handle order updates (for live trading integration)"""
        logger.info(f"Order update received: {order_update}")
        
    async def _process_pending_orders(self):
        """Process pending limit/stop orders"""
        
        pending_orders = self.order_manager.get_open_orders()
        
        for order in pending_orders:
            current_price = self.virtual_broker.get_current_price(order['symbol'])
            if not current_price:
                continue
                
            should_execute = False
            execution_price = current_price
            
            # Check limit orders
            if order['order_type'] == 'LIMIT' and order['price']:
                if order['side'] == 'BUY' and current_price <= order['price']:
                    should_execute = True
                    execution_price = order['price']
                elif order['side'] == 'SELL' and current_price >= order['price']:
                    should_execute = True
                    execution_price = order['price']
                    
            # Check stop orders
            elif order['order_type'] == 'STOP_LOSS' and order['trigger_price']:
                if order['side'] == 'BUY' and current_price >= order['trigger_price']:
                    should_execute = True
                elif order['side'] == 'SELL' and current_price <= order['trigger_price']:
                    should_execute = True
                    
            if should_execute:
                await self._execute_order(order['order_id'], execution_price)
                
    async def _check_exit_conditions(self):
        """Check stop loss and take profit conditions"""
        
        triggered = self.position_manager.check_stop_loss_take_profit()
        
        for trigger in triggered:
            symbol = trigger['symbol']
            position = self.position_manager.get_position(symbol)
            
            if position:
                # Place exit order
                await self.order_manager.place_order(
                    symbol=symbol,
                    side=OrderSide.SELL if position.side == "LONG" else OrderSide.BUY,
                    quantity=position.quantity,
                    order_type=OrderType.MARKET,
                    strategy_id=position.strategy_id,
                    metadata={'exit_type': trigger['type']}
                )
                
    async def _monitor_strategies(self):
        """Monitor strategy health and performance"""
        
        monitor_interval = 30  # seconds
        
        while self.is_running:
            try:
                # Log performance summary
                total_value = self.position_manager.get_portfolio_value()
                total_pnl = sum(p['total_pnl'] for p in self.strategy_performance.values())
                
                logger.info(f"Portfolio Value: {total_value:.2f}, Total P&L: {total_pnl:.2f}")
                logger.info(f"Active Strategies: {len(self.active_strategies)}, Open Positions: {len(self.position_manager.positions)}")
                logger.info(f"Ticks: {self._tick_count}, Signals: {self._signal_count}, Orders: {self._order_count}")
                
                # Check strategy health
                for strategy_id, strategy in self.active_strategies.items():
                    task = self.strategy_tasks.get(strategy_id)
                    if task and task.done():
                        # Strategy task completed unexpectedly
                        logger.error(f"Strategy {strategy_id} task completed unexpectedly")
                        
                        # Try to get exception
                        try:
                            task.result()
                        except Exception as e:
                            logger.error(f"Strategy error: {e}")
                            
                # Save performance snapshot to database
                if self.db:
                    for strategy_id, config in self.strategy_configs.items():
                        if 'db_id' in config:
                            positions = self.position_manager.get_strategy_positions(strategy_id)
                            crud.create_performance_snapshot(
                                self.db,
                                date=datetime.now(),
                                portfolio_value=total_value,
                                cash_balance=self.virtual_broker.capital,
                                positions_value=sum(p['market_value'] for p in positions),
                                strategy_instance_id=config['db_id'],
                                daily_pnl=0,  # Calculate properly
                                total_pnl=self.strategy_performance[strategy_id]['total_pnl'],
                                open_positions=len(positions)
                            )
                            
                await asyncio.sleep(monitor_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring: {e}")
                await asyncio.sleep(monitor_interval)
                
    async def _load_initial_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Load initial historical data for a symbol"""
        
        try:
            # Get instrument token
            token = self.kite_client.get_instrument_token(symbol)
            if not token:
                logger.warning(f"No instrument token found for {symbol}")
                return None
                
            # Load last N periods
            lookback = 500
            data = await self.data_manager.get_historical_data(
                instrument_token=token,
                interval=timeframe,
                lookback_periods=lookback
            )
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading initial data for {symbol}: {e}")
            return None
            
    def _get_strategy_data(self, strategy_id: str, symbol: str) -> Optional[pd.DataFrame]:
        """Get data buffer for strategy and symbol"""
        
        if strategy_id in self._strategy_data_buffer:
            return self._strategy_data_buffer[strategy_id].get(symbol)
        return None
        
    def _update_data_buffers(self, symbol: str, tick: Dict[str, Any]):
        """Update data buffers with new tick"""
        
        # Create OHLCV bar from tick
        bar = {
            'open': tick.get('last_price', 0),
            'high': tick.get('last_price', 0),
            'low': tick.get('last_price', 0),
            'close': tick.get('last_price', 0),
            'volume': tick.get('volume', 0)
        }
        
        timestamp = tick.get('timestamp', datetime.now())
        
        # Update buffers for all strategies using this symbol
        for strategy_id, strategy in self.active_strategies.items():
            if symbol in strategy.symbols:
                if strategy_id not in self._strategy_data_buffer:
                    self._strategy_data_buffer[strategy_id] = {}
                    
                if symbol not in self._strategy_data_buffer[strategy_id]:
                    self._strategy_data_buffer[strategy_id][symbol] = pd.DataFrame()
                    
                # Append to buffer
                buffer = self._strategy_data_buffer[strategy_id][symbol]
                new_row = pd.DataFrame([bar], index=[timestamp])
                buffer = pd.concat([buffer, new_row])
                
                # Limit buffer size
                if len(buffer) > self._buffer_size:
                    buffer = buffer.iloc[-self._buffer_size:]
                    
                self._strategy_data_buffer[strategy_id][symbol] = buffer
                
    async def _get_instruments_for_symbols(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """Get instrument details for symbols"""
        
        instruments = []
        for symbol in symbols:
            token = self.kite_client.get_instrument_token(symbol)
            if token:
                instruments.append({
                    'symbol': symbol,
                    'instrument_token': token
                })
            else:
                logger.warning(f"No instrument token found for {symbol}")
                
        return instruments
        
    def _get_symbol_from_tick(self, tick: Dict[str, Any]) -> Optional[str]:
        """Get symbol from tick data"""
        
        token = tick.get('instrument_token')
        if token:
            return self.kite_client.get_symbol_from_token(token)
        return None
        
    async def _unsubscribe_unused_symbols(self, symbols: List[str]):
        """Unsubscribe from symbols not used by any strategy"""
        
        # Get all symbols used by active strategies
        used_symbols = set()
        for strategy in self.active_strategies.values():
            used_symbols.update(strategy.symbols)
            
        # Find unused symbols
        unused = set(symbols) - used_symbols
        
        if unused:
            # Get tokens and unsubscribe
            tokens = []
            for symbol in unused:
                token = self.kite_client.get_instrument_token(symbol)
                if token:
                    tokens.append(token)
                    
            if tokens:
                self.ws_manager.unsubscribe(tokens)
                
    async def _close_position(self, strategy_id: str, symbol: str, price: float):
        """Close a position for a strategy"""
        
        position = self.position_manager.get_position(symbol)
        if not position:
            return
            
        # Place closing order
        await self.order_manager.place_order(
            symbol=symbol,
            side=OrderSide.SELL if position.side == "LONG" else OrderSide.BUY,
            quantity=position.quantity,
            order_type=OrderType.MARKET,
            strategy_id=strategy_id,
            metadata={'reason': 'strategy_stopped'}
        )
        
    async def _generate_strategy_report(self, strategy_id: str):
        """Generate performance report for a strategy"""
        
        config = self.strategy_configs.get(strategy_id, {})
        performance = self.strategy_performance.get(strategy_id, {})
        
        duration = datetime.now() - config.get('started_at', datetime.now())
        
        report = {
            'strategy_id': strategy_id,
            'name': config.get('name', 'Unknown'),
            'duration': str(duration),
            'symbols': config.get('symbols', []),
            'performance': performance,
            'final_pnl': performance.get('total_pnl', 0),
            'total_trades': performance.get('total_trades', 0),
            'total_signals': performance.get('total_signals', 0),
            'signal_to_trade_ratio': (performance.get('total_trades', 0) / 
                                     max(1, performance.get('total_signals', 0)))
        }
        
        logger.info(f"Strategy Report for {strategy_id}:")
        logger.info(f"  Name: {report['name']}")
        logger.info(f"  Duration: {report['duration']}")
        logger.info(f"  Final P&L: {report['final_pnl']:.2f}")
        logger.info(f"  Total Trades: {report['total_trades']}")
        
        return report
        
    async def _generate_final_report(self):
        """Generate final report for all strategies"""
        
        total_duration = datetime.now() - self._start_time
        portfolio_value = self.position_manager.get_portfolio_value()
        total_return = ((portfolio_value - self.virtual_broker.initial_capital) / 
                       self.virtual_broker.initial_capital * 100)
        
        logger.info("=== Paper Trading Session Summary ===")
        logger.info(f"Duration: {total_duration}")
        logger.info(f"Initial Capital: {self.virtual_broker.initial_capital:.2f}")
        logger.info(f"Final Portfolio Value: {portfolio_value:.2f}")
        logger.info(f"Total Return: {total_return:.2f}%")
        logger.info(f"Total Strategies Run: {len(self.strategy_configs)}")
        logger.info(f"Total Ticks Processed: {self._tick_count}")
        logger.info(f"Total Signals Generated: {self._signal_count}")
        logger.info(f"Total Orders Placed: {self._order_count}")
        
    def get_status(self) -> Dict[str, Any]:
        """Get current engine status"""
        
        return {
            'running': self.is_running,
            'uptime': str(datetime.now() - self._start_time),
            'active_strategies': len(self.active_strategies),
            'open_positions': len(self.position_manager.positions),
            'portfolio_value': self.position_manager.get_portfolio_value(),
            'total_pnl': sum(p['total_pnl'] for p in self.strategy_performance.values()),
            'tick_count': self._tick_count,
            'signal_count': self._signal_count,
            'order_count': self._order_count,
            'strategies': {
                sid: {
                    'name': config['name'],
                    'symbols': config['symbols'],
                    'performance': self.strategy_performance.get(sid, {})
                }
                for sid, config in self.strategy_configs.items()
            }
        }