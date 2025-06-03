# app/paper_trading/live_feed.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import asyncio
import logging
from collections import defaultdict, deque
import threading

from ..core.websocket_manager import WebSocketManager
from ..core.data_manager import DataManager

logger = logging.getLogger(__name__)

class TickData:
    """Container for tick data"""
    def __init__(self, symbol: str, timestamp: datetime, price: float, volume: int):
        self.symbol = symbol
        self.timestamp = timestamp
        self.price = price
        self.volume = volume
        self.bid = price
        self.ask = price
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp,
            'price': self.price,
            'volume': self.volume,
            'bid': self.bid,
            'ask': self.ask
        }

class CandleAggregator:
    """Aggregates ticks into OHLCV candles"""
    
    def __init__(self, symbol: str, timeframe: str):
        self.symbol = symbol
        self.timeframe = timeframe
        self.current_candle: Optional[Dict[str, Any]] = None
        self.completed_candles: deque = deque(maxlen=1000)
        
        # Map timeframe to seconds
        self.timeframe_seconds = self._get_timeframe_seconds(timeframe)
        
    def _get_timeframe_seconds(self, timeframe: str) -> int:
        """Convert timeframe string to seconds"""
        mapping = {
            '1minute': 60, 'minute': 60,
            '3minute': 180, '5minute': 300,
            '10minute': 600, '15minute': 900,
            '30minute': 1800, '60minute': 3600,
            'hour': 3600, 'day': 86400
        }
        return mapping.get(timeframe, 300)  # Default 5 minutes
        
    def add_tick(self, tick: TickData) -> Optional[Dict[str, Any]]:
        """
        Add tick and return completed candle if any
        
        Returns:
            Completed candle dict or None
        """
        
        # Calculate candle timestamp
        candle_time = self._get_candle_time(tick.timestamp)
        
        # Check if we need to close current candle
        if self.current_candle and self.current_candle['timestamp'] < candle_time:
            completed = self.current_candle.copy()
            self.completed_candles.append(completed)
            self.current_candle = None
            
            # Start new candle
            self._start_new_candle(tick, candle_time)
            
            return completed
            
        # Update current candle
        if self.current_candle is None:
            self._start_new_candle(tick, candle_time)
        else:
            self._update_candle(tick)
            
        return None
        
    def _get_candle_time(self, timestamp: datetime) -> datetime:
        """Get candle timestamp based on timeframe"""
        
        # Round down to timeframe
        total_seconds = int(timestamp.timestamp())
        candle_seconds = (total_seconds // self.timeframe_seconds) * self.timeframe_seconds
        
        return datetime.fromtimestamp(candle_seconds)
        
    def _start_new_candle(self, tick: TickData, candle_time: datetime):
        """Start a new candle"""
        
        self.current_candle = {
            'symbol': self.symbol,
            'timestamp': candle_time,
            'open': tick.price,
            'high': tick.price,
            'low': tick.price,
            'close': tick.price,
            'volume': tick.volume,
            'tick_count': 1
        }
        
    def _update_candle(self, tick: TickData):
        """Update current candle with new tick"""
        
        if self.current_candle:
            self.current_candle['high'] = max(self.current_candle['high'], tick.price)
            self.current_candle['low'] = min(self.current_candle['low'], tick.price)
            self.current_candle['close'] = tick.price
            self.current_candle['volume'] += tick.volume
            self.current_candle['tick_count'] += 1
            
    def get_candles(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Get completed candles as DataFrame"""
        
        candles = list(self.completed_candles)
        
        # Add current candle if exists
        if self.current_candle:
            candles.append(self.current_candle)
            
        if not candles:
            return pd.DataFrame()
            
        # Limit results
        if limit:
            candles = candles[-limit:]
            
        # Convert to DataFrame
        df = pd.DataFrame(candles)
        df.set_index('timestamp', inplace=True)
        
        return df

class LiveDataFeed:
    """
    Manages live data feed for paper trading
    Handles tick aggregation, data distribution, and historical data integration
    """
    
    def __init__(self, websocket_manager: WebSocketManager, data_manager: DataManager):
        self.ws_manager = websocket_manager
        self.data_manager = data_manager
        
        # Tick processing
        self.tick_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.latest_ticks: Dict[str, TickData] = {}
        
        # Candle aggregation
        self.candle_aggregators: Dict[Tuple[str, str], CandleAggregator] = {}
        
        # Data callbacks
        self.tick_callbacks: List[Callable] = []
        self.candle_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        # Historical data cache
        self.historical_cache: Dict[str, pd.DataFrame] = {}
        self.cache_expiry = timedelta(minutes=5)
        self.cache_timestamps: Dict[str, datetime] = {}
        
        # Processing control
        self._processing = False
        self._processing_task: Optional[asyncio.Task] = None
        self._lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'ticks_processed': 0,
            'candles_generated': 0,
            'callbacks_executed': 0,
            'errors': 0
        }
        
    def start(self):
        """Start live data feed processing"""
        
        if self._processing:
            logger.warning("Live data feed already running")
            return
            
        logger.info("Starting live data feed")
        self._processing = True
        
        # Register WebSocket callback
        self.ws_manager.register_callback("tick", self._on_websocket_tick)
        
        # Start processing task
        self._processing_task = asyncio.create_task(self._process_loop())
        
    def stop(self):
        """Stop live data feed processing"""
        
        logger.info("Stopping live data feed")
        self._processing = False
        
        # Unregister callback
        self.ws_manager.unregister_callback("tick", self._on_websocket_tick)
        
        # Cancel processing task
        if self._processing_task:
            self._processing_task.cancel()
            
    def subscribe_ticks(self, callback: Callable):
        """Subscribe to tick updates"""
        self.tick_callbacks.append(callback)
        
    def unsubscribe_ticks(self, callback: Callable):
        """Unsubscribe from tick updates"""
        if callback in self.tick_callbacks:
            self.tick_callbacks.remove(callback)
            
    def subscribe_candles(self, symbol: str, timeframe: str, callback: Callable):
        """Subscribe to candle updates"""
        
        key = f"{symbol}_{timeframe}"
        self.candle_callbacks[key].append(callback)
        
        # Create aggregator if doesn't exist
        if (symbol, timeframe) not in self.candle_aggregators:
            self.candle_aggregators[(symbol, timeframe)] = CandleAggregator(symbol, timeframe)
            
    def unsubscribe_candles(self, symbol: str, timeframe: str, callback: Callable):
        """Unsubscribe from candle updates"""
        
        key = f"{symbol}_{timeframe}"
        if callback in self.candle_callbacks[key]:
            self.candle_callbacks[key].remove(callback)
            
    def get_latest_tick(self, symbol: str) -> Optional[TickData]:
        """Get latest tick for symbol"""
        return self.latest_ticks.get(symbol)
        
    def get_tick_buffer(self, symbol: str, limit: Optional[int] = None) -> List[TickData]:
        """Get tick buffer for symbol"""
        
        with self._lock:
            ticks = list(self.tick_buffer[symbol])
            
        if limit:
            ticks = ticks[-limit:]
            
        return ticks
        
    async def get_historical_data(self, 
                                symbol: str, 
                                timeframe: str,
                                periods: int = 100) -> pd.DataFrame:
        """
        Get historical data with caching
        
        Args:
            symbol: Trading symbol
            timeframe: Candle timeframe
            periods: Number of periods to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        
        cache_key = f"{symbol}_{timeframe}_{periods}"
        
        # Check cache
        if cache_key in self.historical_cache:
            cache_time = self.cache_timestamps.get(cache_key)
            if cache_time and datetime.now() - cache_time < self.cache_expiry:
                return self.historical_cache[cache_key]
                
        # Fetch from data manager
        try:
            # Get instrument token
            from ..main import kite_client
            token = kite_client.get_instrument_token(symbol)
            
            if token:
                data = await self.data_manager.get_historical_data(
                    instrument_token=token,
                    interval=timeframe,
                    lookback_periods=periods
                )
                
                # Cache result
                self.historical_cache[cache_key] = data
                self.cache_timestamps[cache_key] = datetime.now()
                
                return data
                
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            
        return pd.DataFrame()
        
    def get_combined_data(self, 
                         symbol: str, 
                         timeframe: str,
                         historical_periods: int = 100) -> pd.DataFrame:
        """
        Get combined historical and live data
        
        Args:
            symbol: Trading symbol
            timeframe: Candle timeframe
            historical_periods: Number of historical periods
            
        Returns:
            DataFrame combining historical and live candles
        """
        
        # Get historical data
        historical = asyncio.run(self.get_historical_data(symbol, timeframe, historical_periods))
        
        # Get live candles
        if (symbol, timeframe) in self.candle_aggregators:
            live_candles = self.candle_aggregators[(symbol, timeframe)].get_candles()
            
            if not live_candles.empty and not historical.empty:
                # Combine data (avoid duplicates)
                last_historical_time = historical.index[-1]
                live_candles = live_candles[live_candles.index > last_historical_time]
                
                if not live_candles.empty:
                    combined = pd.concat([historical, live_candles])
                    return combined
                    
        return historical
        
    def _on_websocket_tick(self, ticks: List[Dict[str, Any]]):
        """Handle incoming WebSocket ticks"""
        
        for tick_data in ticks:
            try:
                # Parse tick
                symbol = self._get_symbol_from_tick(tick_data)
                if not symbol:
                    continue
                    
                # Create TickData object
                tick = TickData(
                    symbol=symbol,
                    timestamp=tick_data.get('timestamp', datetime.now()),
                    price=tick_data.get('last_price', 0),
                    volume=tick_data.get('volume', 0)
                )
                
                # Update bid/ask if available
                if 'depth' in tick_data:
                    buy_depth = tick_data['depth'].get('buy', [])
                    sell_depth = tick_data['depth'].get('sell', [])
                    
                    if buy_depth:
                        tick.bid = buy_depth[0].get('price', tick.price)
                    if sell_depth:
                        tick.ask = sell_depth[0].get('price', tick.price)
                        
                # Process tick
                self._process_tick(tick)
                
            except Exception as e:
                logger.error(f"Error processing WebSocket tick: {e}")
                self.stats['errors'] += 1
                
    def _process_tick(self, tick: TickData):
        """Process individual tick"""
        
        with self._lock:
            # Update buffers
            self.tick_buffer[tick.symbol].append(tick)
            self.latest_ticks[tick.symbol] = tick
            
            # Update statistics
            self.stats['ticks_processed'] += 1
            
        # Process candle aggregation
        for (symbol, timeframe), aggregator in self.candle_aggregators.items():
            if symbol == tick.symbol:
                completed_candle = aggregator.add_tick(tick)
                
                if completed_candle:
                    self.stats['candles_generated'] += 1
                    
                    # Notify candle subscribers
                    key = f"{symbol}_{timeframe}"
                    for callback in self.candle_callbacks.get(key, []):
                        try:
                            asyncio.create_task(self._execute_callback(callback, completed_candle))
                        except Exception as e:
                            logger.error(f"Error executing candle callback: {e}")
                            
        # Notify tick subscribers
        for callback in self.tick_callbacks:
            try:
                asyncio.create_task(self._execute_callback(callback, tick))
            except Exception as e:
                logger.error(f"Error executing tick callback: {e}")
                
    async def _execute_callback(self, callback: Callable, data: Any):
        """Execute callback with proper error handling"""
        
        try:
            self.stats['callbacks_executed'] += 1
            
            if asyncio.iscoroutinefunction(callback):
                await callback(data)
            else:
                await asyncio.get_event_loop().run_in_executor(None, callback, data)
                
        except Exception as e:
            logger.error(f"Error in data feed callback: {e}")
            self.stats['errors'] += 1
            
    async def _process_loop(self):
        """Main processing loop for cleanup and maintenance"""
        
        logger.info("Live data feed processing loop started")
        
        while self._processing:
            try:
                # Clean up old cache entries
                self._cleanup_cache()
                
                # Log statistics periodically
                if self.stats['ticks_processed'] % 1000 == 0:
                    logger.info(f"Live feed stats: {self.stats}")
                    
                await asyncio.sleep(60)  # Run every minute
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                await asyncio.sleep(5)
                
    def _cleanup_cache(self):
        """Clean up expired cache entries"""
        
        current_time = datetime.now()
        expired_keys = []
        
        for key, timestamp in self.cache_timestamps.items():
            if current_time - timestamp > self.cache_expiry:
                expired_keys.append(key)
                
        for key in expired_keys:
            del self.historical_cache[key]
            del self.cache_timestamps[key]
            
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
            
    def _get_symbol_from_tick(self, tick_data: Dict[str, Any]) -> Optional[str]:
        """Extract symbol from tick data"""
        
        # This would map instrument token to symbol
        # In practice, you'd use the KiteClient mapping
        token = tick_data.get('instrument_token')
        
        # Placeholder - in production, use proper mapping
        symbol_map = {
            1000: "NIFTY",
            1001: "BANKNIFTY",
            # Add more mappings
        }
        
        return symbol_map.get(token)
        
    def get_market_snapshot(self) -> Dict[str, Any]:
        """Get current market snapshot"""
        
        snapshot = {
            'timestamp': datetime.now(),
            'symbols': {},
            'statistics': self.stats.copy()
        }
        
        # Add latest prices for all symbols
        for symbol, tick in self.latest_ticks.items():
            snapshot['symbols'][symbol] = {
                'price': tick.price,
                'bid': tick.bid,
                'ask': tick.ask,
                'volume': tick.volume,
                'timestamp': tick.timestamp
            }
            
        return snapshot
        
    def simulate_tick(self, symbol: str, price: float, volume: int = 100):
        """
        Simulate a tick for testing
        
        Args:
            symbol: Trading symbol
            price: Tick price
            volume: Tick volume
        """
        
        tick = TickData(
            symbol=symbol,
            timestamp=datetime.now(),
            price=price,
            volume=volume
        )
        
        # Add small spread
        spread = price * 0.001
        tick.bid = price - spread/2
        tick.ask = price + spread/2
        
        self._process_tick(tick)
        
    def get_data_quality_metrics(self, symbol: str) -> Dict[str, Any]:
        """Get data quality metrics for a symbol"""
        
        metrics = {
            'symbol': symbol,
            'has_data': symbol in self.latest_ticks,
            'last_update': None,
            'tick_count': 0,
            'tick_frequency': 0,
            'gaps': 0
        }
        
        if symbol in self.latest_ticks:
            metrics['last_update'] = self.latest_ticks[symbol].timestamp
            
        if symbol in self.tick_buffer:
            ticks = list(self.tick_buffer[symbol])
            metrics['tick_count'] = len(ticks)
            
            if len(ticks) > 1:
                # Calculate average tick frequency
                time_span = (ticks[-1].timestamp - ticks[0].timestamp).total_seconds()
                if time_span > 0:
                    metrics['tick_frequency'] = len(ticks) / time_span
                    
                # Detect gaps (> 1 minute between ticks)
                for i in range(1, len(ticks)):
                    gap = (ticks[i].timestamp - ticks[i-1].timestamp).total_seconds()
                    if gap > 60:
                        metrics['gaps'] += 1
                        
        return metrics
        
    def reset_statistics(self):
        """Reset statistics counters"""
        
        self.stats = {
            'ticks_processed': 0,
            'candles_generated': 0,
            'callbacks_executed': 0,
            'errors': 0
        }
        
        logger.info("Live data feed statistics reset")


class SimulatedDataFeed(LiveDataFeed):
    """
    Simulated data feed for testing without live connection
    Generates realistic market data based on historical patterns
    """
    
    def __init__(self, data_manager: DataManager):
        # Create dummy WebSocket manager
        class DummyWebSocketManager:
            def register_callback(self, event, callback): pass
            def unregister_callback(self, event, callback): pass
            
        super().__init__(DummyWebSocketManager(), data_manager)
        
        # Simulation parameters
        self.symbols: List[str] = []
        self.base_prices: Dict[str, float] = {}
        self.volatilities: Dict[str, float] = {}
        self.tick_interval = 1.0  # seconds
        
        # Simulation task
        self._simulation_task: Optional[asyncio.Task] = None
        
    def add_symbol(self, symbol: str, base_price: float, volatility: float = 0.02):
        """Add symbol to simulation"""
        
        self.symbols.append(symbol)
        self.base_prices[symbol] = base_price
        self.volatilities[symbol] = volatility
        
        logger.info(f"Added {symbol} to simulation: price={base_price}, volatility={volatility}")
        
    def start(self):
        """Start simulated data feed"""
        
        if self._processing:
            return
            
        logger.info("Starting simulated data feed")
        self._processing = True
        
        # Start simulation
        self._simulation_task = asyncio.create_task(self._simulate_market())
        
    def stop(self):
        """Stop simulated data feed"""
        
        logger.info("Stopping simulated data feed")
        self._processing = False
        
        if self._simulation_task:
            self._simulation_task.cancel()
            
    async def _simulate_market(self):
        """Generate simulated market data"""
        
        logger.info("Market simulation started")
        
        # Initialize prices
        current_prices = self.base_prices.copy()
        
        while self._processing:
            try:
                # Generate ticks for all symbols
                for symbol in self.symbols:
                    if symbol not in current_prices:
                        continue
                        
                    # Random walk with mean reversion
                    volatility = self.volatilities.get(symbol, 0.02)
                    base_price = self.base_prices[symbol]
                    current_price = current_prices[symbol]
                    
                    # Mean reversion factor
                    reversion = 0.01 * (base_price - current_price) / base_price
                    
                    # Random component
                    random_change = np.random.normal(reversion, volatility)
                    
                    # Update price
                    new_price = current_price * (1 + random_change)
                    new_price = max(new_price, base_price * 0.5)  # Floor at 50%
                    new_price = min(new_price, base_price * 2.0)  # Cap at 200%
                    
                    current_prices[symbol] = new_price
                    
                    # Generate volume (random with daily pattern)
                    hour = datetime.now().hour
                    volume_factor = 1.0
                    if 9 <= hour <= 10 or 15 <= hour <= 16:
                        volume_factor = 2.0  # Higher volume at open/close
                        
                    volume = int(np.random.exponential(1000) * volume_factor)
                    
                    # Create tick
                    self.simulate_tick(symbol, new_price, volume)
                    
                # Wait for next tick
                await asyncio.sleep(self.tick_interval)
                
            except Exception as e:
                logger.error(f"Error in market simulation: {e}")
                await asyncio.sleep(1)
                
        logger.info("Market simulation stopped")