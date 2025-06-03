
# app/core/websocket_manager.py
import asyncio
from typing import Dict, List, Callable, Optional, Any, Set
from datetime import datetime, timedelta
import logging
from kiteconnect import KiteTicker
from collections import defaultdict, deque
import json
import threading
from enum import Enum

logger = logging.getLogger(__name__)

class SubscriptionMode(Enum):
    """WebSocket subscription modes"""
    LTP = "ltp"
    QUOTE = "quote"
    FULL = "full"

class ConnectionState(Enum):
    """WebSocket connection states"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"

class WebSocketManager:
    """
    Manages WebSocket connections for real-time market data
    Handles reconnection, subscription management, and data distribution
    """
    
    def __init__(self, api_key: str, access_token: str):
        self.api_key = api_key
        self.access_token = access_token
        
        # KiteTicker instance
        self.ticker = None
        
        # Subscription management
        self.subscriptions: Dict[int, Set[SubscriptionMode]] = defaultdict(set)
        self.mode_subscriptions: Dict[SubscriptionMode, Set[int]] = defaultdict(set)
        
        # Callback management
        self.callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        # Connection state
        self.connection_state = ConnectionState.DISCONNECTED
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 10
        self._reconnect_delay = 5  # seconds
        
        # Data management
        self._tick_buffer: Dict[int, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._last_tick_time: Dict[int, datetime] = {}
        self._tick_count = 0
        self._error_count = 0
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Performance monitoring
        self._start_time = datetime.now()
        self._message_latency: deque = deque(maxlen=100)
        
    def initialize(self):
        """Initialize KiteTicker with callbacks"""
        if self.ticker:
            logger.warning("WebSocket already initialized")
            return
            
        logger.info("Initializing WebSocket connection")
        
        self.ticker = KiteTicker(self.api_key, self.access_token)
        
        # Set up callbacks
        self.ticker.on_ticks = self._on_ticks
        self.ticker.on_connect = self._on_connect
        self.ticker.on_close = self._on_close
        self.ticker.on_error = self._on_error
        self.ticker.on_reconnect = self._on_reconnect
        self.ticker.on_noreconnect = self._on_noreconnect
        self.ticker.on_order_update = self._on_order_update
        
        # Configure reconnection
        self.ticker.enable_reconnect(
            reconnect_interval=self._reconnect_delay,
            reconnect_tries=self._max_reconnect_attempts
        )
        
    async def connect(self):
        """Connect to WebSocket asynchronously"""
        if not self.ticker:
            self.initialize()
            
        if self.connection_state == ConnectionState.CONNECTED:
            logger.warning("Already connected to WebSocket")
            return
            
        self.connection_state = ConnectionState.CONNECTING
        
        # Run ticker in a separate thread
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._connect_sync)
        
    def _connect_sync(self):
        """Synchronous connection method"""
        try:
            self.ticker.connect(threaded=True)
            logger.info("WebSocket connection initiated")
        except Exception as e:
            logger.error(f"Error connecting to WebSocket: {e}")
            self.connection_state = ConnectionState.ERROR
            raise
            
    def disconnect(self):
        """Disconnect WebSocket"""
        if self.ticker and self.ticker.is_connected():
            logger.info("Disconnecting WebSocket")
            self.ticker.close()
            self.connection_state = ConnectionState.DISCONNECTED
            
    def is_connected(self) -> bool:
        """Check if WebSocket is connected"""
        return self.connection_state == ConnectionState.CONNECTED
        
    def subscribe(self, tokens: List[int], mode: str = "full"):
        """
        Subscribe to instruments
        
        Args:
            tokens: List of instrument tokens
            mode: Subscription mode (ltp, quote, full)
        """
        if not tokens:
            return
            
        mode_enum = SubscriptionMode(mode.lower())
        
        with self._lock:
            # Update subscription tracking
            for token in tokens:
                self.subscriptions[token].add(mode_enum)
                self.mode_subscriptions[mode_enum].add(token)
                
            # Subscribe if connected
            if self.is_connected() and self.ticker:
                try:
                    self.ticker.subscribe(tokens)
                    self.ticker.set_mode(mode_enum.value, tokens)
                    logger.info(f"Subscribed to {len(tokens)} instruments in {mode} mode")
                except Exception as e:
                    logger.error(f"Error subscribing: {e}")
            else:
                logger.warning("WebSocket not connected. Subscriptions will be applied on connect.")
                
    def unsubscribe(self, tokens: List[int]):
        """Unsubscribe from instruments"""
        if not tokens:
            return
            
        with self._lock:
            # Update subscription tracking
            for token in tokens:
                if token in self.subscriptions:
                    del self.subscriptions[token]
                    
                # Remove from mode subscriptions
                for mode_set in self.mode_subscriptions.values():
                    mode_set.discard(token)
                    
            # Unsubscribe if connected
            if self.is_connected() and self.ticker:
                try:
                    self.ticker.unsubscribe(tokens)
                    logger.info(f"Unsubscribed from {len(tokens)} instruments")
                except Exception as e:
                    logger.error(f"Error unsubscribing: {e}")
                    
    def set_mode(self, tokens: List[int], mode: str):
        """Change subscription mode for instruments"""
        if not tokens:
            return
            
        mode_enum = SubscriptionMode(mode.lower())
        
        with self._lock:
            # Update mode tracking
            for token in tokens:
                # Remove from other modes
                for m, token_set in self.mode_subscriptions.items():
                    if m != mode_enum:
                        token_set.discard(token)
                        
                # Add to new mode
                self.mode_subscriptions[mode_enum].add(token)
                
                # Update subscription
                if token in self.subscriptions:
                    self.subscriptions[token] = {mode_enum}
                    
            # Apply mode if connected
            if self.is_connected() and self.ticker:
                try:
                    self.ticker.set_mode(mode_enum.value, tokens)
                    logger.info(f"Set mode {mode} for {len(tokens)} instruments")
                except Exception as e:
                    logger.error(f"Error setting mode: {e}")
                    
    def register_callback(self, event: str, callback: Callable):
        """Register callback for events"""
        self.callbacks[event].append(callback)
        logger.debug(f"Registered callback for event: {event}")
        
    def unregister_callback(self, event: str, callback: Callable):
        """Unregister callback"""
        if callback in self.callbacks[event]:
            self.callbacks[event].remove(callback)
            logger.debug(f"Unregistered callback for event: {event}")
            
    def _on_ticks(self, ws, ticks):
        """Handle incoming ticks"""
        try:
            self._tick_count += len(ticks)
            
            # Process each tick
            for tick in ticks:
                token = tick.get('instrument_token')
                if token:
                    # Add timestamp if not present
                    if 'timestamp' not in tick:
                        tick['timestamp'] = datetime.now()
                        
                    # Buffer tick
                    self._tick_buffer[token].append(tick)
                    self._last_tick_time[token] = datetime.now()
                    
                    # Calculate latency if exchange timestamp available
                    if 'exchange_timestamp' in tick:
                        latency = (datetime.now() - tick['exchange_timestamp']).total_seconds()
                        self._message_latency.append(latency)
                        
            # Execute callbacks
            for callback in self.callbacks.get("tick", []):
                try:
                    # Run async callbacks in event loop
                    if asyncio.iscoroutinefunction(callback):
                        asyncio.create_task(callback(ticks))
                    else:
                        callback(ticks)
                except Exception as e:
                    logger.error(f"Error in tick callback: {e}")
                    
        except Exception as e:
            logger.error(f"Error processing ticks: {e}")
            self._error_count += 1
            
    def _on_connect(self, ws, response):
        """Handle connection established"""
        logger.info("WebSocket connected successfully")
        
        self.connection_state = ConnectionState.CONNECTED
        self._reconnect_attempts = 0
        
        # Resubscribe to all instruments
        with self._lock:
            if self.subscriptions:
                all_tokens = list(self.subscriptions.keys())
                
                try:
                    # Subscribe all at once
                    self.ticker.subscribe(all_tokens)
                    
                    # Set modes
                    for mode, tokens in self.mode_subscriptions.items():
                        if tokens:
                            self.ticker.set_mode(mode.value, list(tokens))
                            
                    logger.info(f"Resubscribed to {len(all_tokens)} instruments")
                    
                except Exception as e:
                    logger.error(f"Error resubscribing: {e}")
                    
        # Execute callbacks
        for callback in self.callbacks.get("connect", []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(response))
                else:
                    callback(response)
            except Exception as e:
                logger.error(f"Error in connect callback: {e}")
                
    def _on_close(self, ws, code, reason):
        """Handle connection closed"""
        logger.warning(f"WebSocket closed: {code} - {reason}")
        
        self.connection_state = ConnectionState.DISCONNECTED
        
        # Execute callbacks
        for callback in self.callbacks.get("close", []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback({"code": code, "reason": reason}))
                else:
                    callback({"code": code, "reason": reason})
            except Exception as e:
                logger.error(f"Error in close callback: {e}")
                
    def _on_error(self, ws, code, reason):
        """Handle errors"""
        logger.error(f"WebSocket error: {code} - {reason}")
        
        self._error_count += 1
        
        # Execute callbacks
        for callback in self.callbacks.get("error", []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback({"code": code, "reason": reason}))
                else:
                    callback({"code": code, "reason": reason})
            except Exception as e:
                logger.error(f"Error in error callback: {e}")
                
    def _on_reconnect(self, ws, attempts_count):
        """Handle reconnection attempts"""
        logger.info(f"WebSocket reconnecting... Attempt: {attempts_count}")
        
        self.connection_state = ConnectionState.RECONNECTING
        self._reconnect_attempts = attempts_count
        
    def _on_noreconnect(self, ws):
        """Handle when reconnection fails"""
        logger.error("WebSocket reconnection failed. Max attempts reached.")
        
        self.connection_state = ConnectionState.ERROR
        
        # Execute callbacks
        for callback in self.callbacks.get("noreconnect", []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback())
                else:
                    callback()
            except Exception as e:
                logger.error(f"Error in noreconnect callback: {e}")
                
    def _on_order_update(self, ws, data):
        """Handle order updates"""
        logger.info(f"Order update received: {data}")
        
        # Execute callbacks
        for callback in self.callbacks.get("order_update", []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(data))
                else:
                    callback(data)
            except Exception as e:
                logger.error(f"Error in order update callback: {e}")
                
    def get_tick_buffer(self, token: int, clear: bool = True) -> List[Dict[str, Any]]:
        """Get buffered ticks for an instrument"""
        with self._lock:
            ticks = list(self._tick_buffer[token])
            if clear:
                self._tick_buffer[token].clear()
            return ticks
            
    def get_latest_tick(self, token: int) -> Optional[Dict[str, Any]]:
        """Get latest tick for an instrument"""
        with self._lock:
            if token in self._tick_buffer and self._tick_buffer[token]:
                return self._tick_buffer[token][-1]
        return None
        
    def get_all_latest_ticks(self) -> Dict[int, Dict[str, Any]]:
        """Get latest ticks for all subscribed instruments"""
        latest_ticks = {}
        
        with self._lock:
            for token in self.subscriptions:
                if token in self._tick_buffer and self._tick_buffer[token]:
                    latest_ticks[token] = self._tick_buffer[token][-1]
                    
        return latest_ticks
        
    def get_connection_status(self) -> Dict[str, Any]:
        """Get detailed connection status"""
        uptime = (datetime.now() - self._start_time).total_seconds()
        
        # Calculate average latency
        avg_latency = None
        if self._message_latency:
            avg_latency = sum(self._message_latency) / len(self._message_latency)
            
        return {
            "state": self.connection_state.value,
            "connected": self.is_connected(),
            "subscribed_instruments": len(self.subscriptions),
            "reconnect_attempts": self._reconnect_attempts,
            "tick_count": self._tick_count,
            "error_count": self._error_count,
            "uptime_seconds": uptime,
            "average_latency_ms": avg_latency * 1000 if avg_latency else None,
            "last_tick_times": {
                token: time.isoformat() 
                for token, time in self._last_tick_time.items()
            },
            "buffer_sizes": {
                token: len(buffer) 
                for token, buffer in self._tick_buffer.items()
            }
        }
        
    def get_subscription_info(self) -> Dict[str, Any]:
        """Get subscription information"""
        with self._lock:
            return {
                "total_subscriptions": len(self.subscriptions),
                "mode_breakdown": {
                    mode.value: len(tokens) 
                    for mode, tokens in self.mode_subscriptions.items()
                },
                "subscriptions": {
                    token: [mode.value for mode in modes]
                    for token, modes in self.subscriptions.items()
                }
            }
            
    def clear_buffers(self):
        """Clear all tick buffers"""
        with self._lock:
            for buffer in self._tick_buffer.values():
                buffer.clear()
            logger.info("Cleared all tick buffers")
            
    def reset_statistics(self):
        """Reset connection statistics"""
        self._tick_count = 0
        self._error_count = 0
        self._message_latency.clear()
        self._start_time = datetime.now()
        logger.info("Reset connection statistics")