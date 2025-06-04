import asyncio
import json
from typing import Dict, List, Callable, Optional
from kiteconnect import KiteTicker
import logging
from datetime import datetime
from ..config import settings
from ..models.trading import OHLCData

logger = logging.getLogger(__name__)

class WebSocketManager:
    def __init__(self, api_key: str, access_token: str):
        self.api_key = api_key
        self.access_token = access_token
        self.kws: Optional[KiteTicker] = None
        self.subscribers: Dict[int, List[Callable]] = {}
        self.ohlc_data: Dict[int, OHLCData] = {}
        self.is_connected = False
        
    async def connect(self):
        """Establish WebSocket connection"""
        try:
            self.kws = KiteTicker(self.api_key, self.access_token)
            
            self.kws.on_ticks = self._on_ticks
            self.kws.on_connect = self._on_connect
            self.kws.on_close = self._on_close
            self.kws.on_error = self._on_error
            
            # Run in separate thread
            await asyncio.get_event_loop().run_in_executor(
                None, self.kws.connect, True
            )
            
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            raise
    
    def _on_ticks(self, ws, ticks):
        """Handle incoming tick data"""
        for tick in ticks:
            instrument_token = tick['instrument_token']
            
            # Update OHLC data
            self._update_ohlc(instrument_token, tick)
            
            # Notify subscribers
            if instrument_token in self.subscribers:
                for callback in self.subscribers[instrument_token]:
                    try:
                        callback(tick)
                    except Exception as e:
                        logger.error(f"Subscriber callback error: {e}")
    
    def _update_ohlc(self, instrument_token: int, tick: dict):
        """Update OHLC data for the instrument"""
        timestamp = tick.get('timestamp', datetime.now())
        price = tick.get('last_price', 0)
        volume = tick.get('volume', 0)
        oi = tick.get('oi', None)
        
        if instrument_token not in self.ohlc_data:
            self.ohlc_data[instrument_token] = OHLCData(
                timestamp=timestamp,
                open=price,
                high=price,
                low=price,
                close=price,
                volume=volume,
                oi=oi
            )
        else:
            ohlc = self.ohlc_data[instrument_token]
            ohlc.high = max(ohlc.high, price)
            ohlc.low = min(ohlc.low, price)
            ohlc.close = price
            ohlc.volume = volume
            ohlc.oi = oi
            ohlc.timestamp = timestamp
    
    def _on_connect(self, ws, response):
        """Handle successful connection"""
        logger.info("WebSocket connected successfully")
        self.is_connected = True
    
    def _on_close(self, ws, code, reason):
        """Handle connection close"""
        logger.warning(f"WebSocket closed: {code} - {reason}")
        self.is_connected = False
    
    def _on_error(self, ws, code, reason):
        """Handle WebSocket errors"""
        logger.error(f"WebSocket error: {code} - {reason}")
    
    def subscribe(self, tokens: List[int], callback: Optional[Callable] = None):
        """Subscribe to instruments"""
        if self.kws and self.is_connected:
            self.kws.subscribe(tokens)
            self.kws.set_mode(self.kws.MODE_FULL, tokens)
            
            # Register callbacks
            if callback:
                for token in tokens:
                    if token not in self.subscribers:
                        self.subscribers[token] = []
                    self.subscribers[token].append(callback)
    
    def unsubscribe(self, tokens: List[int]):
        """Unsubscribe from instruments"""
        if self.kws and self.is_connected:
            self.kws.unsubscribe(tokens)
            for token in tokens:
                if token in self.subscribers:
                    del self.subscribers[token]
    
    async def disconnect(self):
        """Disconnect WebSocket"""
        if self.kws:
            self.kws.stop()
            self.is_connected = False