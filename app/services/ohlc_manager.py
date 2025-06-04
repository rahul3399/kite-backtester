from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import pandas as pd
import numpy as np
from ..models.trading import OHLCData
import logging

logger = logging.getLogger(__name__)

class OHLCManager:
    def __init__(self, candle_interval: int = 60):  # 60 seconds = 1 minute candles
        self.candle_interval = candle_interval
        self.ohlc_data: Dict[int, Dict[datetime, OHLCData]] = defaultdict(dict)
        self.current_candles: Dict[int, OHLCData] = {}
        self.tick_buffer: Dict[int, List[Dict]] = defaultdict(list)
        
    def process_tick(self, instrument_token: int, tick: Dict):
        """Process incoming tick and update OHLC data"""
        timestamp = tick.get('timestamp', datetime.now())
        price = tick.get('last_price', 0)
        volume = tick.get('volume', 0)
        oi = tick.get('oi', None)
        
        # Get candle timestamp (rounded to interval)
        candle_time = self._get_candle_timestamp(timestamp)
        
        # Check if we need to create a new candle
        if instrument_token not in self.current_candles:
            self._create_new_candle(instrument_token, candle_time, price, volume, oi)
        elif self.current_candles[instrument_token].timestamp != candle_time:
            # Save completed candle and create new one
            self._save_completed_candle(instrument_token)
            self._create_new_candle(instrument_token, candle_time, price, volume, oi)
        else:
            # Update current candle
            self._update_current_candle(instrument_token, price, volume, oi)
        
        # Buffer tick for detailed analysis if needed
        self.tick_buffer[instrument_token].append(tick)
        
        # Limit buffer size
        if len(self.tick_buffer[instrument_token]) > 1000:
            self.tick_buffer[instrument_token] = self.tick_buffer[instrument_token][-500:]
    
    def _get_candle_timestamp(self, timestamp: datetime) -> datetime:
        """Get candle timestamp rounded to interval"""
        seconds = int(timestamp.timestamp())
        rounded_seconds = (seconds // self.candle_interval) * self.candle_interval
        return datetime.fromtimestamp(rounded_seconds)
    
    def _create_new_candle(self, instrument_token: int, timestamp: datetime, 
                          price: float, volume: int, oi: Optional[int]):
        """Create a new OHLC candle"""
        self.current_candles[instrument_token] = OHLCData(
            timestamp=timestamp,
            open=price,
            high=price,
            low=price,
            close=price,
            volume=volume,
            oi=oi
        )
    
    def _update_current_candle(self, instrument_token: int, price: float, 
                              volume: int, oi: Optional[int]):
        """Update current candle with new tick data"""
        candle = self.current_candles[instrument_token]
        candle.high = max(candle.high, price)
        candle.low = min(candle.low, price)
        candle.close = price
        candle.volume = volume  # Usually cumulative volume from exchange
        candle.oi = oi
    
    def _save_completed_candle(self, instrument_token: int):
        """Save completed candle to historical data"""
        if instrument_token in self.current_candles:
            candle = self.current_candles[instrument_token]
            self.ohlc_data[instrument_token][candle.timestamp] = candle
            
            # Limit historical data (keep last 1000 candles)
            if len(self.ohlc_data[instrument_token]) > 1000:
                oldest_time = min(self.ohlc_data[instrument_token].keys())
                del self.ohlc_data[instrument_token][oldest_time]
    
    def get_latest_ohlc(self, instrument_token: int) -> Optional[OHLCData]:
        """Get the latest OHLC data for an instrument"""
        return self.current_candles.get(instrument_token)
    
    def get_historical_ohlc(self, instrument_token: int, 
                           periods: int = 100) -> List[OHLCData]:
        """Get historical OHLC data"""
        if instrument_token not in self.ohlc_data:
            return []
        
        # Get sorted candles
        candles = sorted(
            self.ohlc_data[instrument_token].items(),
            key=lambda x: x[0],
            reverse=True
        )[:periods]
        
        return [candle for _, candle in reversed(candles)]
    
    def get_ohlc_dataframe(self, instrument_token: int, 
                          periods: int = 100) -> pd.DataFrame:
        """Get OHLC data as pandas DataFrame"""
        ohlc_list = self.get_historical_ohlc(instrument_token, periods)
        
        if not ohlc_list:
            return pd.DataFrame()
        
        data = []
        for ohlc in ohlc_list:
            data.append({
                'timestamp': ohlc.timestamp,
                'open': ohlc.open,
                'high': ohlc.high,
                'low': ohlc.low,
                'close': ohlc.close,
                'volume': ohlc.volume,
                'oi': ohlc.oi
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
    
    def calculate_indicators(self, instrument_token: int) -> Dict:
        """Calculate technical indicators for the instrument"""
        df = self.get_ohlc_dataframe(instrument_token)
        
        if df.empty or len(df) < 20:
            return {}
        
        indicators = {}
        
        # Simple Moving Averages
        indicators['sma_20'] = df['close'].rolling(window=20).mean().iloc[-1]
        if len(df) >= 50:
            indicators['sma_50'] = df['close'].rolling(window=50).mean().iloc[-1]
        
        # Exponential Moving Averages
        indicators['ema_20'] = df['close'].ewm(span=20, adjust=False).mean().iloc[-1]
        
        # RSI
        indicators['rsi'] = self._calculate_rsi(df['close'])
        
        # Bollinger Bands
        sma_20 = df['close'].rolling(window=20).mean()
        std_20 = df['close'].rolling(window=20).std()
        indicators['bb_upper'] = (sma_20 + 2 * std_20).iloc[-1]
        indicators['bb_lower'] = (sma_20 - 2 * std_20).iloc[-1]
        indicators['bb_middle'] = sma_20.iloc[-1]
        
        # VWAP (if volume is available)
        if df['volume'].sum() > 0:
            indicators['vwap'] = ((df['close'] * df['volume']).sum() / df['volume'].sum())
        
        # ATR (Average True Range)
        indicators['atr'] = self._calculate_atr(df)
        
        return indicators
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not rsi.empty else 50.0
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(period).mean()
        
        return atr.iloc[-1] if not atr.empty else 0.0
    
    def get_spread_ohlc(self, token1: int, token2: int, 
                       periods: int = 100) -> pd.DataFrame:
        """Calculate OHLC for spread between two instruments"""
        df1 = self.get_ohlc_dataframe(token1, periods)
        df2 = self.get_ohlc_dataframe(token2, periods)
        
        if df1.empty or df2.empty:
            return pd.DataFrame()
        
        # Align dataframes
        df_merged = pd.merge(
            df1[['open', 'high', 'low', 'close']],
            df2[['open', 'high', 'low', 'close']],
            left_index=True,
            right_index=True,
            suffixes=('_1', '_2')
        )
        
        # Calculate spread OHLC
        spread_df = pd.DataFrame(index=df_merged.index)
        spread_df['open'] = df_merged['open_1'] - df_merged['open_2']
        spread_df['close'] = df_merged['close_1'] - df_merged['close_2']
        
        # High is max spread, Low is min spread during the period
        spread_df['high'] = (df_merged['high_1'] - df_merged['low_2'])
        spread_df['low'] = (df_merged['low_1'] - df_merged['high_2'])
        
        return spread_df