# app/core/data_manager.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import asyncio
from functools import lru_cache
import logging
import hashlib
import json
import pickle
from pathlib import Path

from .kite_client import KiteClient

logger = logging.getLogger(__name__)

class DataManager:
    """
    Manages historical and real-time data with caching and preprocessing
    Provides unified interface for data access with technical indicators
    """
    
    def __init__(self, kite_client: KiteClient, cache_dir: str = "./data_cache"):
        self.kite_client = kite_client
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # In-memory cache
        self._memory_cache: Dict[str, pd.DataFrame] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        self.cache_duration = timedelta(minutes=5)
        
        # Cache size limits
        self.max_memory_cache_size = 100  # Maximum dataframes in memory
        self.max_disk_cache_days = 30     # Days to keep disk cache
        
        # Technical indicators configuration
        self.indicator_periods = {
            'sma': [5, 10, 20, 50, 100, 200],
            'ema': [9, 12, 26, 50],
            'rsi': [14],
            'macd': [(12, 26, 9)],
            'bb': [(20, 2)],
            'atr': [14],
            'adx': [14]
        }
        
    def get_cache_key(self, 
                     instrument_token: int, 
                     interval: str, 
                     from_date: datetime, 
                     to_date: datetime) -> str:
        """Generate unique cache key"""
        key_string = f"{instrument_token}_{interval}_{from_date.isoformat()}_{to_date.isoformat()}"
        return hashlib.md5(key_string.encode()).hexdigest()
        
    async def get_historical_data(self,
                                instrument_token: int,
                                interval: str,
                                lookback_periods: int,
                                to_date: Optional[datetime] = None,
                                include_indicators: bool = True) -> pd.DataFrame:
        """
        Get historical data with intelligent caching
        
        Args:
            instrument_token: Instrument identifier
            interval: Time interval (minute, 5minute, day, etc.)
            lookback_periods: Number of periods to look back
            to_date: End date (default: now)
            include_indicators: Whether to add technical indicators
            
        Returns:
            DataFrame with OHLCV data and indicators
        """
        
        if to_date is None:
            to_date = datetime.now()
            
        # Calculate from_date based on interval and lookback
        from_date = self._calculate_from_date(interval, lookback_periods, to_date)
        
        # Check memory cache first
        cache_key = self.get_cache_key(instrument_token, interval, from_date, to_date)
        
        if cache_key in self._memory_cache:
            cache_time = self._cache_timestamps.get(cache_key)
            if cache_time and datetime.now() - cache_time < self.cache_duration:
                logger.debug(f"Returning data from memory cache for {instrument_token}")
                return self._memory_cache[cache_key]
                
        # Check disk cache
        df = self._load_from_disk_cache(cache_key)
        if df is not None:
            logger.debug(f"Returning data from disk cache for {instrument_token}")
            self._update_memory_cache(cache_key, df)
            return df
            
        # Fetch from API
        try:
            df = await self._fetch_historical_data(
                instrument_token, from_date, to_date, interval
            )
            
            if df.empty:
                logger.warning(f"No data returned for {instrument_token}")
                return df
                
            # Add technical indicators if requested
            if include_indicators:
                df = self.add_technical_indicators(df)
                
            # Cache the data
            self._update_memory_cache(cache_key, df)
            self._save_to_disk_cache(cache_key, df)
            
            # Cleanup old cache entries
            self._cleanup_cache()
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            
            # Try to return stale cache if available
            df = self._load_from_disk_cache(cache_key, ignore_age=True)
            if df is not None:
                logger.warning("Returning stale cache due to API error")
                return df
                
            raise
            
    async def _fetch_historical_data(self,
                                   instrument_token: int,
                                   from_date: datetime,
                                   to_date: datetime,
                                   interval: str) -> pd.DataFrame:
        """Fetch data from Kite API"""
        
        loop = asyncio.get_event_loop()
        
        # Run synchronous Kite API call in executor
        df = await loop.run_in_executor(
            None,
            self.kite_client.get_historical_data,
            instrument_token,
            from_date,
            to_date,
            interval
        )
        
        return df
        
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators to DataFrame"""
        
        if df.empty:
            return df
            
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Price-based indicators
        df = self._add_moving_averages(df)
        df = self._add_momentum_indicators(df)
        df = self._add_volatility_indicators(df)
        df = self._add_volume_indicators(df)
        df = self._add_pattern_recognition(df)
        
        # Additional features
        df = self._add_price_features(df)
        df = self._add_time_features(df)
        
        return df
        
    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add moving averages"""
        
        # Simple Moving Averages
        for period in self.indicator_periods['sma']:
            if len(df) >= period:
                df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
                
        # Exponential Moving Averages
        for period in self.indicator_periods['ema']:
            if len(df) >= period:
                df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
                
        # VWAP (Volume Weighted Average Price)
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        
        # Hull Moving Average
        for period in [20, 50]:
            if len(df) >= period:
                half_period = int(period / 2)
                sqrt_period = int(np.sqrt(period))
                
                wma_half = df['close'].rolling(half_period).mean()
                wma_full = df['close'].rolling(period).mean()
                
                df[f'hma_{period}'] = (2 * wma_half - wma_full).rolling(sqrt_period).mean()
                
        return df
        
    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators"""
        
        # RSI
        for period in self.indicator_periods['rsi']:
            df[f'rsi_{period}'] = self.calculate_rsi(df['close'], period)
            
        # MACD
        for fast, slow, signal in self.indicator_periods['macd']:
            ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
            ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
            
            df['macd'] = ema_fast - ema_slow
            df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
        # Stochastic Oscillator
        period = 14
        if len(df) >= period:
            low_min = df['low'].rolling(window=period).min()
            high_max = df['high'].rolling(window=period).max()
            
            df['stoch_k'] = 100 * ((df['close'] - low_min) / (high_max - low_min))
            df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
            
        # Williams %R
        if len(df) >= 14:
            df['williams_r'] = -100 * ((df['high'].rolling(14).max() - df['close']) / 
                                      (df['high'].rolling(14).max() - df['low'].rolling(14).min()))
            
        # ROC (Rate of Change)
        for period in [10, 20]:
            if len(df) > period:
                df[f'roc_{period}'] = ((df['close'] - df['close'].shift(period)) / 
                                      df['close'].shift(period)) * 100
                
        return df
        
    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators"""
        
        # Bollinger Bands
        for period, std_dev in self.indicator_periods['bb']:
            if len(df) >= period:
                df[f'bb_middle_{period}'] = df['close'].rolling(window=period).mean()
                bb_std = df['close'].rolling(window=period).std()
                df[f'bb_upper_{period}'] = df[f'bb_middle_{period}'] + (std_dev * bb_std)
                df[f'bb_lower_{period}'] = df[f'bb_middle_{period}'] - (std_dev * bb_std)
                df[f'bb_width_{period}'] = df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']
                df[f'bb_percent_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / df[f'bb_width_{period}']
                
        # ATR (Average True Range)
        for period in self.indicator_periods['atr']:
            df[f'atr_{period}'] = self.calculate_atr(df, period)
            
        # Keltner Channels
        period = 20
        multiplier = 2
        if len(df) >= period:
            middle = df['close'].ewm(span=period, adjust=False).mean()
            atr = self.calculate_atr(df, period)
            
            df['kc_upper'] = middle + (multiplier * atr)
            df['kc_lower'] = middle - (multiplier * atr)
            df['kc_middle'] = middle
            
        # Historical Volatility
        for period in [20, 50]:
            if len(df) > period:
                returns = np.log(df['close'] / df['close'].shift(1))
                df[f'hvol_{period}'] = returns.rolling(period).std() * np.sqrt(252)
                
        return df
        
    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume indicators"""
        
        # On-Balance Volume (OBV)
        obv = [0]
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.append(obv[-1] + df['volume'].iloc[i])
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.append(obv[-1] - df['volume'].iloc[i])
            else:
                obv.append(obv[-1])
        df['obv'] = obv
        
        # Volume Moving Averages
        for period in [10, 20, 50]:
            if len(df) >= period:
                df[f'volume_sma_{period}'] = df['volume'].rolling(window=period).mean()
                
        # Volume Ratio
        if len(df) >= 20:
            df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
            
        # Money Flow Index (MFI)
        period = 14
        if len(df) >= period:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            money_flow = typical_price * df['volume']
            
            positive_flow = pd.Series(0.0, index=df.index)
            negative_flow = pd.Series(0.0, index=df.index)
            
            for i in range(1, len(df)):
                if typical_price.iloc[i] > typical_price.iloc[i-1]:
                    positive_flow.iloc[i] = money_flow.iloc[i]
                else:
                    negative_flow.iloc[i] = money_flow.iloc[i]
                    
            positive_mf = positive_flow.rolling(period).sum()
            negative_mf = negative_flow.rolling(period).sum()
            
            mfi_ratio = positive_mf / negative_mf
            df['mfi'] = 100 - (100 / (1 + mfi_ratio))
            
        # Accumulation/Distribution Line
        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        clv = clv.fillna(0)
        df['ad_line'] = (clv * df['volume']).cumsum()
        
        return df
        
    def _add_pattern_recognition(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add candlestick pattern recognition"""
        
        # Doji
        body_size = abs(df['close'] - df['open'])
        total_range = df['high'] - df['low']
        df['doji'] = (body_size <= total_range * 0.1).astype(int)
        
        # Hammer
        lower_shadow = df['open'].where(df['close'] >= df['open'], df['close']) - df['low']
        upper_shadow = df['high'] - df['close'].where(df['close'] >= df['open'], df['open'])
        df['hammer'] = ((lower_shadow > 2 * body_size) & 
                       (upper_shadow < body_size * 0.3) & 
                       (body_size > 0)).astype(int)
        
        # Engulfing patterns
        df['bullish_engulfing'] = ((df['close'] > df['open']) & 
                                  (df['close'].shift(1) < df['open'].shift(1)) &
                                  (df['open'] < df['close'].shift(1)) &
                                  (df['close'] > df['open'].shift(1))).astype(int)
                                  
        df['bearish_engulfing'] = ((df['close'] < df['open']) & 
                                  (df['close'].shift(1) > df['open'].shift(1)) &
                                  (df['open'] > df['close'].shift(1)) &
                                  (df['close'] < df['open'].shift(1))).astype(int)
        
        # Gap detection
        df['gap_up'] = (df['low'] > df['high'].shift(1)).astype(int)
        df['gap_down'] = (df['high'] < df['low'].shift(1)).astype(int)
        
        return df
        
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features"""
        
        # Price position within range
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Multi-period returns
        for period in [5, 10, 20]:
            if len(df) > period:
                df[f'returns_{period}'] = df['close'].pct_change(period)
                
        # Price channels
        for period in [20, 50]:
            if len(df) >= period:
                df[f'high_{period}'] = df['high'].rolling(period).max()
                df[f'low_{period}'] = df['low'].rolling(period).min()
                df[f'channel_position_{period}'] = (df['close'] - df[f'low_{period}']) / (df[f'high_{period}'] - df[f'low_{period}'])
                
        # Typical price
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        
        # Weighted close
        df['weighted_close'] = (df['high'] + df['low'] + 2 * df['close']) / 4
        
        return df
        
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        
        if not isinstance(df.index, pd.DatetimeIndex):
            return df
            
        # Time features
        df['hour'] = df.index.hour
        df['minute'] = df.index.minute
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        
        # Trading session features
        df['is_morning'] = (df.index.hour >= 9) & (df.index.hour < 12)
        df['is_afternoon'] = (df.index.hour >= 12) & (df.index.hour < 15)
        df['is_closing'] = (df.index.hour >= 15) & (df.index.minute >= 0)
        
        # First and last 30 minutes
        df['is_opening_30min'] = (df.index.hour == 9) & (df.index.minute < 30)
        df['is_closing_30min'] = (df.index.hour == 15) & (df.index.minute >= 0)
        
        return df
        
    @staticmethod
    def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
        
    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        
        return true_range.rolling(period).mean()
        
    def _calculate_from_date(self, interval: str, lookback_periods: int, to_date: datetime) -> datetime:
        """Calculate from_date based on interval and lookback periods"""
        
        # Map interval to minutes
        interval_minutes = {
            "minute": 1, "1minute": 1,
            "3minute": 3, "5minute": 5,
            "10minute": 10, "15minute": 15,
            "30minute": 30, "60minute": 60,
            "day": 1440
        }
        
        minutes = interval_minutes.get(interval, 5)
        
        # Calculate required days (with buffer for weekends/holidays)
        if interval == "day":
            required_days = int(lookback_periods * 1.5)
        else:
            # Assuming 6.5 hour trading day
            trading_minutes_per_day = 390
            required_days = int((lookback_periods * minutes / trading_minutes_per_day) * 1.5) + 5
            
        return to_date - timedelta(days=required_days)
        
    def _update_memory_cache(self, key: str, df: pd.DataFrame):
        """Update memory cache with LRU eviction"""
        
        # Add to cache
        self._memory_cache[key] = df
        self._cache_timestamps[key] = datetime.now()
        
        # Evict old entries if cache is too large
        if len(self._memory_cache) > self.max_memory_cache_size:
            # Find oldest entry
            oldest_key = min(self._cache_timestamps, key=self._cache_timestamps.get)
            del self._memory_cache[oldest_key]
            del self._cache_timestamps[oldest_key]
            
    def _save_to_disk_cache(self, key: str, df: pd.DataFrame):
        """Save DataFrame to disk cache"""
        try:
            cache_file = self.cache_dir / f"{key}.pkl"
            df.to_pickle(cache_file)
            logger.debug(f"Saved to disk cache: {key}")
        except Exception as e:
            logger.error(f"Error saving to disk cache: {e}")
            
    def _load_from_disk_cache(self, key: str, ignore_age: bool = False) -> Optional[pd.DataFrame]:
        """Load DataFrame from disk cache"""
        try:
            cache_file = self.cache_dir / f"{key}.pkl"
            
            if not cache_file.exists():
                return None
                
            # Check age unless ignored
            if not ignore_age:
                file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
                if file_age > timedelta(days=1):
                    return None
                    
            df = pd.read_pickle(cache_file)
            logger.debug(f"Loaded from disk cache: {key}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading from disk cache: {e}")
            return None
            
    def _cleanup_cache(self):
        """Remove old cache files"""
        try:
            cutoff_time = datetime.now() - timedelta(days=self.max_disk_cache_days)
            
            for cache_file in self.cache_dir.glob("*.pkl"):
                file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
                if file_time < cutoff_time:
                    cache_file.unlink()
                    logger.debug(f"Removed old cache file: {cache_file}")
                    
        except Exception as e:
            logger.error(f"Error cleaning cache: {e}")
            
    def clear_cache(self):
        """Clear all caches"""
        # Clear memory cache
        self._memory_cache.clear()
        self._cache_timestamps.clear()
        
        # Clear disk cache
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
            except Exception as e:
                logger.error(f"Error deleting cache file: {e}")
                
        logger.info("Cleared all data caches")