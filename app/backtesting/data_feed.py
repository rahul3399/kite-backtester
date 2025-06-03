# app/backtesting/data_feed.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from ..core.data_manager import DataManager

logger = logging.getLogger(__name__)

class DataFeed:
    """
    Data feed for backtesting engine
    Handles data loading, preprocessing, and synchronization
    """
    
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
        self._cache = {}
        self._symbol_mapping = {}
        
    async def load_data(self,
                       instrument_tokens: Dict[str, int],
                       timeframe: str,
                       start_date: datetime,
                       end_date: datetime,
                       lookback_periods: int = 100,
                       preload_all: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Load historical data for multiple instruments
        
        Args:
            instrument_tokens: Mapping of symbol to instrument token
            timeframe: Data timeframe (minute, 5minute, etc.)
            start_date: Start date for data
            end_date: End date for data
            lookback_periods: Number of periods to load before start_date
            preload_all: Whether to load all data upfront
            
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        
        data_dict = {}
        
        # Adjust start date for lookback
        adjusted_start = self._adjust_start_date(start_date, timeframe, lookback_periods)
        
        # Load data for each symbol
        tasks = []
        for symbol, token in instrument_tokens.items():
            self._symbol_mapping[token] = symbol
            
            if preload_all:
                task = self._load_symbol_data(
                    symbol, token, timeframe, adjusted_start, end_date
                )
                tasks.append((symbol, task))
            else:
                # For streaming mode, load minimal data
                task = self._load_symbol_data(
                    symbol, token, timeframe, adjusted_start, 
                    start_date + timedelta(days=1)
                )
                tasks.append((symbol, task))
        
        # Execute loading tasks concurrently
        for symbol, task in tasks:
            try:
                df = await task
                if df is not None and not df.empty:
                    # Ensure data is within requested range
                    df = df.loc[adjusted_start:end_date]
                    data_dict[symbol] = df
                    logger.info(f"Loaded {len(df)} bars for {symbol}")
                else:
                    logger.warning(f"No data available for {symbol}")
                    
            except Exception as e:
                logger.error(f"Error loading data for {symbol}: {e}")
                
        return data_dict
        
    async def _load_symbol_data(self,
                              symbol: str,
                              token: int,
                              timeframe: str,
                              start_date: datetime,
                              end_date: datetime) -> Optional[pd.DataFrame]:
        """Load data for a single symbol"""
        
        # Check cache first
        cache_key = f"{symbol}_{timeframe}_{start_date}_{end_date}"
        if cache_key in self._cache:
            return self._cache[cache_key]
            
        try:
            # Calculate required lookback
            lookback = self._calculate_lookback_periods(timeframe, start_date, end_date)
            
            # Load data from data manager
            df = await self.data_manager.get_historical_data(
                instrument_token=token,
                interval=timeframe,
                lookback_periods=lookback,
                to_date=end_date
            )
            
            if df is not None and not df.empty:
                # Validate and clean data
                df = self._validate_and_clean_data(df, symbol)
                
                # Add additional features
                df = self._add_features(df)
                
                # Cache the data
                self._cache[cache_key] = df
                
            return df
            
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {e}")
            return None
            
    def _adjust_start_date(self, start_date: datetime, timeframe: str, lookback_periods: int) -> datetime:
        """Adjust start date to account for lookback periods"""
        
        # Map timeframe to minutes
        timeframe_minutes = {
            "minute": 1, "1minute": 1,
            "3minute": 3, "5minute": 5,
            "10minute": 10, "15minute": 15,
            "30minute": 30, "60minute": 60,
            "day": 1440
        }
        
        minutes = timeframe_minutes.get(timeframe, 5)
        
        # Calculate additional days needed (assuming 6.5 hour trading day)
        trading_minutes_per_day = 390  # 6.5 hours
        bars_per_day = trading_minutes_per_day / minutes
        additional_days = int(lookback_periods / bars_per_day) + 5  # Add buffer
        
        return start_date - timedelta(days=additional_days)
        
    def _calculate_lookback_periods(self, timeframe: str, start_date: datetime, end_date: datetime) -> int:
        """Calculate number of periods to request"""
        
        # Map timeframe to expected bars per day
        bars_per_day = {
            "minute": 390, "1minute": 390,
            "3minute": 130, "5minute": 78,
            "10minute": 39, "15minute": 26,
            "30minute": 13, "60minute": 7,
            "day": 1
        }
        
        expected_bars = bars_per_day.get(timeframe, 78)
        days = (end_date - start_date).days + 1
        
        # Add 20% buffer for weekends/holidays
        return int(expected_bars * days * 1.2)
        
    def _validate_and_clean_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Validate and clean raw data"""
        
        if df.empty:
            return df
            
        # Check for required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.warning(f"Missing columns for {symbol}: {missing_columns}")
            return pd.DataFrame()
            
        # Remove duplicates
        df = df[~df.index.duplicated(keep='last')]
        
        # Sort by index
        df = df.sort_index()
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Validate OHLC relationships
        df = self._validate_ohlc(df, symbol)
        
        # Remove outliers
        df = self._remove_outliers(df)
        
        return df
        
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in data"""
        
        if df.empty:
            return df
            
        # Forward fill for small gaps (up to 5 bars)
        df = df.fillna(method='ffill', limit=5)
        
        # For remaining NaN values, use interpolation
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].interpolate(method='linear', limit_direction='both')
        
        # Drop rows with any remaining NaN in OHLCV
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        existing_required = [col for col in required_columns if col in df.columns]
        df = df.dropna(subset=existing_required)
        
        return df
        
    def _validate_ohlc(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Validate OHLC data relationships"""
        
        if df.empty:
            return df
            
        # High should be >= max(open, close, low)
        invalid_high = df['high'] < df[['open', 'close', 'low']].max(axis=1)
        if invalid_high.any():
            logger.warning(f"{symbol}: Found {invalid_high.sum()} bars with invalid high")
            df.loc[invalid_high, 'high'] = df.loc[invalid_high, ['open', 'close', 'low']].max(axis=1)
            
        # Low should be <= min(open, close, high)
        invalid_low = df['low'] > df[['open', 'close', 'high']].min(axis=1)
        if invalid_low.any():
            logger.warning(f"{symbol}: Found {invalid_low.sum()} bars with invalid low")
            df.loc[invalid_low, 'low'] = df.loc[invalid_low, ['open', 'close', 'high']].min(axis=1)
            
        # Volume should be non-negative
        df.loc[df['volume'] < 0, 'volume'] = 0
        
        return df
        
    def _remove_outliers(self, df: pd.DataFrame, n_std: float = 5.0) -> pd.DataFrame:
        """Remove price outliers using statistical methods"""
        
        if df.empty or len(df) < 20:
            return df
            
        # Calculate rolling statistics
        window = min(20, len(df) // 10)
        rolling_mean = df['close'].rolling(window=window, center=True).mean()
        rolling_std = df['close'].rolling(window=window, center=True).std()
        
        # Identify outliers
        lower_bound = rolling_mean - n_std * rolling_std
        upper_bound = rolling_mean + n_std * rolling_std
        
        outliers = (df['close'] < lower_bound) | (df['close'] > upper_bound)
        
        if outliers.any():
            logger.warning(f"Removing {outliers.sum()} outliers from data")
            # Instead of removing, cap the values
            df.loc[df['close'] < lower_bound, 'close'] = lower_bound[df['close'] < lower_bound]
            df.loc[df['close'] > upper_bound, 'close'] = upper_bound[df['close'] > upper_bound]
            
            # Adjust OHLC accordingly
            df.loc[outliers, 'high'] = df.loc[outliers, ['open', 'close']].max(axis=1)
            df.loc[outliers, 'low'] = df.loc[outliers, ['open', 'close']].min(axis=1)
            
        return df
        
    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add additional features for strategy use"""
        
        if df.empty:
            return df
            
        # Price-based features
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['weighted_close'] = (df['high'] + df['low'] + 2 * df['close']) / 4
        df['price_range'] = df['high'] - df['low']
        df['body_size'] = abs(df['close'] - df['open'])
        
        # Returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Volume features
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Volatility
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # Gap detection
        df['gap'] = df['open'] - df['close'].shift(1)
        df['gap_pct'] = df['gap'] / df['close'].shift(1) * 100
        
        # Session features (for intraday data)
        if hasattr(df.index, 'time'):
            df['hour'] = df.index.hour
            df['minute'] = df.index.minute
            df['is_opening'] = (df.index.hour == 9) & (df.index.minute < 30)
            df['is_closing'] = (df.index.hour >= 15) & (df.index.minute >= 0)
            
        return df
        
    def get_synchronized_data(self, 
                            data_dict: Dict[str, pd.DataFrame],
                            timestamp: datetime) -> Dict[str, pd.Series]:
        """
        Get synchronized data for all symbols at a specific timestamp
        
        Args:
            data_dict: Dictionary of symbol to DataFrame
            timestamp: Timestamp to get data for
            
        Returns:
            Dictionary of symbol to Series for the timestamp
        """
        
        synchronized_data = {}
        
        for symbol, df in data_dict.items():
            if timestamp in df.index:
                synchronized_data[symbol] = df.loc[timestamp]
            else:
                # Find nearest available timestamp
                nearest_idx = df.index.get_loc(timestamp, method='nearest')
                synchronized_data[symbol] = df.iloc[nearest_idx]
                
        return synchronized_data
        
    def get_lookback_window(self,
                          df: pd.DataFrame,
                          timestamp: datetime,
                          lookback: int) -> pd.DataFrame:
        """
        Get lookback window of data up to timestamp
        
        Args:
            df: DataFrame with time series data
            timestamp: Current timestamp
            lookback: Number of periods to look back
            
        Returns:
            DataFrame with lookback window
        """
        
        # Get data up to timestamp
        mask = df.index <= timestamp
        historical_data = df[mask]
        
        # Return last N periods
        return historical_data.tail(lookback)
        
    def resample_data(self,
                     df: pd.DataFrame,
                     target_timeframe: str) -> pd.DataFrame:
        """
        Resample data to different timeframe
        
        Args:
            df: Original DataFrame
            target_timeframe: Target timeframe (5T, 15T, 1H, 1D, etc.)
            
        Returns:
            Resampled DataFrame
        """
        
        if df.empty:
            return df
            
        # Define aggregation rules
        agg_rules = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        # Apply only to columns that exist
        rules = {col: rule for col, rule in agg_rules.items() if col in df.columns}
        
        # Resample
        resampled = df.resample(target_timeframe).agg(rules)
        
        # Remove any rows with all NaN
        resampled = resampled.dropna(how='all')
        
        # Recalculate features
        if not resampled.empty:
            resampled = self._add_features(resampled)
            
        return resampled
        
    def get_trading_calendar(self,
                           start_date: datetime,
                           end_date: datetime) -> List[datetime]:
        """
        Get list of trading days between dates
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            List of trading days
        """
        
        # For Indian markets (NSE/BSE)
        # This is simplified - in production, use proper holiday calendar
        
        trading_days = []
        current = start_date
        
        while current <= end_date:
            # Skip weekends (Saturday = 5, Sunday = 6)
            if current.weekday() < 5:
                trading_days.append(current)
            current += timedelta(days=1)
            
        return trading_days
        
    def clear_cache(self):
        """Clear data cache"""
        self._cache.clear()
        logger.info("Data cache cleared")