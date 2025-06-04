import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque
from ..models.trading import SpreadData, OHLCData
import logging

logger = logging.getLogger(__name__)

class SpreadCalculator:
    def __init__(self, lookback_period: int = 100):
        self.lookback_period = lookback_period
        self.spread_history: Dict[str, deque] = {}
        self.ma_short_period = 20
        self.ma_long_period = 50
        
    def calculate_spread(self, 
                        instrument1: str, 
                        price1: float, 
                        instrument2: str, 
                        price2: float) -> SpreadData:
        """Calculate spread between two instruments"""
        
        # Basic spread calculation
        spread = price1 - price2
        spread_percentage = (spread / price2) * 100 if price2 != 0 else 0
        
        # Create pair key
        pair_key = f"{instrument1}_{instrument2}"
        
        # Initialize history if needed
        if pair_key not in self.spread_history:
            self.spread_history[pair_key] = deque(maxlen=self.lookback_period)
        
        # Add to history
        self.spread_history[pair_key].append(spread)
        
        # Calculate moving averages and z-score
        ma_20, ma_50, z_score = self._calculate_indicators(pair_key)
        
        spread_data = SpreadData(
            timestamp=datetime.now(),
            instrument1=instrument1,
            instrument2=instrument2,
            price1=price1,
            price2=price2,
            spread=spread,
            spread_percentage=spread_percentage,
            ma_20=ma_20,
            ma_50=ma_50,
            z_score=z_score
        )
        
        return spread_data
    
    def _calculate_indicators(self, pair_key: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Calculate technical indicators for the spread"""
        history = list(self.spread_history[pair_key])
        
        if len(history) < self.ma_short_period:
            return None, None, None
        
        # Moving averages
        ma_20 = np.mean(history[-self.ma_short_period:])
        ma_50 = np.mean(history[-self.ma_long_period:]) if len(history) >= self.ma_long_period else None
        
        # Z-score
        if len(history) >= 20:
            mean = np.mean(history[-20:])
            std = np.std(history[-20:])
            z_score = (history[-1] - mean) / std if std != 0 else 0
        else:
            z_score = None
        
        return ma_20, ma_50, z_score
    
    def get_spread_statistics(self, instrument1: str, instrument2: str) -> Dict:
        """Get statistical analysis of spread"""
        pair_key = f"{instrument1}_{instrument2}"
        
        if pair_key not in self.spread_history or len(self.spread_history[pair_key]) < 2:
            return {}
        
        history = list(self.spread_history[pair_key])
        
        return {
            "mean": np.mean(history),
            "std": np.std(history),
            "min": np.min(history),
            "max": np.max(history),
            "current": history[-1],
            "percentile_25": np.percentile(history, 25),
            "percentile_75": np.percentile(history, 75),
            "data_points": len(history)
        }