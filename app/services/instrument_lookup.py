from typing import Dict, List, Optional, Tuple
import pandas as pd
from datetime import datetime
import logging
from functools import lru_cache
from ..services.kite_client import KiteClient

logger = logging.getLogger(__name__)

class InstrumentLookup:
    def __init__(self, kite_client: KiteClient):
        self.kite_client = kite_client
        self._instruments_cache = {}
        self._last_refresh = None
        
    def refresh_instruments(self, exchange: str = "NFO"):
        """Refresh instruments cache"""
        try:
            logger.info(f"Refreshing instruments for {exchange}")
            instruments = self.kite_client.get_instruments(exchange)
            
            # Create lookup dictionaries
            self._instruments_cache[exchange] = {
                'by_symbol': {},
                'by_token': {},
                'dataframe': pd.DataFrame(instruments)
            }
            
            for inst in instruments:
                symbol = inst['tradingsymbol']
                token = inst['instrument_token']
                
                self._instruments_cache[exchange]['by_symbol'][symbol] = inst
                self._instruments_cache[exchange]['by_token'][token] = inst
            
            self._last_refresh = datetime.now()
            logger.info(f"Cached {len(instruments)} instruments for {exchange}")
            
        except Exception as e:
            logger.error(f"Failed to refresh instruments: {e}")
            raise
    
    def get_instrument_by_symbol(self, symbol: str, exchange: str = "NFO") -> Optional[Dict]:
        """Get instrument details by trading symbol"""
        if exchange not in self._instruments_cache:
            self.refresh_instruments(exchange)
        
        return self._instruments_cache[exchange]['by_symbol'].get(symbol)
    
    def get_instrument_token(self, symbol: str, exchange: str = "NFO") -> Optional[int]:
        """Get instrument token by trading symbol"""
        instrument = self.get_instrument_by_symbol(symbol, exchange)
        return instrument['instrument_token'] if instrument else None
    
    def search_instruments(self, query: str, exchange: str = "NFO") -> List[Dict]:
        """Search instruments by partial symbol match"""
        if exchange not in self._instruments_cache:
            self.refresh_instruments(exchange)
        
        df = self._instruments_cache[exchange]['dataframe']
        
        # Search in trading symbol
        mask = df['tradingsymbol'].str.contains(query.upper(), case=False)
        results = df[mask].to_dict('records')
        
        return results[:20]  # Limit to 20 results
    
    def get_nifty_option_token(self, strike: int, option_type: str, 
                              expiry: Optional[datetime] = None) -> Optional[int]:
        """Get Nifty option token by strike and type"""
        if 'NFO' not in self._instruments_cache:
            self.refresh_instruments('NFO')
        
        df = self._instruments_cache['NFO']['dataframe']
        
        # Filter for Nifty options
        mask = (df['name'] == 'NIFTY') & \
               (df['instrument_type'] == option_type.upper()) & \
               (df['strike'] == strike)
        
        if expiry:
            mask = mask & (df['expiry'] == expiry.date())
        else:
            # Get current/next expiry
            today = datetime.now().date()
            valid_expiries = df[df['expiry'] >= today]['expiry'].unique()
            if len(valid_expiries) > 0:
                next_expiry = min(valid_expiries)
                mask = mask & (df['expiry'] == next_expiry)
        
        results = df[mask]
        
        if not results.empty:
            return int(results.iloc[0]['instrument_token'])
        return None
    
    def parse_option_symbol(self, symbol: str) -> Optional[Dict]:
        """Parse option symbol to extract details
        Example: NIFTY24JAN25000CE -> {name: NIFTY, expiry: 24JAN, strike: 25000, type: CE}
        """
        try:
            # Common patterns for option symbols
            # NIFTY24JAN25000CE, BANKNIFTY24JAN45000PE
            
            import re
            
            # Pattern: NAME + DATE + STRIKE + TYPE
            pattern = r'^([A-Z]+)(\d{2}[A-Z]{3})(\d+)(CE|PE)$'
            match = re.match(pattern, symbol.upper())
            
            if match:
                name = match.group(1)
                expiry_str = match.group(2)
                strike = int(match.group(3))
                option_type = match.group(4)
                
                # Parse expiry date
                year = 2000 + int(expiry_str[:2])
                month_str = expiry_str[2:5]
                
                # Month mapping
                months = {
                    'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4,
                    'MAY': 5, 'JUN': 6, 'JUL': 7, 'AUG': 8,
                    'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
                }
                
                month = months.get(month_str)
                
                if month:
                    return {
                        'name': name,
                        'expiry_str': expiry_str,
                        'year': year,
                        'month': month,
                        'strike': strike,
                        'type': option_type
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to parse symbol {symbol}: {e}")
            return None
    
    def get_option_pairs(self, base_symbol: str, strikes: List[int], 
                        expiry: Optional[datetime] = None) -> List[Dict]:
        """Get CE-PE pairs for given strikes"""
        pairs = []
        
        for strike in strikes:
            ce_token = self.get_nifty_option_token(strike, 'CE', expiry)
            pe_token = self.get_nifty_option_token(strike, 'PE', expiry)
            
            if ce_token and pe_token:
                ce_inst = self._instruments_cache['NFO']['by_token'][ce_token]
                pe_inst = self._instruments_cache['NFO']['by_token'][pe_token]
                
                pairs.append({
                    'strike': strike,
                    'ce': {
                        'symbol': ce_inst['tradingsymbol'],
                        'token': ce_token
                    },
                    'pe': {
                        'symbol': pe_inst['tradingsymbol'],
                        'token': pe_token
                    }
                })
        
        return pairs
    
    def get_atm_strikes(self, spot_price: float, num_strikes: int = 5) -> List[int]:
        """Get ATM and nearby strikes"""
        # Round to nearest 50 for Nifty
        atm_strike = round(spot_price / 50) * 50
        
        strikes = []
        for i in range(-num_strikes//2, num_strikes//2 + 1):
            strikes.append(atm_strike + (i * 50))
        
        return strikes