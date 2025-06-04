from kiteconnect import KiteConnect
from typing import List, Dict, Optional
import logging
from datetime import datetime
from ..config import settings

logger = logging.getLogger(__name__)

class KiteClient:
    def __init__(self, api_key: str, api_secret: str, access_token: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.access_token = access_token
        self.kite = KiteConnect(api_key=api_key)
        self.kite.set_access_token(access_token)
        
    def get_instruments(self, exchange: str = "NFO") -> List[Dict]:
        """Fetch all instruments for given exchange"""
        try:
            return self.kite.instruments(exchange)
        except Exception as e:
            logger.error(f"Failed to fetch instruments: {e}")
            raise
    
    def get_nifty_options(self, expiry_date: Optional[datetime] = None) -> List[Dict]:
        """Get all Nifty options for given expiry"""
        try:
            instruments = self.get_instruments("NFO")
            
            # Filter for Nifty options
            nifty_options = [
                inst for inst in instruments
                if inst['name'] == 'NIFTY' and inst['instrument_type'] in ['CE', 'PE']
            ]
            
            # Filter by expiry if provided
            if expiry_date:
                nifty_options = [
                    opt for opt in nifty_options
                    if opt['expiry'] == expiry_date.date()
                ]
            
            return nifty_options
        
        except Exception as e:
            logger.error(f"Failed to fetch Nifty options: {e}")
            raise
    
    def get_option_chain(self, expiry_date: datetime, strikes: Optional[List[float]] = None) -> Dict:
        """Get option chain for given expiry and strikes"""
        try:
            options = self.get_nifty_options(expiry_date)
            
            # Filter by strikes if provided
            if strikes:
                options = [
                    opt for opt in options
                    if opt['strike'] in strikes
                ]
            
            # Organize into option chain format
            option_chain = {}
            for opt in options:
                strike = opt['strike']
                if strike not in option_chain:
                    option_chain[strike] = {'CE': None, 'PE': None}
                
                option_chain[strike][opt['instrument_type']] = {
                    'instrument_token': opt['instrument_token'],
                    'trading_symbol': opt['tradingsymbol'],
                    'lot_size': opt['lot_size']
                }
            
            return option_chain
        
        except Exception as e:
            logger.error(f"Failed to fetch option chain: {e}")
            raise
    
    def get_quote(self, instruments: List[str]) -> Dict:
        """Get current quotes for instruments"""
        try:
            return self.kite.quote(instruments)
        except Exception as e:
            logger.error(f"Failed to fetch quotes: {e}")
            raise
    
    def get_ltp(self, instruments: List[str]) -> Dict:
        """Get last traded price for instruments"""
        try:
            return self.kite.ltp(instruments)
        except Exception as e:
            logger.error(f"Failed to fetch LTP: {e}")
            raise
    
    def place_order(self, 
                   trading_symbol: str,
                   exchange: str,
                   transaction_type: str,
                   quantity: int,
                   order_type: str = "MARKET",
                   product: str = "MIS",
                   **kwargs) -> str:
        """Place an order (for live trading - use with caution)"""
        try:
            order_id = self.kite.place_order(
                variety=self.kite.VARIETY_REGULAR,
                exchange=exchange,
                tradingsymbol=trading_symbol,
                transaction_type=transaction_type,
                quantity=quantity,
                order_type=order_type,
                product=product,
                **kwargs
            )
            logger.info(f"Order placed successfully: {order_id}")
            return order_id
        
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            raise
    
    def get_positions(self) -> Dict:
        """Get current positions"""
        try:
            return self.kite.positions()
        except Exception as e:
            logger.error(f"Failed to fetch positions: {e}")
            raise
    
    def get_margins(self) -> Dict:
        """Get account margins"""
        try:
            return self.kite.margins()
        except Exception as e:
            logger.error(f"Failed to fetch margins: {e}")
            raise