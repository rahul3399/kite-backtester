# app/core/kite_client.py
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import asyncio
from kiteconnect import KiteConnect, KiteTicker
import pandas as pd
from functools import lru_cache
import logging
import hashlib
import json

from ..config import get_settings

logger = logging.getLogger(__name__)

class KiteClient:
    """
    Wrapper for Kite Connect API with caching and error handling
    Provides unified interface for both historical data and live trading
    """
    
    def __init__(self):
        settings = get_settings()
        self.api_key = settings.KITE_API_KEY
        self.api_secret = settings.KITE_API_SECRET
        self.access_token = settings.KITE_ACCESS_TOKEN
        
        # Initialize KiteConnect
        self.kite = KiteConnect(api_key=self.api_key)
        
        # Set access token if available
        if self.access_token:
            self.kite.set_access_token(self.access_token)
            
        # Cache for instruments and other data
        self._instruments_cache = None
        self._last_instruments_update = None
        self._symbol_token_map = {}
        self._token_symbol_map = {}
        
        # Rate limiting
        self._last_api_call = datetime.now()
        self._api_call_delay = 0.25  # 250ms between calls
        
    def set_access_token(self, access_token: str):
        """Set or update access token"""
        self.access_token = access_token
        self.kite.set_access_token(access_token)
        logger.info("Access token updated")
        
    def get_login_url(self) -> str:
        """Get Kite login URL for authentication"""
        return self.kite.login_url()
        
    def generate_session(self, request_token: str) -> Dict[str, Any]:
        """
        Generate access token from request token
        
        Args:
            request_token: Token received after login
            
        Returns:
            Session data including access token
        """
        try:
            data = self.kite.generate_session(
                request_token=request_token,
                api_secret=self.api_secret
            )
            
            # Update access token
            self.set_access_token(data["access_token"])
            
            logger.info("Session generated successfully")
            return data
            
        except Exception as e:
            logger.error(f"Error generating session: {e}")
            raise
            
    def validate_session(self) -> bool:
        """Check if current session is valid"""
        try:
            # Try to fetch profile
            self.kite.profile()
            return True
        except Exception:
            return False
            
    @lru_cache(maxsize=1)
    def get_profile(self) -> Dict[str, Any]:
        """Get user profile information"""
        try:
            return self.kite.profile()
        except Exception as e:
            logger.error(f"Error fetching profile: {e}")
            raise
            
    def get_margins(self, segment: Optional[str] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Get account margins
        
        Args:
            segment: Trading segment (equity, commodity)
            
        Returns:
            Margin details
        """
        try:
            self._rate_limit()
            return self.kite.margins(segment)
        except Exception as e:
            logger.error(f"Error fetching margins: {e}")
            raise
            
    @lru_cache(maxsize=10)
    def get_instruments(self, exchange: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get instruments list with caching
        
        Args:
            exchange: Exchange name (NSE, BSE, NFO, etc.)
            
        Returns:
            List of instruments
        """
        # Refresh cache daily
        if (self._instruments_cache is None or 
            self._last_instruments_update is None or
            datetime.now() - self._last_instruments_update > timedelta(days=1)):
            
            try:
                logger.info(f"Fetching instruments for exchange: {exchange}")
                
                if exchange:
                    self._instruments_cache = self.kite.instruments(exchange)
                else:
                    self._instruments_cache = self.kite.instruments()
                    
                self._last_instruments_update = datetime.now()
                
                # Build symbol-token mappings
                self._build_symbol_mappings()
                
                logger.info(f"Loaded {len(self._instruments_cache)} instruments")
                
            except Exception as e:
                logger.error(f"Error fetching instruments: {e}")
                raise
                
        return self._instruments_cache
        
    def _build_symbol_mappings(self):
        """Build symbol to token and token to symbol mappings"""
        self._symbol_token_map.clear()
        self._token_symbol_map.clear()
        
        if self._instruments_cache:
            for instrument in self._instruments_cache:
                symbol = instrument['tradingsymbol']
                token = instrument['instrument_token']
                
                self._symbol_token_map[symbol] = token
                self._token_symbol_map[token] = symbol
                
    def get_instrument_token(self, symbol: str) -> Optional[int]:
        """Get instrument token for a symbol"""
        if not self._symbol_token_map:
            self.get_instruments()
            
        return self._symbol_token_map.get(symbol)
        
    def get_symbol_from_token(self, token: int) -> Optional[str]:
        """Get symbol from instrument token"""
        if not self._token_symbol_map:
            self.get_instruments()
            
        return self._token_symbol_map.get(token)
        
    def search_instruments(self, query: str, exchange: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search instruments by name or symbol
        
        Args:
            query: Search query
            exchange: Filter by exchange
            
        Returns:
            Matching instruments
        """
        instruments = self.get_instruments(exchange)
        query_lower = query.lower()
        
        results = []
        for instrument in instruments:
            if (query_lower in instrument['tradingsymbol'].lower() or
                query_lower in instrument.get('name', '').lower()):
                results.append(instrument)
                
        return results
        
    def get_historical_data(self,
                          instrument_token: int,
                          from_date: datetime,
                          to_date: datetime,
                          interval: str,
                          continuous: bool = False,
                          oi: bool = False) -> pd.DataFrame:
        """
        Fetch historical data and return as DataFrame
        
        Args:
            instrument_token: Instrument identifier
            from_date: Start date
            to_date: End date
            interval: Candle interval (minute, 5minute, day, etc.)
            continuous: Continuous data for futures
            oi: Include open interest
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            self._rate_limit()
            
            # Validate date range
            if to_date <= from_date:
                raise ValueError("to_date must be after from_date")
                
            # Kite has a limit on the amount of data per request
            max_days = self._get_max_days_for_interval(interval)
            
            all_data = []
            current_from = from_date
            
            while current_from < to_date:
                current_to = min(current_from + timedelta(days=max_days), to_date)
                
                logger.debug(f"Fetching data from {current_from} to {current_to}")
                
                data = self.kite.historical_data(
                    instrument_token=instrument_token,
                    from_date=current_from,
                    to_date=current_to,
                    interval=interval,
                    continuous=continuous,
                    oi=oi
                )
                
                if data:
                    all_data.extend(data)
                    
                current_from = current_to
                
            if not all_data:
                logger.warning(f"No data returned for token {instrument_token}")
                return pd.DataFrame()
                
            # Convert to DataFrame
            df = pd.DataFrame(all_data)
            
            # Process DataFrame
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df = df.sort_index()
                
                # Remove duplicates
                df = df[~df.index.duplicated(keep='last')]
                
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            raise
            
    def _get_max_days_for_interval(self, interval: str) -> int:
        """Get maximum days allowed per request for interval"""
        interval_limits = {
            "minute": 60,
            "3minute": 100,
            "5minute": 100,
            "10minute": 100,
            "15minute": 200,
            "30minute": 200,
            "60minute": 400,
            "day": 2000
        }
        return interval_limits.get(interval, 100)
        
    def get_quote(self, instruments: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Get real-time quotes for instruments
        
        Args:
            instruments: Single instrument or list of instruments
            
        Returns:
            Quote data
        """
        try:
            self._rate_limit()
            
            if isinstance(instruments, str):
                instruments = [instruments]
                
            # Convert symbols to exchange:symbol format
            formatted_instruments = []
            for inst in instruments:
                if ':' not in inst:
                    # Try to find exchange from instruments list
                    for cached_inst in self.get_instruments():
                        if cached_inst['tradingsymbol'] == inst:
                            formatted_inst = f"{cached_inst['exchange']}:{inst}"
                            formatted_instruments.append(formatted_inst)
                            break
                else:
                    formatted_instruments.append(inst)
                    
            return self.kite.quote(formatted_instruments)
            
        except Exception as e:
            logger.error(f"Error fetching quotes: {e}")
            raise
            
    def get_ltp(self, instruments: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Get last traded price for instruments
        
        Args:
            instruments: Single instrument or list of instruments
            
        Returns:
            LTP data
        """
        try:
            self._rate_limit()
            
            if isinstance(instruments, str):
                instruments = [instruments]
                
            # Format instruments
            formatted_instruments = []
            for inst in instruments:
                if ':' not in inst:
                    # Add exchange prefix
                    for cached_inst in self.get_instruments():
                        if cached_inst['tradingsymbol'] == inst:
                            formatted_inst = f"{cached_inst['exchange']}:{inst}"
                            formatted_instruments.append(formatted_inst)
                            break
                else:
                    formatted_instruments.append(inst)
                    
            return self.kite.ltp(formatted_instruments)
            
        except Exception as e:
            logger.error(f"Error fetching LTP: {e}")
            raise
            
    def place_order(self,
                   variety: str,
                   exchange: str,
                   tradingsymbol: str,
                   transaction_type: str,
                   quantity: int,
                   product: str,
                   order_type: str,
                   price: Optional[float] = None,
                   validity: Optional[str] = None,
                   disclosed_quantity: Optional[int] = None,
                   trigger_price: Optional[float] = None,
                   squareoff: Optional[float] = None,
                   stoploss: Optional[float] = None,
                   trailing_stoploss: Optional[float] = None,
                   tag: Optional[str] = None) -> str:
        """
        Place an order
        
        Returns:
            Order ID
        """
        try:
            self._rate_limit()
            
            order_id = self.kite.place_order(
                variety=variety,
                exchange=exchange,
                tradingsymbol=tradingsymbol,
                transaction_type=transaction_type,
                quantity=quantity,
                product=product,
                order_type=order_type,
                price=price,
                validity=validity,
                disclosed_quantity=disclosed_quantity,
                trigger_price=trigger_price,
                squareoff=squareoff,
                stoploss=stoploss,
                trailing_stoploss=trailing_stoploss,
                tag=tag
            )
            
            logger.info(f"Order placed successfully: {order_id}")
            return order_id
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            raise
            
    def modify_order(self,
                    variety: str,
                    order_id: str,
                    parent_order_id: Optional[str] = None,
                    quantity: Optional[int] = None,
                    price: Optional[float] = None,
                    order_type: Optional[str] = None,
                    trigger_price: Optional[float] = None,
                    validity: Optional[str] = None,
                    disclosed_quantity: Optional[int] = None) -> Dict[str, Any]:
        """Modify an existing order"""
        try:
            self._rate_limit()
            
            return self.kite.modify_order(
                variety=variety,
                order_id=order_id,
                parent_order_id=parent_order_id,
                quantity=quantity,
                price=price,
                order_type=order_type,
                trigger_price=trigger_price,
                validity=validity,
                disclosed_quantity=disclosed_quantity
            )
            
        except Exception as e:
            logger.error(f"Error modifying order: {e}")
            raise
            
    def cancel_order(self, variety: str, order_id: str, parent_order_id: Optional[str] = None) -> Dict[str, Any]:
        """Cancel an order"""
        try:
            self._rate_limit()
            
            return self.kite.cancel_order(
                variety=variety,
                order_id=order_id,
                parent_order_id=parent_order_id
            )
            
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            raise
            
    def get_orders(self) -> List[Dict[str, Any]]:
        """Get list of orders for the day"""
        try:
            self._rate_limit()
            return self.kite.orders()
        except Exception as e:
            logger.error(f"Error fetching orders: {e}")
            raise
            
    def get_order_history(self, order_id: str) -> List[Dict[str, Any]]:
        """Get history of an order"""
        try:
            self._rate_limit()
            return self.kite.order_history(order_id)
        except Exception as e:
            logger.error(f"Error fetching order history: {e}")
            raise
            
    def get_trades(self) -> List[Dict[str, Any]]:
        """Get list of trades for the day"""
        try:
            self._rate_limit()
            return self.kite.trades()
        except Exception as e:
            logger.error(f"Error fetching trades: {e}")
            raise
            
    def get_positions(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get positions"""
        try:
            self._rate_limit()
            return self.kite.positions()
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            raise
            
    def get_holdings(self) -> List[Dict[str, Any]]:
        """Get holdings"""
        try:
            self._rate_limit()
            return self.kite.holdings()
        except Exception as e:
            logger.error(f"Error fetching holdings: {e}")
            raise
            
    def _rate_limit(self):
        """Implement rate limiting to avoid API throttling"""
        elapsed = (datetime.now() - self._last_api_call).total_seconds()
        if elapsed < self._api_call_delay:
            sleep_time = self._api_call_delay - elapsed
            asyncio.sleep(sleep_time)
        self._last_api_call = datetime.now()