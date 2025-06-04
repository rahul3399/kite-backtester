from fastapi import Depends, HTTPException
from typing import Optional
from ..services.websocket_manager import WebSocketManager
from ..services.kite_client import KiteClient
from ..services.spread_calculator import SpreadCalculator
from ..services.trading_strategy import PairTradingStrategy
from ..utils.json_audit import JsonAuditLogger
from ..services.instrument_lookup import InstrumentLookup

def get_ws_manager() -> WebSocketManager:
    """Get WebSocket manager instance"""
    from ..main import ws_manager
    if not ws_manager:
        raise HTTPException(status_code=503, detail="WebSocket manager not initialized")
    return ws_manager

def get_kite_client() -> KiteClient:
    """Get Kite client instance"""
    from ..main import kite_client
    if not kite_client:
        raise HTTPException(status_code=503, detail="Kite client not initialized")
    return kite_client

def get_spread_calculator() -> SpreadCalculator:
    """Get spread calculator instance"""
    from ..main import spread_calculator
    if not spread_calculator:
        raise HTTPException(status_code=503, detail="Spread calculator not initialized")
    return spread_calculator

def get_strategy() -> PairTradingStrategy:
    """Get trading strategy instance"""
    from ..main import trading_strategy
    if not trading_strategy:
        raise HTTPException(status_code=503, detail="Trading strategy not initialized")
    return trading_strategy

def get_audit_logger() -> JsonAuditLogger:
    """Get audit logger instance"""
    from ..main import audit_logger
    if not audit_logger:
        raise HTTPException(status_code=503, detail="Audit logger not initialized")
    return audit_logger

# You can also create dependency functions for common validations
def validate_instrument_token(instrument_token: int) -> int:
    """Validate instrument token"""
    if instrument_token <= 0:
        raise HTTPException(status_code=400, detail="Invalid instrument token")
    return instrument_token

def validate_trading_pair(instrument1: str, instrument2: str) -> tuple:
    """Validate trading pair"""
    if not instrument1 or not instrument2:
        raise HTTPException(status_code=400, detail="Both instruments must be specified")
    
    if instrument1 == instrument2:
        raise HTTPException(status_code=400, detail="Instruments must be different")
    
    # Add more validation logic here (e.g., check if they're valid Nifty options)
    # Example: Check if they follow the pattern "NIFTY{date}{strike}{CE/PE}"
    
    return (instrument1, instrument2)

# Optional: Add authentication dependencies if needed
# def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
#     """Get current authenticated user"""
#     # Implement authentication logic
#     pass