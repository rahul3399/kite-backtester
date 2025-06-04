# app/dependencies.py
from typing import Generator, Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from .database.session import SessionLocal
from .config import get_settings, Settings
from .core.kite_client import KiteClient

# Security
security = HTTPBearer()

def get_db() -> Generator:
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_kite_client() -> KiteClient:
    """Get Kite client instance"""
    from .main import kite_client
    if not kite_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Kite client not initialized"
        )
    return kite_client

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """Get current user from token"""
    # In production, implement proper JWT validation
    # For now, this is a placeholder
    return {"user_id": "demo_user", "token": credentials.credentials}

def get_settings_dependency() -> Settings:
    """Get settings instance"""
    return get_settings()