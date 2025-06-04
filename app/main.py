# app/main.py
from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from typing import Dict, List, Optional, Any
import asyncio

from .config import get_settings
from .core.kite_client import KiteClient
from .core.websocket_manager import WebSocketManager
from .core.data_manager import DataManager
from .backtesting.engine import BacktestingEngine
from .paper_trading.engine import PaperTradingEngine
from .strategies.registry import strategy_registry
from .database.session import init_db  # Added missing import
from .api.v1 import backtest, paper_trade, strategies, reports

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances - will be initialized in lifespan
kite_client = None
ws_manager = None
data_manager = None
backtest_engine = None
paper_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global kite_client, ws_manager, data_manager, backtest_engine, paper_engine
    
    logger.info("Starting application...")
    
    try:
        # Initialize database
        init_db()
        
        # Initialize components
        settings = get_settings()
        kite_client = KiteClient()
        
        # Only initialize WebSocket if we have valid credentials
        if settings.KITE_ACCESS_TOKEN:
            ws_manager = WebSocketManager(settings.KITE_API_KEY, settings.KITE_ACCESS_TOKEN)
        else:
            logger.warning("No Kite access token found - WebSocket will not be available")
            ws_manager = None
            
        data_manager = DataManager(kite_client)
        backtest_engine = BacktestingEngine(data_manager)
        
        # Initialize paper trading engine with proper error handling
        try:
            paper_engine = PaperTradingEngine(kite_client, ws_manager, data_manager)
            # Start paper trading engine
            await paper_engine.start()
        except Exception as e:
            logger.error(f"Failed to initialize paper trading engine: {e}")
            paper_engine = None
        
        # Auto-discover strategies
        try:
            discovered = strategy_registry.auto_discover_strategies()
            logger.info(f"Discovered strategies: {discovered}")
        except Exception as e:
            logger.error(f"Failed to discover strategies: {e}")
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise
    
    yield
    
    # Cleanup
    logger.info("Shutting down application...")
    if paper_engine:
        try:
            await paper_engine.stop()
        except Exception as e:
            logger.error(f"Error stopping paper engine: {e}")

# Create FastAPI app
app = FastAPI(
    title="Kite Trading System",
    description="Professional backtesting and paper trading system for Kite Zerodha",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(backtest.router, prefix="/api/v1/backtest", tags=["Backtesting"])
app.include_router(paper_trade.router, prefix="/api/v1/paper-trade", tags=["Paper Trading"])
app.include_router(strategies.router, prefix="/api/v1/strategies", tags=["Strategies"])
app.include_router(reports.router, prefix="/api/v1/reports", tags=["Reports"])

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Kite Trading System API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global kite_client, ws_manager, paper_engine
    
    components = {
        "kite_client": kite_client is not None,
        "websocket": ws_manager is not None and (ws_manager.is_connected() if hasattr(ws_manager, 'is_connected') else False),
        "paper_trading": paper_engine is not None and (paper_engine.is_running if hasattr(paper_engine, 'is_running') else False),
        "database": True,  # Assume DB is healthy if app started
        "strategies": len(strategy_registry.list_strategies()) > 0
    }
    
    all_healthy = all(components.values())
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "components": components,
        "timestamp": asyncio.get_event_loop().time()
    }

@app.get("/api/v1/status")
async def get_system_status():
    """Get detailed system status"""
    global kite_client, ws_manager, data_manager, backtest_engine, paper_engine
    
    status = {
        "application": {
            "name": "Kite Trading System",
            "version": "1.0.0",
            "status": "running"
        },
        "components": {
            "kite_client": {
                "initialized": kite_client is not None,
                "authenticated": False  # Would need to check actual auth status
            },
            "websocket_manager": {
                "initialized": ws_manager is not None,
                "connected": ws_manager.is_connected() if ws_manager and hasattr(ws_manager, 'is_connected') else False
            },
            "data_manager": {
                "initialized": data_manager is not None
            },
            "backtest_engine": {
                "initialized": backtest_engine is not None
            },
            "paper_engine": {
                "initialized": paper_engine is not None,
                "running": paper_engine.is_running if paper_engine and hasattr(paper_engine, 'is_running') else False
            }
        },
        "strategies": {
            "total_registered": len(strategy_registry.list_strategies()),
            "available_strategies": strategy_registry.list_strategies(),
            "categories": strategy_registry._categories if hasattr(strategy_registry, '_categories') else {}
        }
    }
    
    return status

@app.websocket("/ws/market-data")
async def websocket_market_data(websocket: WebSocket):
    """WebSocket endpoint for real-time market data"""
    await websocket.accept()
    
    if not ws_manager:
        await websocket.send_json({
            "type": "error",
            "message": "WebSocket manager not available"
        })
        await websocket.close()
        return
    
    def on_tick(ticks):
        """Send ticks to WebSocket client"""
        try:
            asyncio.create_task(websocket.send_json({"type": "tick", "data": ticks}))
        except Exception as e:
            logger.error(f"Error sending WebSocket data: {e}")
    
    # Register callback
    ws_manager.register_callback("tick", on_tick)
    
    try:
        while True:
            # Keep connection alive and handle client messages
            try:
                message = await websocket.receive_text()
                # Handle client messages if needed
                logger.debug(f"Received WebSocket message: {message}")
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                break
    finally:
        # Cleanup
        if ws_manager:
            ws_manager.unregister_callback("tick", on_tick)

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {
        "error": "Not Found",
        "message": "The requested resource was not found",
        "status_code": 404
    }

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return {
        "error": "Internal Server Error",
        "message": "An internal server error occurred",
        "status_code": 500
    }

# Dependency providers for avoiding circular imports
def get_kite_client():
    """Get the global kite client instance"""
    if kite_client is None:
        raise HTTPException(status_code=503, detail="Kite client not initialized")
    return kite_client

def get_data_manager():
    """Get the global data manager instance"""
    if data_manager is None:
        raise HTTPException(status_code=503, detail="Data manager not initialized")
    return data_manager

def get_backtest_engine():
    """Get the global backtest engine instance"""
    if backtest_engine is None:
        raise HTTPException(status_code=503, detail="Backtest engine not initialized")
    return backtest_engine

def get_paper_engine():
    """Get the global paper trading engine instance"""
    if paper_engine is None:
        raise HTTPException(status_code=503, detail="Paper trading engine not initialized")
    return paper_engine

def get_websocket_manager():
    """Get the global websocket manager instance"""
    if ws_manager is None:
        raise HTTPException(status_code=503, detail="WebSocket manager not initialized")
    return ws_manager