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
from .api.v1 import backtest, paper_trade, strategies, reports

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
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
    
    # Initialize components
    settings = get_settings()
    kite_client = KiteClient()
    ws_manager = WebSocketManager(settings.KITE_API_KEY, settings.KITE_ACCESS_TOKEN)
    data_manager = DataManager(kite_client)
    backtest_engine = BacktestingEngine(data_manager)
    paper_engine = PaperTradingEngine(kite_client, ws_manager)
    
    # Auto-discover strategies
    strategy_registry.auto_discover_strategies()
    
    # Start paper trading engine
    await paper_engine.start()
    
    yield
    
    # Cleanup
    logger.info("Shutting down application...")
    await paper_engine.stop()

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
    allow_origins=["*"],
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
        "redoc": "/redoc"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "components": {
            "kite_client": kite_client is not None,
            "websocket": ws_manager is not None and ws_manager.is_connected,
            "paper_trading": paper_engine is not None and paper_engine.is_running
        }
    }

@app.websocket("/ws/market-data")
async def websocket_market_data(websocket: WebSocket):
    """WebSocket endpoint for real-time market data"""
    await websocket.accept()
    
    def on_tick(ticks):
        """Send ticks to WebSocket client"""
        try:
            asyncio.create_task(websocket.send_json({"type": "tick", "data": ticks}))
        except:
            pass
    
    # Register callback
    ws_manager.register_callback("tick", on_tick)
    
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.unregister_callback("tick", on_tick)