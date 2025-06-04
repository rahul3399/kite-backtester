from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import os
from .config import settings
from .api.routes import router
from .services.websocket_manager import WebSocketManager
from .services.kite_client import KiteClient
from .services.spread_calculator import SpreadCalculator
from .services.trading_strategy import PairTradingStrategy
from .utils.json_audit import JsonAuditLogger
from .utils.logger import setup_logging

# Setup logging
setup_logging(settings.log_level)
logger = logging.getLogger(__name__)

# Global instances
ws_manager = None
kite_client = None
spread_calculator = None
trading_strategy = None
audit_logger = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global ws_manager, kite_client, spread_calculator, trading_strategy, audit_logger
    
    try:
        # Initialize services
        logger.info("Initializing services...")
        
        # Check if we should skip WebSocket connection
        skip_websocket = os.getenv("SKIP_WEBSOCKET", "false").lower() == "true"
        
        try:
            kite_client = KiteClient(
                api_key=settings.kite_api_key,
                api_secret=settings.kite_api_secret,
                access_token=settings.kite_access_token
            )
            
            # Test Kite connection
            if not skip_websocket:
                profile = kite_client.kite.profile()
                logger.info(f"Kite connected successfully. User: {profile['user_name']}")
        except Exception as e:
            logger.error(f"Failed to initialize Kite client: {e}")
            logger.warning("Starting in offline mode. WebSocket features will be disabled.")
            skip_websocket = True
        
        ws_manager = WebSocketManager(
            api_key=settings.kite_api_key,
            access_token=settings.kite_access_token
        )
        
        spread_calculator = SpreadCalculator()
        audit_logger = JsonAuditLogger()
        
        trading_strategy = PairTradingStrategy(
            spread_calculator=spread_calculator,
            audit_logger=audit_logger
        )
        
        # Connect WebSocket only if credentials are valid
        if not skip_websocket:
            try:
                await ws_manager.connect()
                logger.info("WebSocket connected successfully")
            except Exception as e:
                logger.error(f"WebSocket connection failed: {e}")
                logger.warning("Continuing without WebSocket connection")
        else:
            logger.info("Skipping WebSocket connection (SKIP_WEBSOCKET=true)")
        
        # Store in app state
        app.state.ws_manager = ws_manager
        app.state.kite_client = kite_client
        app.state.spread_calculator = spread_calculator
        app.state.trading_strategy = trading_strategy
        app.state.audit_logger = audit_logger
        
        logger.info("All services initialized successfully")
        
        yield
        
    finally:
        # Cleanup
        logger.info("Shutting down services...")
        
        if trading_strategy:
            trading_strategy.stop_strategy()
        
        if ws_manager and ws_manager.is_connected:
            await ws_manager.disconnect()
        
        logger.info("Shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="Kite Options Trading API",
    description="Real-time options pair trading with Zerodha Kite",
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

# Include routes
app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    return {
        "message": "Kite Options Trading API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    ws_connected = ws_manager.is_connected if ws_manager else False
    
    # Try to check Kite connection
    kite_connected = False
    try:
        if kite_client:
            kite_client.kite.profile()
            kite_connected = True
    except:
        pass
    
    return {
        "status": "healthy",
        "kite_connected": kite_connected,
        "websocket_connected": ws_connected,
        "strategy_active": trading_strategy.is_active if trading_strategy else False
    }

@app.get("/auth/test")
async def test_auth():
    """Test Kite authentication"""
    try:
        if not kite_client:
            raise HTTPException(status_code=503, detail="Kite client not initialized")
        
        profile = kite_client.kite.profile()
        margins = kite_client.kite.margins()
        
        return {
            "status": "authenticated",
            "user": profile['user_name'],
            "email": profile['email'],
            "broker": profile['broker'],
            "margins": {
                "equity": margins.get('equity', {}),
                "commodity": margins.get('commodity', {})
            }
        }
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Authentication failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.log_level.lower()
    )