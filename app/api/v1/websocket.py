# app/api/v1/websocket.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Depends, Query
from typing import Dict, List, Set, Optional, Any
import asyncio
import json
import logging
from datetime import datetime
from collections import defaultdict

from ...api.models.requests import WebSocketSubscribeRequest
from ...api.models.responses import WebSocketMessage, MarketDataTick
from ...core.websocket_manager import WebSocketManager

router = APIRouter()
logger = logging.getLogger(__name__)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.subscriptions: Dict[str, Set[str]] = defaultdict(set)  # client_id -> symbols
        self.client_modes: Dict[str, str] = {}  # client_id -> mode (ltp/quote/full)
        
    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept new WebSocket connection"""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected")
        
    def disconnect(self, client_id: str):
        """Remove WebSocket connection"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            del self.subscriptions[client_id]
            if client_id in self.client_modes:
                del self.client_modes[client_id]
            logger.info(f"Client {client_id} disconnected")
            
    async def send_personal_message(self, message: dict, client_id: str):
        """Send message to specific client"""
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_json(message)
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")
                
    async def broadcast(self, message: dict, symbol: Optional[str] = None):
        """Broadcast message to all connected clients"""
        # If symbol is specified, only send to clients subscribed to that symbol
        for client_id, websocket in self.active_connections.items():
            if symbol is None or symbol in self.subscriptions.get(client_id, set()):
                try:
                    await websocket.send_json(message)
                except Exception as e:
                    logger.error(f"Error broadcasting to {client_id}: {e}")

# Global connection manager
manager = ConnectionManager()

@router.websocket("/stream")
async def websocket_endpoint(
    websocket: WebSocket,
    client_id: str = Query(..., description="Unique client identifier")
):
    """
    WebSocket endpoint for real-time market data and updates
    
    Message format:
    ```json
    {
        "type": "subscribe|unsubscribe|ping",
        "data": {
            "symbols": ["SYMBOL1", "SYMBOL2"],
            "mode": "ltp|quote|full"
        }
    }
    ```
    """
    await manager.connect(websocket, client_id)
    
    # Send welcome message
    await manager.send_personal_message({
        "type": "connected",
        "data": {
            "client_id": client_id,
            "timestamp": datetime.now().isoformat(),
            "message": "Connected to Kite Trading System WebSocket"
        }
    }, client_id)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                await handle_client_message(client_id, message)
            except json.JSONDecodeError:
                await manager.send_personal_message({
                    "type": "error",
                    "data": {"message": "Invalid JSON format"}
                }, client_id)
            except Exception as e:
                await manager.send_personal_message({
                    "type": "error",
                    "data": {"message": str(e)}
                }, client_id)
                
    except WebSocketDisconnect:
        manager.disconnect(client_id)
        # Unsubscribe from all symbols
        await handle_client_disconnect(client_id)

async def handle_client_message(client_id: str, message: dict):
    """Handle incoming client messages"""
    msg_type = message.get("type")
    data = message.get("data", {})
    
    if msg_type == "subscribe":
        await handle_subscribe(client_id, data)
    elif msg_type == "unsubscribe":
        await handle_unsubscribe(client_id, data)
    elif msg_type == "ping":
        await handle_ping(client_id)
    elif msg_type == "get_positions":
        await handle_get_positions(client_id)
    elif msg_type == "get_orders":
        await handle_get_orders(client_id)
    else:
        await manager.send_personal_message({
            "type": "error",
            "data": {"message": f"Unknown message type: {msg_type}"}
        }, client_id)

async def handle_subscribe(client_id: str, data: dict):
    """Handle subscription request"""
    symbols = data.get("symbols", [])
    mode = data.get("mode", "full")
    
    if not symbols:
        await manager.send_personal_message({
            "type": "error",
            "data": {"message": "No symbols provided"}
        }, client_id)
        return
    
    # Update subscriptions
    manager.subscriptions[client_id].update(symbols)
    manager.client_modes[client_id] = mode
    
    # Subscribe to Kite WebSocket
    from ...main import ws_manager
    
    # Get instrument tokens for symbols
    tokens = await get_instrument_tokens(symbols)
    
    if tokens:
        ws_manager.subscribe(tokens, mode)
    
    # Send confirmation
    await manager.send_personal_message({
        "type": "subscribed",
        "data": {
            "symbols": symbols,
            "mode": mode,
            "timestamp": datetime.now().isoformat()
        }
    }, client_id)
    
    logger.info(f"Client {client_id} subscribed to {symbols} in {mode} mode")

async def handle_unsubscribe(client_id: str, data: dict):
    """Handle unsubscription request"""
    symbols = data.get("symbols", [])
    
    if not symbols:
        # Unsubscribe from all
        symbols = list(manager.subscriptions[client_id])
    
    # Update subscriptions
    for symbol in symbols:
        manager.subscriptions[client_id].discard(symbol)
    
    # Check if any other clients are subscribed to these symbols
    symbols_to_unsubscribe = []
    for symbol in symbols:
        still_subscribed = any(
            symbol in subs 
            for cid, subs in manager.subscriptions.items() 
            if cid != client_id
        )
        if not still_subscribed:
            symbols_to_unsubscribe.append(symbol)
    
    # Unsubscribe from Kite WebSocket if no other clients need the data
    if symbols_to_unsubscribe:
        from ...main import ws_manager
        tokens = await get_instrument_tokens(symbols_to_unsubscribe)
        if tokens:
            ws_manager.unsubscribe(tokens)
    
    # Send confirmation
    await manager.send_personal_message({
        "type": "unsubscribed",
        "data": {
            "symbols": symbols,
            "timestamp": datetime.now().isoformat()
        }
    }, client_id)

async def handle_ping(client_id: str):
    """Handle ping message"""
    await manager.send_personal_message({
        "type": "pong",
        "data": {
            "timestamp": datetime.now().isoformat()
        }
    }, client_id)

async def handle_get_positions(client_id: str):
    """Send current positions to client"""
    from ...main import paper_engine
    
    positions = paper_engine.virtual_broker.get_positions()
    
    position_data = []
    for symbol, position in positions.items():
        current_price = paper_engine.virtual_broker.market_prices.get(
            symbol, position["avg_price"]
        )
        unrealized_pnl = (current_price - position["avg_price"]) * position["quantity"]
        
        position_data.append({
            "symbol": symbol,
            "quantity": position["quantity"],
            "avg_price": position["avg_price"],
            "current_price": current_price,
            "unrealized_pnl": unrealized_pnl,
            "unrealized_pnl_pct": (unrealized_pnl / (position["avg_price"] * position["quantity"])) * 100
        })
    
    await manager.send_personal_message({
        "type": "positions",
        "data": {
            "positions": position_data,
            "timestamp": datetime.now().isoformat()
        }
    }, client_id)

async def handle_get_orders(client_id: str):
    """Send recent orders to client"""
    from ...main import paper_engine
    
    orders = paper_engine.virtual_broker.orders[-20:]  # Last 20 orders
    
    await manager.send_personal_message({
        "type": "orders",
        "data": {
            "orders": orders,
            "timestamp": datetime.now().isoformat()
        }
    }, client_id)

async def handle_client_disconnect(client_id: str):
    """Handle client disconnect"""
    # Get symbols that might need unsubscription
    symbols = list(manager.subscriptions.get(client_id, set()))
    
    if symbols:
        # Check if any other clients are subscribed
        symbols_to_unsubscribe = []
        for symbol in symbols:
            still_subscribed = any(
                symbol in subs 
                for cid, subs in manager.subscriptions.items() 
                if cid != client_id
            )
            if not still_subscribed:
                symbols_to_unsubscribe.append(symbol)
        
        # Unsubscribe from Kite WebSocket
        if symbols_to_unsubscribe:
            from ...main import ws_manager
            tokens = await get_instrument_tokens(symbols_to_unsubscribe)
            if tokens:
                ws_manager.unsubscribe(tokens)

async def get_instrument_tokens(symbols: List[str]) -> List[int]:
    """Get instrument tokens for symbols"""
    # In production, this would map symbols to actual Kite instrument tokens
    # For now, using placeholder tokens
    tokens = []
    for i, symbol in enumerate(symbols):
        tokens.append(1000 + i)
    return tokens

# Market data broadcast function (called by WebSocket manager)
async def broadcast_market_data(ticks: List[Dict[str, Any]]):
    """Broadcast market data to subscribed clients"""
    for tick in ticks:
        # Map instrument token to symbol
        symbol = get_symbol_from_token(tick['instrument_token'])
        if not symbol:
            continue
        
        # Create market data message
        market_data = MarketDataTick(
            symbol=symbol,
            last_price=tick.get('last_price', 0),
            last_quantity=tick.get('last_quantity', 0),
            buy_quantity=tick.get('buy_quantity', 0),
            sell_quantity=tick.get('sell_quantity', 0),
            volume=tick.get('volume', 0),
            bid_price=tick.get('depth', {}).get('buy', [{}])[0].get('price', 0),
            ask_price=tick.get('depth', {}).get('sell', [{}])[0].get('price', 0),
            open=tick.get('ohlc', {}).get('open', 0),
            high=tick.get('ohlc', {}).get('high', 0),
            low=tick.get('ohlc', {}).get('low', 0),
            close=tick.get('ohlc', {}).get('close', 0),
            timestamp=datetime.now()
        )
        
        # Broadcast to subscribed clients
        message = WebSocketMessage(
            type="tick",
            data=market_data.dict()
        )
        
        await manager.broadcast(message.dict(), symbol)

def get_symbol_from_token(token: int) -> Optional[str]:
    """Map instrument token to symbol"""
    # In production, maintain a proper mapping
    # For now, using placeholder logic
    token_symbol_map = {
        1000: "NIFTY",
        1001: "BANKNIFTY",
        1002: "RELIANCE",
        1003: "TCS",
        # Add more mappings
    }
    return token_symbol_map.get(token)

# Additional WebSocket endpoints for specific data streams

@router.websocket("/stream/positions")
async def websocket_positions(
    websocket: WebSocket,
    client_id: str = Query(..., description="Unique client identifier")
):
    """WebSocket endpoint for real-time position updates"""
    await websocket.accept()
    
    try:
        while True:
            # Send position updates every 5 seconds
            await asyncio.sleep(5)
            
            from ...main import paper_engine
            positions = paper_engine.virtual_broker.get_positions()
            
            position_data = []
            for symbol, position in positions.items():
                current_price = paper_engine.virtual_broker.market_prices.get(
                    symbol, position["avg_price"]
                )
                unrealized_pnl = (current_price - position["avg_price"]) * position["quantity"]
                
                position_data.append({
                    "symbol": symbol,
                    "quantity": position["quantity"],
                    "avg_price": position["avg_price"],
                    "current_price": current_price,
                    "unrealized_pnl": unrealized_pnl,
                    "timestamp": datetime.now().isoformat()
                })
            
            await websocket.send_json({
                "type": "position_update",
                "data": position_data
            })
            
    except WebSocketDisconnect:
        logger.info(f"Position stream client {client_id} disconnected")

@router.websocket("/stream/trades")
async def websocket_trades(
    websocket: WebSocket,
    client_id: str = Query(..., description="Unique client identifier"),
    strategy_id: Optional[str] = Query(None, description="Filter by strategy ID")
):
    """WebSocket endpoint for real-time trade updates"""
    await websocket.accept()
    
    last_trade_count = 0
    
    try:
        while True:
            await asyncio.sleep(1)  # Check for new trades every second
            
            from ...main import paper_engine
            trades = paper_engine.virtual_broker.trades
            
            # Filter by strategy if specified
            if strategy_id:
                trades = [t for t in trades if t.get("strategy_id") == strategy_id]
            
            # Send only new trades
            if len(trades) > last_trade_count:
                new_trades = trades[last_trade_count:]
                last_trade_count = len(trades)
                
                for trade in new_trades:
                    await websocket.send_json({
                        "type": "trade",
                        "data": {
                            "trade_id": trade.get("order_id"),
                            "symbol": trade["symbol"],
                            "side": trade["side"],
                            "quantity": trade["quantity"],
                            "price": trade["price"],
                            "timestamp": trade["timestamp"].isoformat() if isinstance(trade["timestamp"], datetime) else trade["timestamp"],
                            "pnl": trade.get("pnl"),
                            "strategy_id": trade.get("strategy_id")
                        }
                    })
                    
    except WebSocketDisconnect:
        logger.info(f"Trade stream client {client_id} disconnected")

@router.websocket("/stream/performance")
async def websocket_performance(
    websocket: WebSocket,
    client_id: str = Query(..., description="Unique client identifier")
):
    """WebSocket endpoint for real-time performance metrics"""
    await websocket.accept()
    
    try:
        while True:
            await asyncio.sleep(10)  # Update every 10 seconds
            
            from ...main import paper_engine
            performance = paper_engine.virtual_broker.get_performance_summary()
            
            await websocket.send_json({
                "type": "performance_update",
                "data": {
                    "portfolio_value": performance["portfolio_value"],
                    "total_pnl": performance["total_pnl"],
                    "total_pnl_pct": (performance["total_pnl"] / performance["initial_capital"]) * 100,
                    "positions_count": performance["positions_count"],
                    "win_rate": performance["win_rate"],
                    "timestamp": datetime.now().isoformat()
                }
            })
            
    except WebSocketDisconnect:
        logger.info(f"Performance stream client {client_id} disconnected")

# Initialize WebSocket handlers
async def initialize_websocket_handlers():
    """Initialize WebSocket handlers with Kite WebSocket manager"""
    from ...main import ws_manager
    
    # Register callback for market data
    ws_manager.register_callback("tick", broadcast_market_data)
    
    logger.info("WebSocket handlers initialized")