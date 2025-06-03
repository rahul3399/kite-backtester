# app/api/v1/strategies.py
from fastapi import APIRouter, HTTPException, File, UploadFile
from typing import Dict, List, Optional, Any
import logging
import json

from ...api.models.requests import RegisterStrategyRequest
from ...api.models.responses import StrategyInfoResponse, StrategyListResponse
from ...strategies.registry import strategy_registry
from ...strategies.base_strategy import BaseStrategy

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/list", response_model=StrategyListResponse)
async def list_strategies():
    """
    List all available strategies
    
    Returns a list of all registered strategies with their metadata
    """
    
    strategies = []
    
    for name in strategy_registry.list_strategies():
        info = strategy_registry.get_strategy_info(name)
        
        # Count active instances
        from ..paper_trade import active_strategies
        active_instances = sum(1 for s in active_strategies.values() 
                             if s["name"] == name and s["status"] == "running")
        
        strategies.append(StrategyInfoResponse(
            name=name,
            class_name=info.get("class", ""),
            module=info.get("module", ""),
            description=info.get("docstring", "").strip() if info.get("docstring") else None,
            parameters={},  # Would need to extract from strategy class
            required_indicators=info.get("required_indicators", []),
            is_active=active_instances > 0,
            instances=active_instances
        ))
    
    return StrategyListResponse(
        strategies=strategies,
        total=len(strategies)
    )

@router.get("/info/{strategy_name}", response_model=StrategyInfoResponse)
async def get_strategy_info(strategy_name: str):
    """Get detailed information about a specific strategy"""
    
    info = strategy_registry.get_strategy_info(strategy_name)
    
    if not info:
        raise HTTPException(
            status_code=404,
            detail=f"Strategy '{strategy_name}' not found"
        )
    
    # Get strategy class to extract parameters
    strategy_class = strategy_registry.get_strategy_class(strategy_name)
    parameters = {}
    
    if strategy_class:
        # Try to extract default parameters
        try:
            # Create a dummy instance to get default parameters
            dummy_config = {
                "name": strategy_name,
                "symbols": ["DUMMY"],
                "parameters": {}
            }
            from ...strategies.base_strategy import StrategyConfig
            config = StrategyConfig(**dummy_config)
            instance = strategy_class(config)
            
            # Extract parameters from instance attributes
            if hasattr(instance, "fast_period"):
                parameters["fast_period"] = getattr(instance, "fast_period", None)
            if hasattr(instance, "slow_period"):
                parameters["slow_period"] = getattr(instance, "slow_period", None)
            # Add more parameter extraction as needed
            
        except Exception as e:
            logger.warning(f"Could not extract parameters for {strategy_name}: {e}")
    
    # Count active instances
    from ..paper_trade import active_strategies
    active_instances = sum(1 for s in active_strategies.values() 
                         if s["name"] == strategy_name and s["status"] == "running")
    
    return StrategyInfoResponse(
        name=strategy_name,
        class_name=info.get("class", ""),
        module=info.get("module", ""),
        description=info.get("docstring", "").strip() if info.get("docstring") else None,
        parameters=parameters,
        required_indicators=info.get("required_indicators", []),
        is_active=active_instances > 0,
        instances=active_instances
    )

@router.post("/register")
async def register_custom_strategy(request: RegisterStrategyRequest):
    """
    Register a custom strategy from code
    
    **Security Warning**: This endpoint executes arbitrary Python code.
    In production, implement proper sandboxing and security measures.
    """
    
    try:
        # Create a temporary module
        import types
        module = types.ModuleType(f"custom_{request.strategy_name}")
        
        # Add necessary imports to the module namespace
        module.__dict__['BaseStrategy'] = BaseStrategy
        module.__dict__['Signal'] = Signal
        module.__dict__['OrderSide'] = OrderSide
        module.__dict__['OrderType'] = OrderType
        module.__dict__['pd'] = pd
        module.__dict__['np'] = np
        module.__dict__['Optional'] = Optional
        module.__dict__['Dict'] = Dict
        module.__dict__['List'] = List
        module.__dict__['Any'] = Any
        
        # Execute the code
        exec(request.code, module.__dict__)
        
        # Find the strategy class
        strategy_class = None
        
        for name, obj in module.__dict__.items():
            if (isinstance(obj, type) and 
                issubclass(obj, BaseStrategy) and 
                obj is not BaseStrategy):
                strategy_class = obj
                break
        
        if not strategy_class:
            raise ValueError("No valid strategy class found in code. "
                           "Make sure your class inherits from BaseStrategy.")
        
        # Register the strategy
        strategy_registry.register(request.strategy_name, strategy_class)
        
        logger.info(f"Registered custom strategy: {request.strategy_name}")
        
        return {
            "message": f"Strategy '{request.strategy_name}' registered successfully",
            "strategy_name": request.strategy_name,
            "class_name": strategy_class.__name__
        }
        
    except SyntaxError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Syntax error in strategy code: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error registering strategy: {str(e)}"
        )

@router.post("/upload")
async def upload_strategy_file(
    file: UploadFile = File(..., description="Python file containing strategy implementation")
):
    """Upload a strategy file"""
    
    # Validate file extension
    if not file.filename.endswith('.py'):
        raise HTTPException(
            status_code=400,
            detail="Only Python (.py) files are allowed"
        )
    
    try:
        # Read file content
        content = await file.read()
        code = content.decode('utf-8')
        
        # Extract strategy name from filename
        strategy_name = file.filename.replace('.py', '')
        
        # Register the strategy
        request = RegisterStrategyRequest(
            strategy_name=strategy_name,
            code=code
        )
        
        return await register_custom_strategy(request)
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error processing file: {str(e)}"
        )

@router.get("/parameters/{strategy_name}")
async def get_strategy_parameters(strategy_name: str):
    """Get configurable parameters for a strategy"""
    
    strategy_class = strategy_registry.get_strategy_class(strategy_name)
    
    if not strategy_class:
        raise HTTPException(
            status_code=404,
            detail=f"Strategy '{strategy_name}' not found"
        )
    
    # Define parameter schemas for known strategies
    parameter_schemas = {
        "MovingAverageCrossStrategy": {
            "fast_period": {
                "type": "integer",
                "default": 20,
                "min": 1,
                "max": 200,
                "description": "Fast moving average period"
            },
            "slow_period": {
                "type": "integer",
                "default": 50,
                "min": 1,
                "max": 500,
                "description": "Slow moving average period"
            },
            "position_size_pct": {
                "type": "float",
                "default": 0.1,
                "min": 0.01,
                "max": 1.0,
                "description": "Position size as percentage of capital"
            }
        },
        "ArbitrageStrategy": {
            "spread_threshold": {
                "type": "float",
                "default": 0.01,
                "min": 0.0001,
                "max": 0.1,
                "description": "Minimum spread to trigger trade"
            },
            "position_ratio": {
                "type": "float",
                "default": 1.0,
                "min": 0.1,
                "max": 10.0,
                "description": "Position size ratio between instruments"
            }
        }
    }
    
    # Get parameter schema for this strategy
    schema = parameter_schemas.get(strategy_class.__name__, {})
    
    return {
        "strategy_name": strategy_name,
        "parameters": schema,
        "description": f"Configurable parameters for {strategy_name}"
    }

@router.get("/templates")
async def get_strategy_templates():
    """Get strategy code templates"""
    
    templates = {
        "basic": '''from app.strategies.base_strategy import BaseStrategy, Signal, OrderSide, OrderType
import pandas as pd
from typing import Optional, Dict, Any

class MyStrategy(BaseStrategy):
    """My custom trading strategy"""
    
    def initialize(self):
        """Initialize strategy parameters"""
        self.param1 = self.config.parameters.get('param1', 10)
        
    def on_data(self, symbol: str, data: pd.DataFrame) -> Optional[Signal]:
        """Generate trading signal from data"""
        
        # Your strategy logic here
        if len(data) < self.param1:
            return None
            
        # Example: Buy if price increases
        if data['close'].iloc[-1] > data['close'].iloc[-2]:
            return Signal(
                symbol=symbol,
                side=OrderSide.BUY,
                quantity=0,  # Will be calculated by position sizing
                order_type=OrderType.MARKET
            )
            
        return None
        
    def on_order_fill(self, order: Dict[str, Any]):
        """Handle order fill"""
        pass
        
    def on_position_update(self, position: Dict[str, Any]):
        """Handle position update"""
        pass
''',
        "indicator_based": '''from app.strategies.base_strategy import BaseStrategy, Signal, OrderSide, OrderType
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any

class IndicatorStrategy(BaseStrategy):
    """Strategy based on technical indicators"""
    
    def initialize(self):
        """Initialize strategy parameters"""
        self.rsi_period = self.config.parameters.get('rsi_period', 14)
        self.rsi_oversold = self.config.parameters.get('rsi_oversold', 30)
        self.rsi_overbought = self.config.parameters.get('rsi_overbought', 70)
        
    def on_data(self, symbol: str, data: pd.DataFrame) -> Optional[Signal]:
        """Generate signal based on RSI"""
        
        if len(data) < self.rsi_period + 1:
            return None
            
        # RSI is already calculated in the data
        current_rsi = data['rsi'].iloc[-1]
        prev_rsi = data['rsi'].iloc[-2]
        
        # Buy when RSI crosses above oversold
        if prev_rsi <= self.rsi_oversold and current_rsi > self.rsi_oversold:
            return Signal(
                symbol=symbol,
                side=OrderSide.BUY,
                quantity=0,
                order_type=OrderType.MARKET,
                metadata={"rsi": current_rsi}
            )
            
        # Sell when RSI crosses below overbought
        if prev_rsi >= self.rsi_overbought and current_rsi < self.rsi_overbought:
            if symbol in self.state.positions:
                return Signal(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    quantity=self.state.positions[symbol].get('quantity', 0),
                    order_type=OrderType.MARKET,
                    metadata={"rsi": current_rsi}
                )
                
        return None
        
    def on_order_fill(self, order: Dict[str, Any]):
        """Update positions on order fill"""
        symbol = order['symbol']
        if order['side'] == 'BUY':
            self.state.positions[symbol] = {
                'quantity': order['quantity'],
                'entry_price': order['price']
            }
        else:
            if symbol in self.state.positions:
                del self.state.positions[symbol]
                
    def on_position_update(self, position: Dict[str, Any]):
        """Handle position updates"""
        pass
        
    def get_required_indicators(self) -> List[str]:
        """Return required indicators"""
        return ["RSI"]
'''
    }
    
    return {
        "templates": templates,
        "description": "Strategy code templates to get started"
    }

@router.delete("/{strategy_name}")
async def unregister_strategy(strategy_name: str):
    """Unregister a custom strategy"""
    
    # Check if strategy exists
    if not strategy_registry.get_strategy_class(strategy_name):
        raise HTTPException(
            status_code=404,
            detail=f"Strategy '{strategy_name}' not found"
        )
    
    # Check if strategy has active instances
    from ..paper_trade import active_strategies
    active_instances = [s for s in active_strategies.values() 
                       if s["name"] == strategy_name and s["status"] == "running"]
    
    if active_instances:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot unregister strategy with {len(active_instances)} active instances"
        )
    
    # Remove from registry
    if strategy_name in strategy_registry._strategies:
        del strategy_registry._strategies[strategy_name]
        
    return {"message": f"Strategy '{strategy_name}' unregistered successfully"}