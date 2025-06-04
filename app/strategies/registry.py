# app/strategies/registry.py
import importlib
import inspect
import sys
from typing import Dict, Type, List, Optional, Any, Tuple
from pathlib import Path
import logging
import json
from datetime import datetime
import traceback

from .base_strategy import BaseStrategy, StrategyConfig

logger = logging.getLogger(__name__)

class StrategyRegistry:
    """
    Registry for dynamic strategy loading and management
    Handles strategy discovery, registration, validation, and instantiation
    """
    
    def __init__(self):
        self._strategies: Dict[str, Type[BaseStrategy]] = {}
        self._instances: Dict[str, BaseStrategy] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._validation_errors: Dict[str, str] = {}
        
        # Strategy categories for organization
        self._categories: Dict[str, List[str]] = {
            'momentum': [],
            'mean_reversion': [],
            'arbitrage': [],
            'market_making': [],
            'machine_learning': [],
            'custom': []
        }
        
    def register(self, 
                 name: str, 
                 strategy_class: Type[BaseStrategy],
                 category: str = 'custom',
                 override: bool = False) -> None:
        """
        Register a strategy class
        
        Args:
            name: Strategy name
            strategy_class: Strategy class
            category: Strategy category
            override: Whether to override existing strategy
            
        Raises:
            ValueError: If strategy is invalid or name exists
        """
        
        # Validate strategy class
        if not self._validate_strategy_class(strategy_class):
            raise ValueError(f"{strategy_class} must inherit from BaseStrategy")
            
        # Check if already registered
        if name in self._strategies and not override:
            raise ValueError(f"Strategy '{name}' already registered. Use override=True to replace.")
            
        # Validate strategy implementation
        validation_errors = self._validate_strategy_implementation(strategy_class)
        if validation_errors:
            self._validation_errors[name] = "; ".join(validation_errors)
            raise ValueError(f"Strategy validation failed: {'; '.join(validation_errors)}")
            
        # Register strategy
        self._strategies[name] = strategy_class
        
        # Extract and store metadata
        self._metadata[name] = self._extract_strategy_metadata(name, strategy_class)
        
        # Add to category
        if category in self._categories:
            if name not in self._categories[category]:
                self._categories[category].append(name)
        else:
            self._categories['custom'].append(name)
            
        logger.info(f"Registered strategy '{name}' in category '{category}'")
        
    def unregister(self, name: str) -> bool:
        """
        Unregister a strategy
        
        Args:
            name: Strategy name
            
        Returns:
            True if unregistered, False if not found
        """
        
        if name not in self._strategies:
            return False
            
        # Remove from registry
        del self._strategies[name]
        
        # Remove metadata
        if name in self._metadata:
            del self._metadata[name]
            
        # Remove from categories
        for category_strategies in self._categories.values():
            if name in category_strategies:
                category_strategies.remove(name)
                
        # Remove any instances
        instances_to_remove = [key for key in self._instances if key.startswith(f"{name}_")]
        for key in instances_to_remove:
            del self._instances[key]
            
        logger.info(f"Unregistered strategy '{name}'")
        return True
        
    def get_strategy_class(self, name: str) -> Optional[Type[BaseStrategy]]:
        """Get strategy class by name"""
        return self._strategies.get(name)
    
    def create_instance(self, 
                       name: str, 
                       config: StrategyConfig,
                       instance_id: Optional[str] = None) -> BaseStrategy:
        """
        Create strategy instance
        
        Args:
            name: Strategy name
            config: Strategy configuration
            instance_id: Optional instance ID
            
        Returns:
            Strategy instance
            
        Raises:
            ValueError: If strategy not found
        """
        
        strategy_class = self.get_strategy_class(name)
        if not strategy_class:
            raise ValueError(f"Strategy '{name}' not found")
            
        # Validate configuration
        validation_errors = self._validate_strategy_config(strategy_class, config)
        if validation_errors:
            raise ValueError(f"Invalid configuration: {'; '.join(validation_errors)}")
        
        try:
            # Create instance
            instance = strategy_class(config)
            
            # Initialize strategy
            instance.initialize()
            
            # Store instance
            if instance_id is None:
                instance_id = f"{name}_{id(instance)}"
            self._instances[instance_id] = instance
            
            logger.info(f"Created instance of strategy '{name}' with ID '{instance_id}'")
            return instance
            
        except Exception as e:
            logger.error(f"Failed to create strategy instance: {e}")
            raise
    
    def get_instance(self, instance_id: str) -> Optional[BaseStrategy]:
        """Get strategy instance by ID"""
        return self._instances.get(instance_id)
    
    def list_strategies(self, category: Optional[str] = None) -> List[str]:
        """
        List all registered strategies
        
        Args:
            category: Optional category filter
            
        Returns:
            List of strategy names
        """
        
        if category and category in self._categories:
            return self._categories[category].copy()
        
        return list(self._strategies.keys())
    
    def list_instances(self, strategy_name: Optional[str] = None) -> List[str]:
        """
        List all strategy instances
        
        Args:
            strategy_name: Optional filter by strategy name
            
        Returns:
            List of instance IDs
        """
        
        if strategy_name:
            return [key for key in self._instances if key.startswith(f"{strategy_name}_")]
        
        return list(self._instances.keys())
    
    def get_strategy_info(self, name: str) -> Dict[str, Any]:
        """
        Get strategy information
        
        Args:
            name: Strategy name
            
        Returns:
            Strategy metadata
        """
        
        if name not in self._metadata:
            strategy_class = self.get_strategy_class(name)
            if strategy_class:
                self._metadata[name] = self._extract_strategy_metadata(name, strategy_class)
            else:
                return {}
                
        return self._metadata.get(name, {})
    
    def get_all_strategies_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information for all strategies"""
        
        info = {}
        for name in self._strategies:
            info[name] = self.get_strategy_info(name)
        return info
    
    def auto_discover_strategies(self, 
                               paths: Optional[List[str]] = None,
                               recursive: bool = True) -> Dict[str, List[str]]:
        """
        Auto-discover and register strategies from directories
        
        Args:
            paths: List of paths to search (default: standard locations)
            recursive: Whether to search recursively
            
        Returns:
            Dictionary of discovered strategies by path
        """
        
        if paths is None:
            paths = [
                "app/strategies/implementations",
                "app/strategies/implementations/custom",
                "strategies"  # User strategies directory
            ]
            
        discovered = {}
        
        for path_str in paths:
            path = Path(path_str)
            if not path.exists():
                logger.warning(f"Strategy path does not exist: {path}")
                continue
                
            path_strategies = self._discover_in_path(path, recursive)
            if path_strategies:
                discovered[str(path)] = path_strategies
                
        return discovered
    
    def _discover_in_path(self, path: Path, recursive: bool) -> List[str]:
        """Discover strategies in a specific path"""
        
        discovered = []
        
        # Get all Python files
        if recursive:
            python_files = path.rglob("*.py")
        else:
            python_files = path.glob("*.py")
            
        for file_path in python_files:
            # Skip special files
            if file_path.name.startswith("_") or file_path.name == "setup.py":
                continue
                
            try:
                # Import module
                module = self._import_module_from_path(file_path)
                
                # Find strategy classes
                strategies = self._find_strategies_in_module(module)
                
                # Register strategies
                for strategy_name, strategy_class in strategies:
                    # Determine category from path
                    category = self._determine_category(file_path)
                    
                    try:
                        self.register(strategy_name, strategy_class, category)
                        discovered.append(strategy_name)
                    except ValueError as e:
                        logger.warning(f"Failed to register {strategy_name}: {e}")
                        
            except Exception as e:
                logger.error(f"Failed to import module {file_path}: {e}")
                logger.debug(traceback.format_exc())
                
        return discovered
    
    def _import_module_from_path(self, file_path: Path):
        """Import a module from file path"""
        
        # Convert path to module name
        module_path = str(file_path).replace("/", ".").replace("\\", ".").replace(".py", "")
        
        # Try standard import first
        try:
            return importlib.import_module(module_path)
        except ImportError:
            pass
            
        # Try dynamic import
        spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[file_path.stem] = module
            spec.loader.exec_module(module)
            return module
            
        raise ImportError(f"Could not import module from {file_path}")
    
    def _find_strategies_in_module(self, module) -> List[Tuple[str, Type[BaseStrategy]]]:
        """Find all strategy classes in a module"""
        
        strategies = []
        
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and 
                issubclass(obj, BaseStrategy) and 
                obj is not BaseStrategy and
                not inspect.isabstract(obj)):
                
                # Use class name as strategy name
                strategy_name = obj.__name__
                
                # Remove common suffixes
                for suffix in ['Strategy', 'Algo', 'System']:
                    if strategy_name.endswith(suffix):
                        strategy_name = strategy_name[:-len(suffix)]
                        break
                        
                strategies.append((strategy_name, obj))
                
        return strategies
    
    def _determine_category(self, file_path: Path) -> str:
        """Determine strategy category from file path"""
        
        path_str = str(file_path).lower()
        
        # Check for category keywords in path
        category_keywords = {
            'momentum': 'momentum',
            'mean_reversion': 'mean_reversion',
            'arbitrage': 'arbitrage',
            'market_making': 'market_making',
            'ml': 'machine_learning',
            'machine_learning': 'machine_learning',
            'custom': 'custom'
        }
        
        for keyword, category in category_keywords.items():
            if keyword in path_str:
                return category
                
        return 'custom'
    
    def _validate_strategy_class(self, strategy_class: Type) -> bool:
        """Validate that class is a proper strategy"""
        
        return (inspect.isclass(strategy_class) and 
                issubclass(strategy_class, BaseStrategy) and
                strategy_class is not BaseStrategy)
    
    def _validate_strategy_implementation(self, strategy_class: Type[BaseStrategy]) -> List[str]:
        """Validate strategy implementation"""
        
        errors = []
        
        # Check required methods
        required_methods = ['initialize', 'on_data', 'on_order_fill', 'on_position_update']
        
        for method_name in required_methods:
            method = getattr(strategy_class, method_name, None)
            if not method:
                errors.append(f"Missing required method: {method_name}")
            elif method is getattr(BaseStrategy, method_name, None):
                errors.append(f"Method not implemented: {method_name}")
                
        # Check method signatures
        if hasattr(strategy_class, 'on_data'):
            sig = inspect.signature(strategy_class.on_data)
            params = list(sig.parameters.keys())
            if len(params) < 3 or params[1] != 'symbol' or params[2] != 'data':
                errors.append("on_data method has incorrect signature")
                
        return errors
    
    def _validate_strategy_config(self, 
                                strategy_class: Type[BaseStrategy], 
                                config: StrategyConfig) -> List[str]:
        """Validate strategy configuration"""
        
        errors = []
        
        # Basic validation
        if not config.name:
            errors.append("Strategy name is required")
            
        if not config.symbols:
            errors.append("At least one symbol is required")
            
        # Check if strategy has specific validation
        if hasattr(strategy_class, 'validate_config'):
            try:
                strategy_errors = strategy_class.validate_config(config)
                if strategy_errors:
                    errors.extend(strategy_errors)
            except Exception as e:
                errors.append(f"Config validation error: {str(e)}")
                
        return errors
    
    def _extract_strategy_metadata(self, name: str, strategy_class: Type[BaseStrategy]) -> Dict[str, Any]:
        """Extract metadata from strategy class"""
        
        metadata = {
            'name': name,
            'class': strategy_class.__name__,
            'module': strategy_class.__module__,
            'docstring': inspect.getdoc(strategy_class) or "No description available",
            'author': getattr(strategy_class, '__author__', 'Unknown'),
            'version': getattr(strategy_class, '__version__', '1.0.0'),
            'created_at': datetime.now().isoformat()
        }
        
        # Extract parameters
        metadata['parameters'] = self._extract_parameters(strategy_class)
        
        # Extract required indicators
        if hasattr(strategy_class, 'get_required_indicators'):
            try:
                # Create dummy instance to get indicators
                dummy_config = StrategyConfig(
                    name="dummy",
                    symbols=["DUMMY"],
                    parameters={}
                )
                dummy_instance = strategy_class(dummy_config)
                metadata['required_indicators'] = dummy_instance.get_required_indicators()
            except:
                metadata['required_indicators'] = []
        else:
            metadata['required_indicators'] = []
            
        # Extract trading style
        metadata['trading_style'] = self._determine_trading_style(strategy_class)
        
        # Extract risk profile
        metadata['risk_profile'] = self._determine_risk_profile(strategy_class)
        
        return metadata
    
    def _extract_parameters(self, strategy_class: Type[BaseStrategy]) -> Dict[str, Any]:
        """Extract strategy parameters"""
        
        parameters = {}
        
        # Check for parameter definitions
        if hasattr(strategy_class, 'PARAMETERS'):
            return strategy_class.PARAMETERS
            
        # Try to extract from __init__ or initialize method
        try:
            # Look for common parameter patterns in the class
            source = inspect.getsource(strategy_class)
            
            # Common parameter patterns
            import re
            
            # Pattern: self.param = self.config.parameters.get('param', default)
            pattern1 = r"self\.(\w+)\s*=\s*self\.config\.parameters\.get\(['\"](\w+)['\"]\s*,\s*([^)]+)\)"
            matches1 = re.findall(pattern1, source)
            
            for attr, param, default in matches1:
                try:
                    # Evaluate default value
                    default_value = eval(default)
                    parameters[param] = {
                        'type': type(default_value).__name__,
                        'default': default_value,
                        'description': f"{param} parameter"
                    }
                except:
                    parameters[param] = {
                        'type': 'unknown',
                        'default': default,
                        'description': f"{param} parameter"
                    }
                    
        except Exception as e:
            logger.debug(f"Could not extract parameters: {e}")
            
        return parameters
    
    def _determine_trading_style(self, strategy_class: Type[BaseStrategy]) -> str:
        """Determine trading style from strategy implementation"""
        
        # Check class attributes
        if hasattr(strategy_class, 'TRADING_STYLE'):
            return strategy_class.TRADING_STYLE
            
        # Analyze class name and docstring
        class_name = strategy_class.__name__.lower()
        docstring = (inspect.getdoc(strategy_class) or "").lower()
        
        # Pattern matching
        styles = {
            'scalping': ['scalp', 'hft', 'high frequency'],
            'day_trading': ['day', 'intraday', 'daily'],
            'swing_trading': ['swing', 'position', 'trend'],
            'arbitrage': ['arbitrage', 'spread', 'pairs'],
            'market_making': ['market making', 'liquidity', 'bid ask']
        }
        
        for style, keywords in styles.items():
            for keyword in keywords:
                if keyword in class_name or keyword in docstring:
                    return style
                    
        return 'unknown'
    
    def _determine_risk_profile(self, strategy_class: Type[BaseStrategy]) -> str:
        """Determine risk profile from strategy implementation"""
        
        # Check class attributes
        if hasattr(strategy_class, 'RISK_PROFILE'):
            return strategy_class.RISK_PROFILE
            
        # Default to medium
        return 'medium'
    
    def export_registry(self, filepath: str):
        """Export registry to JSON file"""
        
        registry_data = {
            'strategies': {},
            'categories': self._categories,
            'metadata': self._metadata,
            'exported_at': datetime.now().isoformat()
        }
        
        # Add strategy information
        for name, strategy_class in self._strategies.items():
            registry_data['strategies'][name] = {
                'class': strategy_class.__name__,
                'module': strategy_class.__module__,
                'metadata': self._metadata.get(name, {})
            }
            
        with open(filepath, 'w') as f:
            json.dump(registry_data, f, indent=2)
            
        logger.info(f"Exported registry to {filepath}")
    
    def import_custom_strategy(self, 
                             code: str, 
                             strategy_name: str,
                             validate: bool = True) -> Type[BaseStrategy]:
        """
        Import a strategy from code string
        
        Args:
            code: Python code containing strategy
            strategy_name: Name to register strategy as
            validate: Whether to validate before registering
            
        Returns:
            Strategy class
            
        Raises:
            ValueError: If import or validation fails
        """
        
        # Create a temporary module
        import types
        module_name = f"custom_strategy_{strategy_name}_{id(code)}"
        module = types.ModuleType(module_name)
        
        # Add necessary imports to module namespace
        module.__dict__.update({
            'BaseStrategy': BaseStrategy,
            'StrategyConfig': StrategyConfig,
            'Signal': Signal,
            'OrderType': OrderType,
            'OrderSide': OrderSide,
            'pd': pd,
            'np': np,
            'datetime': datetime,
            'Optional': Optional,
            'Dict': Dict,
            'List': List,
            'Any': Any
        })
        
        # Import common dependencies
        import pandas as pd
        import numpy as np
        from .base_strategy import Signal, OrderType, OrderSide
        
        try:
            # Execute code in module context
            exec(code, module.__dict__)
            
            # Find strategy class
            strategy_class = None
            for name, obj in inspect.getmembers(module):
                if self._validate_strategy_class(obj):
                    strategy_class = obj
                    break
                    
            if not strategy_class:
                raise ValueError("No valid strategy class found in code")
                
            # Validate if requested
            if validate:
                errors = self._validate_strategy_implementation(strategy_class)
                if errors:
                    raise ValueError(f"Strategy validation failed: {'; '.join(errors)}")
                    
            # Register strategy
            self.register(strategy_name, strategy_class, category='custom')
            
            return strategy_class
            
        except SyntaxError as e:
            raise ValueError(f"Syntax error in strategy code: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error importing strategy: {str(e)}")
    
    def get_strategy_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        
        total_strategies = len(self._strategies)
        total_instances = len(self._instances)
        
        # Count by category
        category_counts = {}
        for category, strategies in self._categories.items():
            category_counts[category] = len(strategies)
            
        # Count instances by strategy
        instance_counts = {}
        for instance_id in self._instances:
            strategy_name = instance_id.split('_')[0]
            instance_counts[strategy_name] = instance_counts.get(strategy_name, 0) + 1
            
        return {
            'total_strategies': total_strategies,
            'total_instances': total_instances,
            'strategies_by_category': category_counts,
            'instances_by_strategy': instance_counts,
            'validation_errors': len(self._validation_errors),
            'categories': list(self._categories.keys())
        }
    
    def clear(self):
        """Clear all registered strategies and instances"""
        
        self._strategies.clear()
        self._instances.clear()
        self._metadata.clear()
        self._validation_errors.clear()
        
        for category in self._categories:
            self._categories[category].clear()
            
        logger.info("Registry cleared")

# Global registry instance
strategy_registry = StrategyRegistry()


# Helper functions for strategy development
def validate_strategy(strategy_class: Type[BaseStrategy]) -> Tuple[bool, List[str]]:
    """
    Validate a strategy class without registering
    
    Args:
        strategy_class: Strategy class to validate
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    
    registry = StrategyRegistry()
    
    if not registry._validate_strategy_class(strategy_class):
        return False, ["Class must inherit from BaseStrategy"]
        
    errors = registry._validate_strategy_implementation(strategy_class)
    
    return len(errors) == 0, errors


def create_strategy_template(strategy_name: str, 
                           trading_style: str = "swing_trading",
                           indicators: Optional[List[str]] = None) -> str:
    """
    Generate a strategy template
    
    Args:
        strategy_name: Name of the strategy
        trading_style: Trading style
        indicators: List of indicators to include
        
    Returns:
        Python code template
    """
    
    if indicators is None:
        indicators = ["SMA", "RSI"]
        
    template = f'''"""
{strategy_name} Strategy Implementation
Trading Style: {trading_style}
"""

from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
from datetime import datetime

from app.strategies.base_strategy import BaseStrategy, Signal, OrderSide, OrderType, StrategyConfig


class {strategy_name}Strategy(BaseStrategy):
    """
    {strategy_name} trading strategy
    
    This strategy implements...
    """
    
    __author__ = "Your Name"
    __version__ = "1.0.0"
    
    # Strategy metadata
    TRADING_STYLE = "{trading_style}"
    RISK_PROFILE = "medium"
    
    # Parameter definitions
    PARAMETERS = {{
        "param1": {{
            "type": "int",
            "default": 20,
            "min": 1,
            "max": 100,
            "description": "First parameter"
        }},
        "param2": {{
            "type": "float", 
            "default": 0.02,
            "min": 0.001,
            "max": 0.1,
            "description": "Second parameter"
        }}
    }}
    
    def initialize(self) -> None:
        """Initialize strategy parameters and state"""
        
        # Extract parameters
        self.param1 = self.config.parameters.get('param1', 20)
        self.param2 = self.config.parameters.get('param2', 0.02)
        
        # Initialize state
        self.in_position = {{}}
        self.entry_prices = {{}}
        
        # Log initialization
        print(f"Initialized {{self.name}} with param1={{self.param1}}, param2={{self.param2}}")
        
    def on_data(self, symbol: str, data: pd.DataFrame) -> Optional[Signal]:
        """
        Process new data and generate trading signals
        
        Args:
            symbol: Trading symbol
            data: DataFrame with OHLCV data and indicators
            
        Returns:
            Trading signal or None
        """
        
        # Ensure we have enough data
        if len(data) < self.param1:
            return None
            
        # Get latest values
        current_price = data['close'].iloc[-1]
        
        # Example signal generation logic
        # TODO: Implement your strategy logic here
        
        # Example: Simple moving average crossover
        if 'sma_{{self.param1}}' in data.columns:
            sma = data[f'sma_{{self.param1}}'].iloc[-1]
            
            # Buy signal
            if current_price > sma and symbol not in self.in_position:
                return Signal(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    quantity=0,  # Will be calculated by position sizing
                    order_type=OrderType.MARKET,
                    metadata={{
                        'reason': 'Price crossed above SMA',
                        'sma': sma,
                        'price': current_price
                    }}
                )
                
            # Sell signal
            elif current_price < sma and symbol in self.in_position:
                return Signal(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    quantity=self.in_position.get(symbol, 0),
                    order_type=OrderType.MARKET,
                    metadata={{
                        'reason': 'Price crossed below SMA',
                        'sma': sma,
                        'price': current_price
                    }}
                )
                
        return None
        
    def on_order_fill(self, order: Dict[str, Any]) -> None:
        """
        Handle order fill events
        
        Args:
            order: Order fill details
        """
        
        symbol = order['symbol']
        side = order['side']
        quantity = order['quantity']
        price = order['price']
        
        if side == 'BUY':
            self.in_position[symbol] = quantity
            self.entry_prices[symbol] = price
            print(f"Bought {{quantity}} {{symbol}} @ {{price}}")
            
        elif side == 'SELL':
            if symbol in self.in_position:
                entry_price = self.entry_prices.get(symbol, price)
                pnl = (price - entry_price) * quantity
                print(f"Sold {{quantity}} {{symbol}} @ {{price}}, P&L: {{pnl:.2f}}")
                
                del self.in_position[symbol]
                del self.entry_prices[symbol]
                
    def on_position_update(self, position: Dict[str, Any]) -> None:
        """
        Handle position updates
        
        Args:
            position: Position details
        """
        
        # Update internal position tracking if needed
        symbol = position['symbol']
        quantity = position['quantity']
        
        if quantity == 0 and symbol in self.in_position:
            del self.in_position[symbol]
            if symbol in self.entry_prices:
                del self.entry_prices[symbol]
                
    def calculate_position_size(self, 
                              symbol: str, 
                              signal: Signal, 
                              capital: float, 
                              current_price: float) -> int:
        """
        Calculate position size based on risk management
        
        Args:
            symbol: Trading symbol
            signal: Trading signal
            capital: Available capital
            current_price: Current market price
            
        Returns:
            Position size in units
        """
        
        # Simple fixed percentage position sizing
        position_value = capital * self.param2
        position_size = int(position_value / current_price)
        
        return max(1, position_size)  # At least 1 unit
        
    def get_required_indicators(self) -> List[str]:
        """Return list of required technical indicators"""
        
        return {indicators}
        
    @classmethod
    def validate_config(cls, config: StrategyConfig) -> List[str]:
        """
        Validate strategy configuration
        
        Args:
            config: Strategy configuration
            
        Returns:
            List of validation errors
        """
        
        errors = []
        
        # Validate parameters
        param1 = config.parameters.get('param1', 20)
        if not isinstance(param1, int) or param1 < 1 or param1 > 100:
            errors.append("param1 must be an integer between 1 and 100")
            
        param2 = config.parameters.get('param2', 0.02)
        if not isinstance(param2, (int, float)) or param2 < 0.001 or param2 > 0.1:
            errors.append("param2 must be a number between 0.001 and 0.1")
            
        return errors
'''
    
    return template