# app/strategies/registry.py
import importlib
import inspect
from typing import Dict, Type, List, Optional, Any
from pathlib import Path
from .base_strategy import BaseStrategy, StrategyConfig

class StrategyRegistry:
    """Registry for dynamic strategy loading and management"""
    
    def __init__(self):
        self._strategies: Dict[str, Type[BaseStrategy]] = {}
        self._instances: Dict[str, BaseStrategy] = {}
        
    def register(self, name: str, strategy_class: Type[BaseStrategy]) -> None:
        """Register a strategy class"""
        if not issubclass(strategy_class, BaseStrategy):
            raise ValueError(f"{strategy_class} must inherit from BaseStrategy")
        self._strategies[name] = strategy_class
        
    def get_strategy_class(self, name: str) -> Optional[Type[BaseStrategy]]:
        """Get strategy class by name"""
        return self._strategies.get(name)
    
    def create_instance(self, name: str, config: StrategyConfig) -> BaseStrategy:
        """Create strategy instance"""
        strategy_class = self.get_strategy_class(name)
        if not strategy_class:
            raise ValueError(f"Strategy {name} not found")
        
        instance = strategy_class(config)
        instance.initialize()
        
        instance_key = f"{name}_{id(instance)}"
        self._instances[instance_key] = instance
        
        return instance
    
    def list_strategies(self) -> List[str]:
        """List all registered strategies"""
        return list(self._strategies.keys())
    
    def auto_discover_strategies(self, path: str = "app/strategies/implementations") -> None:
        """Auto-discover and register strategies from a directory"""
        strategy_dir = Path(path)
        
        for file_path in strategy_dir.rglob("*.py"):
            if file_path.name.startswith("_"):
                continue
                
            module_path = str(file_path).replace("/", ".").replace(".py", "")
            try:
                module = importlib.import_module(module_path)
                
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, BaseStrategy) and 
                        obj is not BaseStrategy):
                        self.register(name, obj)
                        
            except Exception as e:
                print(f"Failed to load module {module_path}: {e}")
    
    def get_strategy_info(self, name: str) -> Dict[str, Any]:
        """Get strategy information"""
        strategy_class = self.get_strategy_class(name)
        if not strategy_class:
            return {}
            
        return {
            "name": name,
            "class": strategy_class.__name__,
            "module": strategy_class.__module__,
            "docstring": strategy_class.__doc__,
            "required_indicators": strategy_class.get_required_indicators(None) if hasattr(strategy_class, 'get_required_indicators') else []
        }

# Global registry instance
strategy_registry = StrategyRegistry()