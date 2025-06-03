# app/paper_trading/risk_manager.py
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
from collections import defaultdict
import numpy as np

from ..strategies.base_strategy import Signal, OrderSide
from ..core.position_manager import PositionManager
from .virtual_broker import VirtualBroker

logger = logging.getLogger(__name__)

class RiskLimits:
    """Risk limit configuration"""
    
    def __init__(self):
        # Position limits
        self.max_position_size_pct = 0.25      # 25% of capital per position
        self.max_portfolio_risk_pct = 0.02     # 2% portfolio risk
        self.max_open_positions = 10           # Maximum concurrent positions
        self.max_correlation_exposure = 0.6    # Max exposure to correlated assets
        
        # Daily limits
        self.max_daily_loss_pct = 0.05         # 5% daily loss limit
        self.max_daily_trades = 50             # Maximum trades per day
        self.max_consecutive_losses = 5        # Circuit breaker
        
        # Order limits
        self.max_order_size_pct = 0.1          # 10% of average volume
        self.min_order_value = 100             # Minimum order value
        self.max_order_value_pct = 0.2         # 20% of capital
        
        # Risk metrics thresholds
        self.min_sharpe_ratio = -0.5           # Minimum acceptable Sharpe
        self.max_drawdown_pct = 0.20          # 20% maximum drawdown
        self.max_leverage = 1.0                # No leverage by default
        
        # Time-based restrictions
        self.no_trade_hours = []               # Hours when trading is restricted
        self.reduce_risk_hours = [9, 15]       # Reduce risk at open/close
        
class RiskMetrics:
    """Real-time risk metrics tracking"""
    
    def __init__(self):
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.peak_portfolio_value = 0.0
        self.current_drawdown = 0.0
        self.daily_var = 0.0
        self.position_correlations = {}
        self.last_reset = datetime.now().date()
        
    def reset_daily_metrics(self):
        """Reset daily metrics"""
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.last_reset = datetime.now().date()
        
class RiskManager:
    """
    Comprehensive risk management system for paper trading
    Monitors and enforces risk limits in real-time
    """
    
    def __init__(self, initial_capital: float = 1000000):
        self.initial_capital = initial_capital
        self.limits = RiskLimits()
        self.metrics = RiskMetrics()
        
        # Risk monitoring
        self._monitoring = False
        self._monitoring_task: Optional[asyncio.Task] = None
        
        # Position tracking
        self.position_risks: Dict[str, Dict[str, float]] = {}
        self.symbol_exposure: Dict[str, float] = defaultdict(float)
        self.sector_exposure: Dict[str, float] = defaultdict(float)
        
        # Historical tracking for risk calculations
        self.pnl_history: List[float] = []
        self.portfolio_values: List[Tuple[datetime, float]] = []
        self.trade_history: List[Dict[str, Any]] = []
        
        # Risk events and alerts
        self.risk_events: List[Dict[str, Any]] = []
        self.alert_callbacks = []
        
        # Correlation data (simplified - in production, calculate from market data)
        self.correlation_matrix = self._initialize_correlations()
        
    async def check_signal(self, 
                          signal: Signal,
                          position_manager: PositionManager,
                          strategy_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if signal passes risk checks
        
        Returns:
            Dict with 'approved' bool and 'reason' if rejected
        """
        
        # Reset daily metrics if needed
        if datetime.now().date() > self.metrics.last_reset:
            self.metrics.reset_daily_metrics()
            
        # Run all risk checks
        checks = [
            self._check_position_size(signal, position_manager),
            self._check_portfolio_risk(signal, position_manager),
            self._check_daily_limits(),
            self._check_drawdown_limit(position_manager),
            self._check_correlation_risk(signal, position_manager),
            self._check_time_restrictions(),
            self._check_consecutive_losses(),
            self._check_position_limits(position_manager),
            self._check_order_value(signal, position_manager)
        ]
        
        # Process all checks
        for approved, reason in checks:
            if not approved:
                self._log_risk_event("signal_rejected", signal.symbol, reason)
                return {'approved': False, 'reason': reason}
                
        return {'approved': True, 'reason': None}
        
    def _check_position_size(self, 
                           signal: Signal,
                           position_manager: PositionManager) -> Tuple[bool, Optional[str]]:
        """Check position size limits"""
        
        portfolio_value = position_manager.get_portfolio_value()
        
        # Get current position if exists
        current_position = position_manager.get_position(signal.symbol)
        current_quantity = current_position.quantity if current_position else 0
        
        # Calculate new position size
        if signal.side == OrderSide.BUY:
            new_quantity = current_quantity + signal.quantity
        else:
            new_quantity = max(0, current_quantity - signal.quantity)
            
        # Estimate position value (need current price)
        estimated_price = signal.price or 100  # Fallback
        position_value = new_quantity * estimated_price
        
        # Check against limit
        position_pct = position_value / portfolio_value
        if position_pct > self.limits.max_position_size_pct:
            return False, f"Position size {position_pct:.1%} exceeds limit {self.limits.max_position_size_pct:.1%}"
            
        return True, None
        
    def _check_portfolio_risk(self,
                            signal: Signal,
                            position_manager: PositionManager) -> Tuple[bool, Optional[str]]:
        """Check portfolio-wide risk"""
        
        # Calculate current portfolio risk
        portfolio_value = position_manager.get_portfolio_value()
        total_risk = 0.0
        
        # Sum risk across all positions
        for symbol, position in position_manager.positions.items():
            position_risk = self._calculate_position_risk(position, portfolio_value)
            total_risk += position_risk
            
        # Add risk from new signal
        if signal.side == OrderSide.BUY and signal.stop_loss:
            estimated_price = signal.price or 100
            potential_loss = abs(estimated_price - signal.stop_loss) * signal.quantity
            signal_risk = potential_loss / portfolio_value
            total_risk += signal_risk
            
        # Check against limit
        if total_risk > self.limits.max_portfolio_risk_pct:
            return False, f"Portfolio risk {total_risk:.1%} exceeds limit {self.limits.max_portfolio_risk_pct:.1%}"
            
        return True, None
        
    def _check_daily_limits(self) -> Tuple[bool, Optional[str]]:
        """Check daily trading limits"""
        
        # Check daily loss limit
        if self.metrics.daily_pnl < 0:
            daily_loss_pct = abs(self.metrics.daily_pnl) / self.initial_capital
            if daily_loss_pct > self.limits.max_daily_loss_pct:
                return False, f"Daily loss {daily_loss_pct:.1%} exceeds limit {self.limits.max_daily_loss_pct:.1%}"
                
        # Check daily trade count
        if self.metrics.daily_trades >= self.limits.max_daily_trades:
            return False, f"Daily trade limit ({self.limits.max_daily_trades}) reached"
            
        return True, None
        
    def _check_drawdown_limit(self, position_manager: PositionManager) -> Tuple[bool, Optional[str]]:
        """Check maximum drawdown limit"""
        
        current_value = position_manager.get_portfolio_value()
        
        # Update peak value
        if current_value > self.metrics.peak_portfolio_value:
            self.metrics.peak_portfolio_value = current_value
            
        # Calculate drawdown
        if self.metrics.peak_portfolio_value > 0:
            drawdown = (self.metrics.peak_portfolio_value - current_value) / self.metrics.peak_portfolio_value
            self.metrics.current_drawdown = drawdown
            
            if drawdown > self.limits.max_drawdown_pct:
                return False, f"Drawdown {drawdown:.1%} exceeds limit {self.limits.max_drawdown_pct:.1%}"
                
        return True, None
        
    def _check_correlation_risk(self,
                              signal: Signal,
                              position_manager: PositionManager) -> Tuple[bool, Optional[str]]:
        """Check correlation-based risk limits"""
        
        if signal.side != OrderSide.BUY:
            return True, None  # Only check for new long positions
            
        # Get current positions
        positions = position_manager.get_all_positions()
        if not positions:
            return True, None
            
        # Calculate correlation exposure
        total_correlation_exposure = 0.0
        portfolio_value = position_manager.get_portfolio_value()
        
        for position in positions:
            if position['symbol'] != signal.symbol:
                correlation = self._get_correlation(signal.symbol, position['symbol'])
                position_weight = position['market_value'] / portfolio_value
                total_correlation_exposure += abs(correlation * position_weight)
                
        # Check limit
        if total_correlation_exposure > self.limits.max_correlation_exposure:
            return False, f"Correlation exposure {total_correlation_exposure:.2f} exceeds limit"
            
        return True, None
        
    def _check_time_restrictions(self) -> Tuple[bool, Optional[str]]:
        """Check time-based trading restrictions"""
        
        current_hour = datetime.now().hour
        
        # Check no-trade hours
        if current_hour in self.limits.no_trade_hours:
            return False, f"Trading restricted at hour {current_hour}"
            
        # Note: Could implement reduced position sizing during certain hours
        
        return True, None
        
    def _check_consecutive_losses(self) -> Tuple[bool, Optional[str]]:
        """Check consecutive loss limit"""
        
        if self.metrics.consecutive_losses >= self.limits.max_consecutive_losses:
            return False, f"Consecutive losses ({self.metrics.consecutive_losses}) exceeds limit"
            
        return True, None
        
    def _check_position_limits(self, position_manager: PositionManager) -> Tuple[bool, Optional[str]]:
        """Check position count limits"""
        
        open_positions = len(position_manager.positions)
        
        if open_positions >= self.limits.max_open_positions:
            return False, f"Maximum open positions ({self.limits.max_open_positions}) reached"
            
        return True, None
        
    def _check_order_value(self,
                         signal: Signal,
                         position_manager: PositionManager) -> Tuple[bool, Optional[str]]:
        """Check order value limits"""
        
        estimated_price = signal.price or 100
        order_value = signal.quantity * estimated_price
        
        # Check minimum order value
        if order_value < self.limits.min_order_value:
            return False, f"Order value ${order_value:.2f} below minimum ${self.limits.min_order_value}"
            
        # Check maximum order value
        portfolio_value = position_manager.get_portfolio_value()
        order_value_pct = order_value / portfolio_value
        
        if order_value_pct > self.limits.max_order_value_pct:
            return False, f"Order value {order_value_pct:.1%} exceeds limit {self.limits.max_order_value_pct:.1%}"
            
        return True, None
        
    async def start_monitoring(self, 
                             position_manager: PositionManager,
                             virtual_broker: VirtualBroker):
        """Start real-time risk monitoring"""
        
        if self._monitoring:
            return
            
        logger.info("Starting risk monitoring")
        self._monitoring = True
        
        # Initialize peak value
        self.metrics.peak_portfolio_value = position_manager.get_portfolio_value()
        
        # Start monitoring task
        self._monitoring_task = asyncio.create_task(
            self._monitor_loop(position_manager, virtual_broker)
        )
        
    def stop_monitoring(self):
        """Stop risk monitoring"""
        
        logger.info("Stopping risk monitoring")
        self._monitoring = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            
    async def _monitor_loop(self,
                          position_manager: PositionManager,
                          virtual_broker: VirtualBroker):
        """Main risk monitoring loop"""
        
        logger.info("Risk monitoring loop started")
        
        while self._monitoring:
            try:
                # Update portfolio metrics
                portfolio_value = position_manager.get_portfolio_value()
                self.portfolio_values.append((datetime.now(), portfolio_value))
                
                # Limit history size
                if len(self.portfolio_values) > 1000:
                    self.portfolio_values = self.portfolio_values[-1000:]
                    
                # Calculate Value at Risk (VaR)
                self._calculate_var()
                
                # Check for risk breaches
                await self._check_risk_breaches(position_manager)
                
                # Update position risks
                self._update_position_risks(position_manager)
                
                # Sleep interval
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in risk monitoring: {e}")
                await asyncio.sleep(5)
                
    def _calculate_var(self):
        """Calculate Value at Risk"""
        
        if len(self.portfolio_values) < 20:
            return
            
        # Simple historical VaR calculation
        values = [v for _, v in self.portfolio_values[-100:]]
        returns = np.diff(values) / values[:-1]
        
        # 95% VaR (5th percentile of returns)
        if len(returns) > 0:
            var_95 = np.percentile(returns, 5)
            self.metrics.daily_var = var_95
            
    async def _check_risk_breaches(self, position_manager: PositionManager):
        """Check for risk limit breaches"""
        
        # Check drawdown
        current_value = position_manager.get_portfolio_value()
        if self.metrics.peak_portfolio_value > 0:
            drawdown = (self.metrics.peak_portfolio_value - current_value) / self.metrics.peak_portfolio_value
            
            if drawdown > self.limits.max_drawdown_pct * 0.8:  # 80% of limit
                await self._trigger_alert(
                    "drawdown_warning",
                    f"Drawdown {drawdown:.1%} approaching limit",
                    severity="warning"
                )
                
        # Check daily loss
        if self.metrics.daily_pnl < 0:
            daily_loss_pct = abs(self.metrics.daily_pnl) / self.initial_capital
            
            if daily_loss_pct > self.limits.max_daily_loss_pct * 0.8:
                await self._trigger_alert(
                    "daily_loss_warning",
                    f"Daily loss {daily_loss_pct:.1%} approaching limit",
                    severity="warning"
                )
                
    def _update_position_risks(self, position_manager: PositionManager):
        """Update position-specific risk metrics"""
        
        portfolio_value = position_manager.get_portfolio_value()
        
        for symbol, position in position_manager.positions.items():
            # Calculate position risk metrics
            position_value = position.market_value
            position_weight = position_value / portfolio_value
            
            # Estimate risk (simplified - use historical volatility in production)
            estimated_volatility = 0.02  # 2% daily volatility
            position_var = position_value * estimated_volatility * 1.645  # 95% VaR
            
            self.position_risks[symbol] = {
                'weight': position_weight,
                'value_at_risk': position_var,
                'unrealized_pnl': position.unrealized_pnl,
                'duration': (datetime.now() - position.entry_time).total_seconds() / 3600
            }
            
    def _calculate_position_risk(self, position: Any, portfolio_value: float) -> float:
        """Calculate risk for a single position"""
        
        # Simple risk calculation based on position size
        # In production, use stop loss, volatility, etc.
        position_weight = position.market_value / portfolio_value
        volatility = 0.02  # Assumed 2% volatility
        
        return position_weight * volatility
        
    def _get_correlation(self, symbol1: str, symbol2: str) -> float:
        """Get correlation between two symbols"""
        
        # Simplified correlation matrix
        # In production, calculate from historical data
        
        if symbol1 == symbol2:
            return 1.0
            
        # Example correlations
        correlations = {
            ('NIFTY', 'BANKNIFTY'): 0.85,
            ('RELIANCE', 'TCS'): 0.6,
            # Add more correlations
        }
        
        key = tuple(sorted([symbol1, symbol2]))
        return correlations.get(key, 0.3)  # Default correlation
        
    def _initialize_correlations(self) -> Dict[Tuple[str, str], float]:
        """Initialize correlation matrix"""
        
        # In production, calculate from historical data
        return {}
        
    async def _trigger_alert(self, alert_type: str, message: str, severity: str = "info"):
        """Trigger risk alert"""
        
        alert = {
            'type': alert_type,
            'message': message,
            'severity': severity,
            'timestamp': datetime.now()
        }
        
        self.risk_events.append(alert)
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                await callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
                
        logger.warning(f"Risk Alert [{severity}]: {message}")
        
    def _log_risk_event(self, event_type: str, symbol: str, details: str):
        """Log risk event"""
        
        event = {
            'type': event_type,
            'symbol': symbol,
            'details': details,
            'timestamp': datetime.now()
        }
        
        self.risk_events.append(event)
        
        # Limit event history
        if len(self.risk_events) > 1000:
            self.risk_events = self.risk_events[-1000:]
            
    def update_trade_result(self, trade: Dict[str, Any]):
        """Update metrics based on trade result"""
        
        self.metrics.daily_trades += 1
        
        if 'pnl' in trade:
            self.metrics.daily_pnl += trade['pnl']
            self.pnl_history.append(trade['pnl'])
            
            # Update consecutive wins/losses
            if trade['pnl'] > 0:
                self.metrics.consecutive_wins += 1
                self.metrics.consecutive_losses = 0
            elif trade['pnl'] < 0:
                self.metrics.consecutive_losses += 1
                self.metrics.consecutive_wins = 0
                
        # Limit history size
        if len(self.pnl_history) > 1000:
            self.pnl_history = self.pnl_history[-1000:]
            
    def get_risk_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk report"""
        
        return {
            'metrics': {
                'daily_pnl': self.metrics.daily_pnl,
                'daily_trades': self.metrics.daily_trades,
                'consecutive_losses': self.metrics.consecutive_losses,
                'current_drawdown': self.metrics.current_drawdown,
                'daily_var': self.metrics.daily_var
            },
            'position_risks': self.position_risks,
            'recent_events': self.risk_events[-10:],
            'limits': {
                'max_position_size': self.limits.max_position_size_pct,
                'max_daily_loss': self.limits.max_daily_loss_pct,
                'max_drawdown': self.limits.max_drawdown_pct,
                'max_open_positions': self.limits.max_open_positions
            }
        }
        
    def register_alert_callback(self, callback: Callable):
        """Register callback for risk alerts"""
        self.alert_callbacks.append(callback)
        
    def set_risk_limits(self, **kwargs):
        """Update risk limits"""
        
        for key, value in kwargs.items():
            if hasattr(self.limits, key):
                setattr(self.limits, key, value)
                logger.info(f"Updated risk limit {key} to {value}")