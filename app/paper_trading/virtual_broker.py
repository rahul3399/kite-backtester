# app/paper_trading/virtual_broker.py
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import uuid
import logging
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)

class VirtualBroker:
    """
    Virtual broker for paper trading
    Simulates realistic order execution with slippage, commission, and market impact
    """
    
    def __init__(self, initial_capital: float = 1000000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        
        # Position tracking
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.orders: List[Dict[str, Any]] = []
        self.trades: List[Dict[str, Any]] = []
        
        # Market data
        self.market_prices: Dict[str, float] = {}
        self.bid_ask_spreads: Dict[str, Tuple[float, float]] = {}  # (bid, ask)
        
        # Execution parameters
        self.commission_rate = 0.0002  # 0.02%
        self.slippage_rate = 0.0001   # 0.01%
        self.market_impact_factor = 0.00001  # Price impact per unit traded
        
        # Risk limits
        self.max_position_size = 0.25  # 25% of capital per position
        self.max_leverage = 1.0        # No leverage by default
        
        # Performance tracking
        self.total_commission = 0.0
        self.total_slippage = 0.0
        self.execution_stats = {
            'total_orders': 0,
            'filled_orders': 0,
            'rejected_orders': 0,
            'partial_fills': 0,
            'avg_slippage': 0.0,
            'avg_fill_time': 0.0
        }
        
        # Order book simulation
        self.liquidity_factor = 1.0  # 1.0 = normal liquidity
        self.volatility_factor = 1.0  # 1.0 = normal volatility
        
    def set_execution_params(self, 
                           commission_rate: float = None,
                           slippage_rate: float = None,
                           market_impact_factor: float = None):
        """Update execution parameters"""
        if commission_rate is not None:
            self.commission_rate = commission_rate
        if slippage_rate is not None:
            self.slippage_rate = slippage_rate
        if market_impact_factor is not None:
            self.market_impact_factor = market_impact_factor
            
    def update_market_price(self, symbol: str, price: float, 
                          bid: Optional[float] = None, ask: Optional[float] = None):
        """Update market prices"""
        self.market_prices[symbol] = price
        
        # Update bid-ask spread
        if bid and ask:
            self.bid_ask_spreads[symbol] = (bid, ask)
        else:
            # Simulate spread based on volatility
            spread = price * 0.001 * self.volatility_factor  # 0.1% spread
            self.bid_ask_spreads[symbol] = (price - spread/2, price + spread/2)
            
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price"""
        return self.market_prices.get(symbol)
        
    def get_bid_ask(self, symbol: str) -> Tuple[Optional[float], Optional[float]]:
        """Get bid-ask prices"""
        return self.bid_ask_spreads.get(symbol, (None, None))
        
    def execute_order(self, order: Dict[str, Any], market_price: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Execute an order with realistic simulation
        
        Args:
            order: Order details
            market_price: Current market price (if not provided, uses stored price)
            
        Returns:
            Fill details if successful, None otherwise
        """
        
        symbol = order['symbol']
        side = order['side']
        quantity = order['quantity']
        order_type = order.get('order_type', 'MARKET')
        
        # Update execution stats
        self.execution_stats['total_orders'] += 1
        
        # Get market price
        if market_price is None:
            market_price = self.market_prices.get(symbol)
            
        if not market_price or market_price <= 0:
            logger.warning(f"No valid market price for {symbol}")
            self.execution_stats['rejected_orders'] += 1
            return None
            
        # Check risk limits
        is_valid, error = self._validate_order(order, market_price)
        if not is_valid:
            logger.warning(f"Order validation failed: {error}")
            self.execution_stats['rejected_orders'] += 1
            return None
            
        # Calculate execution price with slippage and market impact
        execution_price = self._calculate_execution_price(
            symbol, side, quantity, order_type, market_price, order.get('price')
        )
        
        if execution_price is None:
            self.execution_stats['rejected_orders'] += 1
            return None
            
        # Calculate costs
        order_value = execution_price * quantity
        commission = self._calculate_commission(order_value)
        slippage_cost = abs(execution_price - market_price) * quantity
        
        # Check capital for buy orders
        if side == 'BUY':
            required_capital = order_value + commission
            if required_capital > self.capital:
                logger.warning(f"Insufficient capital. Required: {required_capital}, Available: {self.capital}")
                self.execution_stats['rejected_orders'] += 1
                return None
                
        # Simulate partial fills for large orders
        filled_quantity = self._simulate_fill_quantity(symbol, quantity, side)
        
        if filled_quantity < quantity:
            self.execution_stats['partial_fills'] += 1
            logger.info(f"Partial fill: {filled_quantity}/{quantity} units")
            
        # Create fill record
        fill = {
            'fill_id': str(uuid.uuid4()),
            'order_id': order.get('order_id', str(uuid.uuid4())),
            'symbol': symbol,
            'side': side,
            'quantity': filled_quantity,
            'requested_quantity': quantity,
            'price': execution_price,
            'market_price': market_price,
            'commission': commission * (filled_quantity / quantity),
            'slippage': slippage_cost * (filled_quantity / quantity),
            'timestamp': datetime.now(),
            'order_type': order_type,
            'metadata': order.get('metadata', {})
        }
        
        # Update positions
        if side == 'BUY':
            self._open_or_add_position(fill)
        else:
            pnl = self._close_or_reduce_position(fill)
            fill['pnl'] = pnl
            
        # Record order and trade
        self.orders.append(order)
        self.trades.append(fill)
        
        # Update capital
        if side == 'BUY':
            self.capital -= (filled_quantity * execution_price + fill['commission'])
        else:
            self.capital += (filled_quantity * execution_price - fill['commission'])
            
        # Update statistics
        self.total_commission += fill['commission']
        self.total_slippage += fill['slippage']
        self.execution_stats['filled_orders'] += 1
        self._update_avg_slippage(fill['slippage'])
        
        logger.info(f"Order executed: {symbol} {side} {filled_quantity}@{execution_price:.2f}")
        
        return fill
        
    def _calculate_execution_price(self, 
                                 symbol: str, 
                                 side: str, 
                                 quantity: int,
                                 order_type: str,
                                 market_price: float,
                                 limit_price: Optional[float] = None) -> Optional[float]:
        """Calculate realistic execution price"""
        
        # Get bid-ask spread
        bid, ask = self.bid_ask_spreads.get(symbol, (None, None))
        if not bid or not ask:
            # Use market price with default spread
            spread = market_price * 0.001
            bid = market_price - spread/2
            ask = market_price + spread/2
            
        # Base execution price
        if side == 'BUY':
            base_price = ask
        else:
            base_price = bid
            
        # Apply slippage based on order size and market conditions
        slippage = self._calculate_slippage(symbol, quantity, side)
        
        # Apply market impact
        market_impact = self._calculate_market_impact(symbol, quantity)
        
        # Calculate final execution price
        if side == 'BUY':
            execution_price = base_price * (1 + slippage + market_impact)
        else:
            execution_price = base_price * (1 - slippage - market_impact)
            
        # Handle limit orders
        if order_type == 'LIMIT' and limit_price:
            if side == 'BUY' and execution_price > limit_price:
                # Cannot fill buy limit order above limit price
                return None
            elif side == 'SELL' and execution_price < limit_price:
                # Cannot fill sell limit order below limit price
                return None
            else:
                # Fill at limit price if better
                execution_price = limit_price
                
        return execution_price
        
    def _calculate_slippage(self, symbol: str, quantity: int, side: str) -> float:
        """Calculate slippage based on order size and market conditions"""
        
        # Base slippage
        base_slippage = self.slippage_rate
        
        # Increase slippage for larger orders
        size_factor = min(quantity / 10000, 2.0)  # Cap at 2x for very large orders
        
        # Adjust for liquidity
        liquidity_adjustment = 1.0 / self.liquidity_factor
        
        # Adjust for volatility
        volatility_adjustment = self.volatility_factor
        
        # Random component to simulate market noise
        random_factor = np.random.uniform(0.5, 1.5)
        
        total_slippage = (base_slippage * size_factor * liquidity_adjustment * 
                         volatility_adjustment * random_factor)
        
        return min(total_slippage, 0.01)  # Cap at 1%
        
    def _calculate_market_impact(self, symbol: str, quantity: int) -> float:
        """Calculate market impact of large orders"""
        
        # Linear market impact model
        impact = self.market_impact_factor * quantity
        
        # Adjust for liquidity
        impact *= (1.0 / self.liquidity_factor)
        
        return min(impact, 0.005)  # Cap at 0.5%
        
    def _calculate_commission(self, order_value: float) -> float:
        """Calculate commission"""
        
        # Percentage-based commission
        commission = order_value * self.commission_rate
        
        # Minimum commission
        min_commission = 1.0  # $1 minimum
        
        return max(commission, min_commission)
        
    def _simulate_fill_quantity(self, symbol: str, quantity: int, side: str) -> int:
        """Simulate partial fills for large orders"""
        
        # Always fill small orders completely
        if quantity < 1000:
            return quantity
            
        # Simulate available liquidity
        market_depth = 100000 * self.liquidity_factor
        
        # Calculate fill probability based on order size
        fill_ratio = min(1.0, market_depth / quantity)
        
        # Add randomness
        if fill_ratio < 1.0:
            fill_ratio *= np.random.uniform(0.8, 1.0)
            
        filled_quantity = int(quantity * fill_ratio)
        
        return max(filled_quantity, 1)  # At least 1 unit
        
    def _validate_order(self, order: Dict[str, Any], market_price: float) -> Tuple[bool, Optional[str]]:
        """Validate order against risk limits"""
        
        symbol = order['symbol']
        side = order['side']
        quantity = order['quantity']
        
        # Check quantity
        if quantity <= 0:
            return False, "Invalid quantity"
            
        # Check position concentration
        order_value = quantity * market_price
        if order_value > self.capital * self.max_position_size:
            return False, f"Order exceeds maximum position size ({self.max_position_size*100}% of capital)"
            
        # Check leverage
        total_exposure = sum(abs(pos['quantity'] * self.market_prices.get(s, pos['avg_price'])) 
                           for s, pos in self.positions.items())
        
        if side == 'BUY':
            new_exposure = total_exposure + order_value
        else:
            # For sells, check if we have the position
            if symbol not in self.positions:
                return False, "No position to sell"
            if self.positions[symbol]['quantity'] < quantity:
                return False, "Insufficient position size"
            new_exposure = total_exposure
            
        if new_exposure > self.capital * self.max_leverage:
            return False, f"Order would exceed maximum leverage ({self.max_leverage}x)"
            
        return True, None
        
    def _open_or_add_position(self, fill: Dict[str, Any]):
        """Open new position or add to existing"""
        
        symbol = fill['symbol']
        
        if symbol in self.positions:
            # Add to existing position
            position = self.positions[symbol]
            total_cost = (position['quantity'] * position['avg_price'] + 
                         fill['quantity'] * fill['price'])
            total_quantity = position['quantity'] + fill['quantity']
            
            position['quantity'] = total_quantity
            position['avg_price'] = total_cost / total_quantity
            position['last_update'] = fill['timestamp']
            position['total_commission'] += fill['commission']
            
        else:
            # Create new position
            self.positions[symbol] = {
                'symbol': symbol,
                'quantity': fill['quantity'],
                'avg_price': fill['price'],
                'entry_time': fill['timestamp'],
                'last_update': fill['timestamp'],
                'total_commission': fill['commission'],
                'realized_pnl': 0.0,
                'trades': 1
            }
            
    def _close_or_reduce_position(self, fill: Dict[str, Any]) -> float:
        """Close or reduce position and return P&L"""
        
        symbol = fill['symbol']
        
        if symbol not in self.positions:
            logger.warning(f"No position to close for {symbol}")
            return 0.0
            
        position = self.positions[symbol]
        
        # Calculate P&L
        pnl = (fill['price'] - position['avg_price']) * min(fill['quantity'], position['quantity'])
        
        if fill['quantity'] >= position['quantity']:
            # Close entire position
            position['realized_pnl'] += pnl
            del self.positions[symbol]
        else:
            # Reduce position
            position['quantity'] -= fill['quantity']
            position['realized_pnl'] += pnl
            position['last_update'] = fill['timestamp']
            position['trades'] += 1
            
        return pnl
        
    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get current positions with unrealized P&L"""
        
        positions_with_pnl = {}
        
        for symbol, position in self.positions.items():
            pos_copy = position.copy()
            
            # Calculate unrealized P&L
            current_price = self.market_prices.get(symbol, position['avg_price'])
            unrealized_pnl = (current_price - position['avg_price']) * position['quantity']
            
            pos_copy['current_price'] = current_price
            pos_copy['market_value'] = current_price * position['quantity']
            pos_copy['unrealized_pnl'] = unrealized_pnl
            pos_copy['total_pnl'] = position['realized_pnl'] + unrealized_pnl
            
            positions_with_pnl[symbol] = pos_copy
            
        return positions_with_pnl
        
    def get_portfolio_value(self) -> float:
        """Get total portfolio value"""
        
        positions_value = sum(
            pos['quantity'] * self.market_prices.get(pos['symbol'], pos['avg_price'])
            for pos in self.positions.values()
        )
        
        return self.capital + positions_value
        
    def get_available_capital(self) -> float:
        """Get available capital for trading"""
        return self.capital
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        
        # Calculate metrics
        total_trades = len([t for t in self.trades if t['side'] == 'SELL'])
        winning_trades = len([t for t in self.trades if t.get('pnl', 0) > 0])
        losing_trades = len([t for t in self.trades if t.get('pnl', 0) < 0])
        
        total_pnl = sum(t.get('pnl', 0) for t in self.trades)
        portfolio_value = self.get_portfolio_value()
        
        return {
            'initial_capital': self.initial_capital,
            'current_capital': self.capital,
            'portfolio_value': portfolio_value,
            'total_pnl': total_pnl,
            'total_return': ((portfolio_value - self.initial_capital) / 
                           self.initial_capital * 100),
            'total_commission': self.total_commission,
            'total_slippage': self.total_slippage,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': winning_trades / total_trades * 100 if total_trades > 0 else 0,
            'positions_count': len(self.positions),
            'execution_stats': self.execution_stats
        }
        
    def _update_avg_slippage(self, slippage: float):
        """Update average slippage statistic"""
        
        filled = self.execution_stats['filled_orders']
        current_avg = self.execution_stats['avg_slippage']
        
        # Update moving average
        self.execution_stats['avg_slippage'] = (
            (current_avg * (filled - 1) + slippage) / filled
        )
        
    def reset(self, capital: Optional[float] = None):
        """Reset broker state"""
        
        self.capital = capital or self.initial_capital
        self.positions.clear()
        self.orders.clear()
        self.trades.clear()
        self.market_prices.clear()
        self.bid_ask_spreads.clear()
        
        self.total_commission = 0.0
        self.total_slippage = 0.0
        
        self.execution_stats = {
            'total_orders': 0,
            'filled_orders': 0,
            'rejected_orders': 0,
            'partial_fills': 0,
            'avg_slippage': 0.0,
            'avg_fill_time': 0.0
        }
        
        logger.info("Virtual broker reset")
        
    def set_market_conditions(self, liquidity_factor: float = 1.0, volatility_factor: float = 1.0):
        """Set market conditions for simulation"""
        
        self.liquidity_factor = max(0.1, min(10.0, liquidity_factor))
        self.volatility_factor = max(0.1, min(10.0, volatility_factor))
        
        logger.info(f"Market conditions set - Liquidity: {self.liquidity_factor}x, Volatility: {self.volatility_factor}x")