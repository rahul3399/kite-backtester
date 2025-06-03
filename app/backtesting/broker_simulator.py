# app/backtesting/broker_simulator.py
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import pandas as pd
from collections import defaultdict
import uuid
import logging

logger = logging.getLogger(__name__)

class BrokerSimulator:
    """
    Simulates broker functionality for backtesting
    Handles order execution, position management, and portfolio tracking
    """
    
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.reset(initial_capital)
        
    def reset(self, capital: float):
        """Reset broker state"""
        self.capital = capital
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.orders: List[Dict[str, Any]] = []
        self.trades: List[Dict[str, Any]] = []
        self.equity_curve: List[Dict[str, Any]] = []
        self.market_prices: Dict[str, float] = {}
        self.current_time = None
        self.commission_rate = 0.0002
        self.slippage_rate = 0.0001
        
        # Performance tracking
        self.total_commission = 0.0
        self.total_slippage = 0.0
        self.max_positions = 0
        self.position_history: List[Dict[str, Any]] = []
        
    def set_commission(self, rate: float):
        """Set commission rate"""
        self.commission_rate = max(0, min(rate, 0.01))  # Cap at 1%
        
    def set_slippage(self, rate: float):
        """Set slippage rate"""
        self.slippage_rate = max(0, min(rate, 0.01))  # Cap at 1%
        
    def set_current_time(self, timestamp: datetime):
        """Set current simulation time"""
        self.current_time = timestamp
        
    def update_market_price(self, symbol: str, price: float):
        """Update market price for symbol"""
        if price <= 0:
            logger.warning(f"Invalid price {price} for {symbol}")
            return
        self.market_prices[symbol] = price
        
    def execute_order(self, order: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Execute order and return fill details
        
        Args:
            order: Order details including symbol, side, quantity, type
            
        Returns:
            Fill details if successful, None otherwise
        """
        
        symbol = order['symbol']
        side = order['side']
        quantity = order['quantity']
        order_type = order.get('order_type', 'MARKET')
        
        # Validate order
        if quantity <= 0:
            logger.warning(f"Invalid quantity {quantity} for order")
            return None
        
        # Get execution price with slippage
        market_price = self.market_prices.get(symbol)
        if not market_price or market_price <= 0:
            logger.warning(f"No valid market price for {symbol}")
            return None
            
        # Apply slippage based on order side
        if side == 'BUY':
            execution_price = market_price * (1 + self.slippage_rate)
        else:
            execution_price = market_price * (1 - self.slippage_rate)
            
        # Handle limit orders
        if order_type == 'LIMIT' and 'price' in order:
            limit_price = order['price']
            if side == 'BUY' and execution_price > limit_price:
                # Buy limit order - only execute if market price <= limit
                return None
            elif side == 'SELL' and execution_price < limit_price:
                # Sell limit order - only execute if market price >= limit
                return None
            execution_price = limit_price
            
        # Calculate costs
        order_value = execution_price * quantity
        commission = order_value * self.commission_rate
        slippage_cost = abs(execution_price - market_price) * quantity
        
        # Check capital for buy orders
        if side == 'BUY':
            required_capital = order_value + commission
            if required_capital > self.capital:
                logger.warning(f"Insufficient capital. Required: {required_capital}, Available: {self.capital}")
                return None
                
        # Create fill record
        fill = {
            'order_id': str(uuid.uuid4()),
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': execution_price,
            'market_price': market_price,
            'commission': commission,
            'slippage': slippage_cost,
            'timestamp': self.current_time,
            'order_type': order_type,
            'metadata': order.get('metadata', {})
        }
        
        # Update positions
        if side == 'BUY':
            self._open_position(fill)
        else:
            self._close_position(fill)
            
        # Record order and trade
        order['order_id'] = fill['order_id']
        order['status'] = 'FILLED'
        order['filled_quantity'] = quantity
        order['avg_fill_price'] = execution_price
        self.orders.append(order)
        self.trades.append(fill)
        
        # Update totals
        self.total_commission += commission
        self.total_slippage += slippage_cost
        
        return fill
        
    def _open_position(self, fill: Dict[str, Any]):
        """Open or add to position"""
        symbol = fill['symbol']
        
        if symbol in self.positions:
            # Add to existing position (averaging)
            pos = self.positions[symbol]
            total_quantity = pos['quantity'] + fill['quantity']
            total_cost = (pos['quantity'] * pos['avg_price']) + (fill['quantity'] * fill['price'])
            
            pos['quantity'] = total_quantity
            pos['avg_price'] = total_cost / total_quantity if total_quantity > 0 else 0
            pos['commission'] += fill['commission']
            pos['last_update'] = fill['timestamp']
            
        else:
            # Create new position
            self.positions[symbol] = {
                'symbol': symbol,
                'quantity': fill['quantity'],
                'avg_price': fill['price'],
                'entry_time': fill['timestamp'],
                'last_update': fill['timestamp'],
                'commission': fill['commission'],
                'realized_pnl': 0,
                'trades': 1
            }
            
        # Deduct capital
        self.capital -= (fill['quantity'] * fill['price'] + fill['commission'])
        
        # Update max positions
        self.max_positions = max(self.max_positions, len(self.positions))
        
        # Record position snapshot
        self._record_position_snapshot()
        
    def _close_position(self, fill: Dict[str, Any]):
        """Close or reduce position"""
        symbol = fill['symbol']
        
        if symbol not in self.positions:
            logger.warning(f"No position to close for {symbol}")
            return
            
        pos = self.positions[symbol]
        
        # Calculate P&L
        pnl = (fill['price'] - pos['avg_price']) * min(fill['quantity'], pos['quantity'])
        pnl_pct = (pnl / (pos['avg_price'] * min(fill['quantity'], pos['quantity']))) * 100 if pos['avg_price'] > 0 else 0
        
        if fill['quantity'] >= pos['quantity']:
            # Close entire position
            self.capital += (pos['quantity'] * fill['price'] - fill['commission'])
            
            # Record final position stats
            pos['exit_time'] = fill['timestamp']
            pos['exit_price'] = fill['price']
            pos['realized_pnl'] += pnl
            pos['total_commission'] = pos['commission'] + fill['commission']
            pos['holding_period'] = (fill['timestamp'] - pos['entry_time']).total_seconds() / 3600  # Hours
            
            # Add to position history
            self.position_history.append(pos.copy())
            
            # Remove from active positions
            del self.positions[symbol]
            
        else:
            # Partial close
            pos['quantity'] -= fill['quantity']
            pos['realized_pnl'] += pnl
            pos['commission'] += fill['commission']
            pos['last_update'] = fill['timestamp']
            pos['trades'] += 1
            
            self.capital += (fill['quantity'] * fill['price'] - fill['commission'])
            
        # Add P&L to fill record
        fill['pnl'] = pnl
        fill['pnl_pct'] = pnl_pct
        
        # Record position snapshot
        self._record_position_snapshot()
        
    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get current positions with unrealized P&L"""
        positions_with_pnl = {}
        
        for symbol, pos in self.positions.items():
            pos_copy = pos.copy()
            
            # Calculate unrealized P&L
            current_price = self.market_prices.get(symbol, pos['avg_price'])
            unrealized_pnl = (current_price - pos['avg_price']) * pos['quantity']
            unrealized_pnl_pct = (unrealized_pnl / (pos['avg_price'] * pos['quantity'])) * 100 if pos['avg_price'] > 0 else 0
            
            pos_copy['current_price'] = current_price
            pos_copy['market_value'] = current_price * pos['quantity']
            pos_copy['unrealized_pnl'] = unrealized_pnl
            pos_copy['unrealized_pnl_pct'] = unrealized_pnl_pct
            pos_copy['total_pnl'] = pos['realized_pnl'] + unrealized_pnl
            
            positions_with_pnl[symbol] = pos_copy
            
        return positions_with_pnl
        
    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get position for specific symbol"""
        if symbol in self.positions:
            positions = self.get_positions()
            return positions[symbol]
        return None
        
    def get_available_capital(self) -> float:
        """Get available capital for trading"""
        return self.capital
        
    def get_portfolio_value(self) -> float:
        """Get total portfolio value (capital + positions)"""
        positions_value = sum(
            pos['quantity'] * self.market_prices.get(pos['symbol'], pos['avg_price'])
            for pos in self.positions.values()
        )
        return self.capital + positions_value
        
    def get_margin_used(self) -> float:
        """Get margin used by positions"""
        return sum(
            pos['quantity'] * pos['avg_price']
            for pos in self.positions.values()
        )
        
    def record_equity(self):
        """Record current equity snapshot"""
        portfolio_value = self.get_portfolio_value()
        
        equity_point = {
            'timestamp': self.current_time,
            'capital': self.capital,
            'portfolio_value': portfolio_value,
            'positions_count': len(self.positions),
            'positions_value': portfolio_value - self.capital,
            'margin_used': self.get_margin_used(),
            'total_commission': self.total_commission,
            'total_slippage': self.total_slippage
        }
        
        # Calculate returns
        if self.equity_curve:
            prev_value = self.equity_curve[-1]['portfolio_value']
            equity_point['daily_return'] = (portfolio_value - prev_value) / prev_value if prev_value > 0 else 0
        else:
            equity_point['daily_return'] = 0
            
        equity_point['total_return'] = (portfolio_value - self.initial_capital) / self.initial_capital
        
        self.equity_curve.append(equity_point)
        
    def _record_position_snapshot(self):
        """Record snapshot of all positions"""
        # This can be used for position-level analysis
        snapshot = {
            'timestamp': self.current_time,
            'positions': self.get_positions().copy(),
            'capital': self.capital,
            'portfolio_value': self.get_portfolio_value()
        }
        
    def get_trades(self) -> List[Dict[str, Any]]:
        """Get all executed trades"""
        return self.trades.copy()
        
    def get_orders(self) -> List[Dict[str, Any]]:
        """Get all orders"""
        return self.orders.copy()
        
    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve as DataFrame"""
        if not self.equity_curve:
            return pd.DataFrame()
            
        df = pd.DataFrame(self.equity_curve)
        df.set_index('timestamp', inplace=True)
        
        # Add additional metrics
        if not df.empty:
            df['drawdown'] = (df['portfolio_value'] - df['portfolio_value'].cummax()) / df['portfolio_value'].cummax()
            df['cumulative_return'] = (df['portfolio_value'] - self.initial_capital) / self.initial_capital
            
        return df
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics"""
        
        # Get completed trades
        completed_trades = [t for t in self.trades if 'pnl' in t]
        winning_trades = [t for t in completed_trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in completed_trades if t.get('pnl', 0) < 0]
        
        # Calculate summary
        summary = {
            'initial_capital': self.initial_capital,
            'final_capital': self.capital,
            'portfolio_value': self.get_portfolio_value(),
            'total_return': (self.get_portfolio_value() - self.initial_capital) / self.initial_capital * 100,
            'total_trades': len(self.trades),
            'completed_trades': len(completed_trades),
            'open_positions': len(self.positions),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(completed_trades) * 100 if completed_trades else 0,
            'total_commission': self.total_commission,
            'total_slippage': self.total_slippage,
            'max_positions_held': self.max_positions,
            'avg_position_size': sum(t['quantity'] * t['price'] for t in self.trades) / len(self.trades) if self.trades else 0
        }
        
        # Add P&L statistics
        if completed_trades:
            pnls = [t['pnl'] for t in completed_trades]
            summary['total_pnl'] = sum(pnls)
            summary['avg_pnl'] = sum(pnls) / len(pnls)
            summary['max_win'] = max(pnls) if pnls else 0
            summary['max_loss'] = min(pnls) if pnls else 0
            
            if winning_trades:
                summary['avg_win'] = sum(t['pnl'] for t in winning_trades) / len(winning_trades)
            else:
                summary['avg_win'] = 0
                
            if losing_trades:
                summary['avg_loss'] = sum(t['pnl'] for t in losing_trades) / len(losing_trades)
            else:
                summary['avg_loss'] = 0
                
        return summary
        
    def validate_portfolio_constraints(self, symbol: str, quantity: int, side: str) -> Tuple[bool, Optional[str]]:
        """
        Validate portfolio constraints before order execution
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        
        # Position concentration limit
        max_position_pct = 0.25  # Max 25% in one position
        
        if side == 'BUY':
            position_value = quantity * self.market_prices.get(symbol, 0)
            portfolio_value = self.get_portfolio_value()
            
            if portfolio_value > 0:
                position_pct = position_value / portfolio_value
                if position_pct > max_position_pct:
                    return False, f"Position would exceed {max_position_pct*100}% concentration limit"
                    
        return True, None