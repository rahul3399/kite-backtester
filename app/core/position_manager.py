# app/core/position_manager.py
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class Position:
    """Represents a trading position"""
    
    def __init__(self, symbol: str, quantity: int, avg_price: float, 
                 side: str, strategy_id: Optional[str] = None):
        self.symbol = symbol
        self.quantity = quantity
        self.avg_price = avg_price
        self.side = side  # LONG or SHORT
        self.strategy_id = strategy_id
        
        # Tracking
        self.entry_time = datetime.now()
        self.last_update = datetime.now()
        self.realized_pnl = 0.0
        self.commission = 0.0
        self.trades = []
        
        # Risk management
        self.stop_loss = None
        self.take_profit = None
        self.max_quantity = quantity
        
        # Performance
        self.high_water_mark = avg_price
        self.low_water_mark = avg_price
        
    def add_trade(self, quantity: int, price: float, commission: float = 0):
        """Add to position"""
        total_cost = (self.quantity * self.avg_price) + (quantity * price)
        self.quantity += quantity
        self.avg_price = total_cost / self.quantity if self.quantity > 0 else 0
        self.commission += commission
        self.max_quantity = max(self.max_quantity, abs(self.quantity))
        self.last_update = datetime.now()
        
        self.trades.append({
            'time': datetime.now(),
            'quantity': quantity,
            'price': price,
            'commission': commission
        })
        
    def reduce_position(self, quantity: int, price: float, commission: float = 0) -> float:
        """Reduce position and return realized PnL"""
        if abs(quantity) > abs(self.quantity):
            quantity = self.quantity
            
        # Calculate PnL
        if self.side == "LONG":
            pnl = (price - self.avg_price) * quantity
        else:  # SHORT
            pnl = (self.avg_price - price) * quantity
            
        self.quantity -= quantity
        self.realized_pnl += pnl
        self.commission += commission
        self.last_update = datetime.now()
        
        self.trades.append({
            'time': datetime.now(),
            'quantity': -quantity,
            'price': price,
            'commission': commission,
            'pnl': pnl
        })
        
        return pnl
        
    def get_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized PnL"""
        if self.quantity == 0:
            return 0.0
            
        if self.side == "LONG":
            return (current_price - self.avg_price) * self.quantity
        else:  # SHORT
            return (self.avg_price - current_price) * abs(self.quantity)
            
    def get_total_pnl(self, current_price: float) -> float:
        """Get total PnL (realized + unrealized)"""
        return self.realized_pnl + self.get_unrealized_pnl(current_price)
        
    def update_market_price(self, price: float):
        """Update high/low water marks"""
        if self.side == "LONG":
            self.high_water_mark = max(self.high_water_mark, price)
        else:
            self.low_water_mark = min(self.low_water_mark, price)
            
    def to_dict(self, current_price: Optional[float] = None) -> Dict[str, Any]:
        """Convert position to dictionary"""
        data = {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'avg_price': self.avg_price,
            'side': self.side,
            'strategy_id': self.strategy_id,
            'entry_time': self.entry_time,
            'last_update': self.last_update,
            'realized_pnl': self.realized_pnl,
            'commission': self.commission,
            'trade_count': len(self.trades),
            'max_quantity': self.max_quantity,
            'high_water_mark': self.high_water_mark,
            'low_water_mark': self.low_water_mark,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit
        }
        
        if current_price:
            data['current_price'] = current_price
            data['unrealized_pnl'] = self.get_unrealized_pnl(current_price)
            data['total_pnl'] = self.get_total_pnl(current_price)
            data['pnl_percentage'] = (data['total_pnl'] / (self.avg_price * self.max_quantity)) * 100
            
        return data

class PositionManager:
    """
    Manages portfolio positions and P&L tracking
    """
    
    def __init__(self, initial_capital: float = 1000000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        # Position tracking
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        
        # Strategy position mapping
        self.strategy_positions: Dict[str, List[str]] = defaultdict(list)
        
        # Market prices
        self.market_prices: Dict[str, float] = {}
        
        # Performance tracking
        self.equity_curve: List[Dict[str, Any]] = []
        self.daily_pnl: Dict[str, float] = defaultdict(float)
        
        # Risk parameters
        self.max_positions = 20
        self.max_position_size_pct = 0.25  # 25% of capital
        self.max_portfolio_risk_pct = 0.02  # 2% portfolio risk
        
        # Statistics
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_commission': 0.0,
            'gross_profit': 0.0,
            'gross_loss': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'consecutive_wins': 0,
            'consecutive_losses': 0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0
        }
        
    def open_position(self, symbol: str, quantity: int, price: float, 
                     side: str = "LONG", strategy_id: Optional[str] = None,
                     commission: float = 0.0) -> Tuple[bool, str]:
        """
        Open a new position or add to existing
        
        Returns:
            Tuple of (success, message)
        """
        
        # Validate
        if quantity <= 0:
            return False, "Quantity must be positive"
            
        # Check position limits
        if symbol not in self.positions and len(self.positions) >= self.max_positions:
            return False, f"Maximum {self.max_positions} positions exceeded"
            
        # Check position size
        position_value = quantity * price
        if position_value > self.current_capital * self.max_position_size_pct:
            return False, f"Position size exceeds {self.max_position_size_pct*100}% of capital"
            
        # Open or add to position
        if symbol in self.positions:
            position = self.positions[symbol]
            
            # Check if same direction
            if position.side != side:
                return False, "Cannot add to position in opposite direction"
                
            position.add_trade(quantity, price, commission)
            logger.info(f"Added to position: {symbol} +{quantity} @ {price}")
            
        else:
            # Create new position
            position = Position(symbol, quantity, price, side, strategy_id)
            position.commission = commission
            self.positions[symbol] = position
            
            if strategy_id:
                self.strategy_positions[strategy_id].append(symbol)
                
            logger.info(f"Opened position: {symbol} {quantity} @ {price}")
            
        # Update capital
        self.current_capital -= (position_value + commission)
        
        # Update statistics
        self.stats['total_trades'] += 1
        self.stats['total_commission'] += commission
        
        # Update market price
        self.market_prices[symbol] = price
        
        return True, f"Position opened: {symbol}"
        
    def close_position(self, symbol: str, quantity: Optional[int] = None, 
                      price: float = 0, commission: float = 0.0) -> Tuple[bool, str, float]:
        """
        Close position fully or partially
        
        Returns:
            Tuple of (success, message, pnl)
        """
        
        if symbol not in self.positions:
            return False, "Position not found", 0.0
            
        position = self.positions[symbol]
        
        # Use full quantity if not specified
        if quantity is None:
            quantity = abs(position.quantity)
        else:
            quantity = min(quantity, abs(position.quantity))
            
        # Get PnL
        pnl = position.reduce_position(quantity, price, commission)
        
        # Update capital
        self.current_capital += (quantity * price - commission)
        
        # Update statistics
        self._update_trade_statistics(pnl)
        self.stats['total_commission'] += commission
        
        # Update market price
        self.market_prices[symbol] = price
        
        # Check if position is fully closed
        if position.quantity == 0:
            # Move to closed positions
            self.closed_positions.append(position)
            del self.positions[symbol]
            
            # Remove from strategy mapping
            if position.strategy_id:
                self.strategy_positions[position.strategy_id].remove(symbol)
                
            logger.info(f"Closed position: {symbol} PnL: {pnl:.2f}")
            return True, f"Position closed: {symbol}", pnl
            
        else:
            logger.info(f"Partially closed position: {symbol} -{quantity} PnL: {pnl:.2f}")
            return True, f"Position partially closed: {symbol}", pnl
            
    def update_market_prices(self, prices: Dict[str, float]):
        """Update market prices for positions"""
        self.market_prices.update(prices)
        
        # Update position water marks
        for symbol, position in self.positions.items():
            if symbol in prices:
                position.update_market_price(prices[symbol])
                
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol"""
        return self.positions.get(symbol)
        
    def get_all_positions(self, include_closed: bool = False) -> List[Dict[str, Any]]:
        """Get all positions"""
        positions = []
        
        # Active positions
        for symbol, position in self.positions.items():
            current_price = self.market_prices.get(symbol, position.avg_price)
            positions.append(position.to_dict(current_price))
            
        # Closed positions if requested
        if include_closed:
            for position in self.closed_positions:
                positions.append(position.to_dict())
                
        return positions
        
    def get_strategy_positions(self, strategy_id: str) -> List[Dict[str, Any]]:
        """Get positions for a specific strategy"""
        positions = []
        
        for symbol in self.strategy_positions.get(strategy_id, []):
            if symbol in self.positions:
                position = self.positions[symbol]
                current_price = self.market_prices.get(symbol, position.avg_price)
                positions.append(position.to_dict(current_price))
                
        return positions
        
    def get_portfolio_value(self) -> float:
        """Get total portfolio value"""
        positions_value = 0.0
        
        for symbol, position in self.positions.items():
            current_price = self.market_prices.get(symbol, position.avg_price)
            positions_value += abs(position.quantity) * current_price
            
        return self.current_capital + positions_value
        
    def get_portfolio_metrics(self) -> Dict[str, Any]:
        """Get portfolio-level metrics"""
        
        # Calculate aggregate metrics
        total_unrealized_pnl = 0.0
        total_realized_pnl = sum(p.realized_pnl for p in self.positions.values())
        total_realized_pnl += sum(p.realized_pnl for p in self.closed_positions)
        
        positions_value = 0.0
        position_count = len(self.positions)
        
        for symbol, position in self.positions.items():
            current_price = self.market_prices.get(symbol, position.avg_price)
            unrealized_pnl = position.get_unrealized_pnl(current_price)
            total_unrealized_pnl += unrealized_pnl
            positions_value += abs(position.quantity) * current_price
            
        portfolio_value = self.get_portfolio_value()
        
        # Calculate returns
        total_return = ((portfolio_value - self.initial_capital) / 
                       self.initial_capital) * 100
        
        # Win rate
        total_trades = self.stats['winning_trades'] + self.stats['losing_trades']
        win_rate = (self.stats['winning_trades'] / total_trades * 100 
                   if total_trades > 0 else 0)
        
        # Profit factor
        profit_factor = (self.stats['gross_profit'] / abs(self.stats['gross_loss'])
                        if self.stats['gross_loss'] != 0 else 0)
        
        # Average win/loss
        avg_win = (self.stats['gross_profit'] / self.stats['winning_trades']
                  if self.stats['winning_trades'] > 0 else 0)
        avg_loss = (self.stats['gross_loss'] / self.stats['losing_trades']
                   if self.stats['losing_trades'] > 0 else 0)
        
        return {
            'portfolio_value': portfolio_value,
            'cash_balance': self.current_capital,
            'positions_value': positions_value,
            'position_count': position_count,
            'total_return': total_return,
            'total_return_pct': total_return,
            'realized_pnl': total_realized_pnl,
            'unrealized_pnl': total_unrealized_pnl,
            'total_pnl': total_realized_pnl + total_unrealized_pnl,
            'total_commission': self.stats['total_commission'],
            'total_trades': self.stats['total_trades'],
            'winning_trades': self.stats['winning_trades'],
            'losing_trades': self.stats['losing_trades'],
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'average_win': avg_win,
            'average_loss': avg_loss,
            'largest_win': self.stats['largest_win'],
            'largest_loss': self.stats['largest_loss'],
            'consecutive_wins': self.stats['consecutive_wins'],
            'consecutive_losses': self.stats['consecutive_losses'],
            'max_consecutive_wins': self.stats['max_consecutive_wins'],
            'max_consecutive_losses': self.stats['max_consecutive_losses']
        }
        
    def calculate_position_size(self, symbol: str, stop_loss_pct: float, 
                              risk_per_trade_pct: float = 0.02) -> int:
        """
        Calculate position size based on risk management
        
        Args:
            symbol: Trading symbol
            stop_loss_pct: Stop loss percentage (e.g., 0.02 for 2%)
            risk_per_trade_pct: Risk per trade as % of portfolio
            
        Returns:
            Position size in units
        """
        
        portfolio_value = self.get_portfolio_value()
        risk_amount = portfolio_value * risk_per_trade_pct
        
        current_price = self.market_prices.get(symbol, 100)
        stop_loss_amount = current_price * stop_loss_pct
        
        position_size = int(risk_amount / stop_loss_amount)
        
        # Apply position size limits
        max_position_value = portfolio_value * self.max_position_size_pct
        max_position_size = int(max_position_value / current_price)
        
        return min(position_size, max_position_size)
        
    def set_stop_loss(self, symbol: str, stop_loss: float) -> bool:
        """Set stop loss for position"""
        if symbol in self.positions:
            self.positions[symbol].stop_loss = stop_loss
            return True
        return False
        
    def set_take_profit(self, symbol: str, take_profit: float) -> bool:
        """Set take profit for position"""
        if symbol in self.positions:
            self.positions[symbol].take_profit = take_profit
            return True
        return False
        
    def check_stop_loss_take_profit(self) -> List[Dict[str, Any]]:
        """Check if any positions hit stop loss or take profit"""
        
        triggered = []
        
        for symbol, position in self.positions.items():
            current_price = self.market_prices.get(symbol)
            if not current_price:
                continue
                
            # Check stop loss
            if position.stop_loss:
                if ((position.side == "LONG" and current_price <= position.stop_loss) or
                    (position.side == "SHORT" and current_price >= position.stop_loss)):
                    triggered.append({
                        'symbol': symbol,
                        'type': 'stop_loss',
                        'trigger_price': position.stop_loss,
                        'current_price': current_price
                    })
                    
            # Check take profit
            if position.take_profit:
                if ((position.side == "LONG" and current_price >= position.take_profit) or
                    (position.side == "SHORT" and current_price <= position.take_profit)):
                    triggered.append({
                        'symbol': symbol,
                        'type': 'take_profit',
                        'trigger_price': position.take_profit,
                        'current_price': current_price
                    })
                    
        return triggered
        
    def record_equity_snapshot(self):
        """Record current equity snapshot"""
        
        metrics = self.get_portfolio_metrics()
        
        snapshot = {
            'timestamp': datetime.now(),
            'portfolio_value': metrics['portfolio_value'],
            'cash_balance': metrics['cash_balance'],
            'positions_value': metrics['positions_value'],
            'realized_pnl': metrics['realized_pnl'],
            'unrealized_pnl': metrics['unrealized_pnl'],
            'position_count': metrics['position_count'],
            'win_rate': metrics['win_rate']
        }
        
        self.equity_curve.append(snapshot)
        
        # Update daily PnL
        today = datetime.now().date()
        if self.equity_curve:
            yesterday_value = self.initial_capital
            for snapshot in reversed(self.equity_curve[:-1]):
                if snapshot['timestamp'].date() < today:
                    yesterday_value = snapshot['portfolio_value']
                    break
                    
            self.daily_pnl[str(today)] = metrics['portfolio_value'] - yesterday_value
            
    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve as DataFrame"""
        if not self.equity_curve:
            return pd.DataFrame()
            
        df = pd.DataFrame(self.equity_curve)
        df.set_index('timestamp', inplace=True)
        
        # Add returns
        df['returns'] = df['portfolio_value'].pct_change()
        df['cumulative_returns'] = (1 + df['returns']).cumprod() - 1
        
        # Add drawdown
        df['peak'] = df['portfolio_value'].cummax()
        df['drawdown'] = (df['portfolio_value'] - df['peak']) / df['peak']
        
        return df
        
    def _update_trade_statistics(self, pnl: float):
        """Update trade statistics"""
        
        if pnl > 0:
            self.stats['winning_trades'] += 1
            self.stats['gross_profit'] += pnl
            self.stats['consecutive_wins'] += 1
            self.stats['consecutive_losses'] = 0
            
            self.stats['max_consecutive_wins'] = max(
                self.stats['max_consecutive_wins'],
                self.stats['consecutive_wins']
            )
            
            if pnl > self.stats['largest_win']:
                self.stats['largest_win'] = pnl
                
        else:
            self.stats['losing_trades'] += 1
            self.stats['gross_loss'] += abs(pnl)
            self.stats['consecutive_losses'] += 1
            self.stats['consecutive_wins'] = 0
            
            self.stats['max_consecutive_losses'] = max(
                self.stats['max_consecutive_losses'],
                self.stats['consecutive_losses']
            )
            
            if pnl < self.stats['largest_loss']:
                self.stats['largest_loss'] = pnl
                
    def reset(self):
        """Reset position manager"""
        
        self.current_capital = self.initial_capital
        self.positions.clear()
        self.closed_positions.clear()
        self.strategy_positions.clear()
        self.market_prices.clear()
        self.equity_curve.clear()
        self.daily_pnl.clear()
        
        # Reset statistics
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_commission': 0.0,
            'gross_profit': 0.0,
            'gross_loss': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'consecutive_wins': 0,
            'consecutive_losses': 0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0
        }
        
        logger.info("Position manager reset")