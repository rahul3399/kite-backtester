# app/core/order_manager.py
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import logging
import uuid
from collections import defaultdict
import pandas as pd

from ..strategies.base_strategy import OrderType, OrderSide

logger = logging.getLogger(__name__)

class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "PENDING"
    OPEN = "OPEN"
    COMPLETE = "COMPLETE"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    TRIGGER_PENDING = "TRIGGER_PENDING"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"

class OrderManager:
    """
    Manages order lifecycle, execution, and tracking
    Handles both live and paper trading orders
    """
    
    def __init__(self, kite_client=None, mode: str = "paper"):
        """
        Initialize order manager
        
        Args:
            kite_client: KiteClient instance for live trading
            mode: Trading mode (paper/live)
        """
        self.kite_client = kite_client
        self.mode = mode
        
        # Order tracking
        self.orders: Dict[str, Dict[str, Any]] = {}
        self.order_history: List[Dict[str, Any]] = []
        
        # Order queues by status
        self.pending_orders: List[str] = []
        self.open_orders: List[str] = []
        
        # Strategy order mapping
        self.strategy_orders: Dict[str, List[str]] = defaultdict(list)
        
        # Performance tracking
        self.order_stats = {
            'total_orders': 0,
            'filled_orders': 0,
            'rejected_orders': 0,
            'cancelled_orders': 0,
            'avg_fill_time': 0
        }
        
        # Risk management
        self.max_orders_per_symbol = 5
        self.max_orders_per_minute = 20
        self.order_timestamps: List[datetime] = []
        
    async def place_order(self,
                        symbol: str,
                        side: OrderSide,
                        quantity: int,
                        order_type: OrderType,
                        price: Optional[float] = None,
                        trigger_price: Optional[float] = None,
                        stop_loss: Optional[float] = None,
                        take_profit: Optional[float] = None,
                        strategy_id: Optional[str] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
        """
        Place an order
        
        Returns:
            Tuple of (success, order_id/error_message)
        """
        
        # Validate order
        is_valid, error = self._validate_order(
            symbol, side, quantity, order_type, price
        )
        
        if not is_valid:
            logger.error(f"Order validation failed: {error}")
            return False, error
            
        # Check rate limits
        if not self._check_rate_limit():
            return False, "Rate limit exceeded"
            
        # Create order
        order_id = str(uuid.uuid4())
        order = {
            'order_id': order_id,
            'symbol': symbol,
            'side': side.value,
            'quantity': quantity,
            'order_type': order_type.value,
            'price': price,
            'trigger_price': trigger_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'strategy_id': strategy_id,
            'metadata': metadata or {},
            'status': OrderStatus.PENDING.value,
            'filled_quantity': 0,
            'avg_fill_price': None,
            'placed_time': datetime.now(),
            'updated_time': datetime.now(),
            'fills': []
        }
        
        # Store order
        self.orders[order_id] = order
        self.pending_orders.append(order_id)
        
        if strategy_id:
            self.strategy_orders[strategy_id].append(order_id)
            
        # Update stats
        self.order_stats['total_orders'] += 1
        self.order_timestamps.append(datetime.now())
        
        # Execute order based on mode
        if self.mode == "live":
            success = await self._place_live_order(order)
        else:
            success = await self._place_paper_order(order)
            
        if success:
            logger.info(f"Order placed successfully: {order_id}")
            return True, order_id
        else:
            self.orders[order_id]['status'] = OrderStatus.REJECTED.value
            self.order_stats['rejected_orders'] += 1
            return False, "Order placement failed"
            
    async def _place_live_order(self, order: Dict[str, Any]) -> bool:
        """Place order through Kite API"""
        
        if not self.kite_client:
            logger.error("Kite client not initialized for live trading")
            return False
            
        try:
            # Map internal order to Kite order format
            kite_order = self._map_to_kite_order(order)
            
            # Place order through Kite
            kite_order_id = await asyncio.get_event_loop().run_in_executor(
                None,
                self.kite_client.place_order,
                **kite_order
            )
            
            # Update order with Kite order ID
            order['kite_order_id'] = kite_order_id
            order['status'] = OrderStatus.OPEN.value
            
            # Move to open orders
            self.pending_orders.remove(order['order_id'])
            self.open_orders.append(order['order_id'])
            
            return True
            
        except Exception as e:
            logger.error(f"Error placing live order: {e}")
            return False
            
    async def _place_paper_order(self, order: Dict[str, Any]) -> bool:
        """Place paper trading order"""
        
        # In paper trading, orders are immediately accepted
        order['status'] = OrderStatus.OPEN.value
        order['updated_time'] = datetime.now()
        
        # Move to open orders
        self.pending_orders.remove(order['order_id'])
        self.open_orders.append(order['order_id'])
        
        # Simulate order fill for market orders
        if order['order_type'] == OrderType.MARKET.value:
            # Immediate fill simulation
            await asyncio.sleep(0.1)  # Simulate execution delay
            await self._simulate_fill(order['order_id'])
            
        return True
        
    async def _simulate_fill(self, order_id: str):
        """Simulate order fill for paper trading"""
        
        order = self.orders.get(order_id)
        if not order:
            return
            
        # Create fill
        fill = {
            'fill_id': str(uuid.uuid4()),
            'quantity': order['quantity'],
            'price': order['price'] or 100.0,  # Use provided price or default
            'timestamp': datetime.now()
        }
        
        # Update order
        order['fills'].append(fill)
        order['filled_quantity'] = order['quantity']
        order['avg_fill_price'] = fill['price']
        order['status'] = OrderStatus.COMPLETE.value
        order['updated_time'] = datetime.now()
        
        # Update tracking
        if order_id in self.open_orders:
            self.open_orders.remove(order_id)
            
        # Update stats
        self.order_stats['filled_orders'] += 1
        fill_time = (order['updated_time'] - order['placed_time']).total_seconds()
        self._update_avg_fill_time(fill_time)
        
        # Add to history
        self.order_history.append(order.copy())
        
        logger.info(f"Order filled: {order_id}")
        
    def modify_order(self,
                    order_id: str,
                    quantity: Optional[int] = None,
                    price: Optional[float] = None,
                    trigger_price: Optional[float] = None) -> Tuple[bool, str]:
        """
        Modify an existing order
        
        Returns:
            Tuple of (success, message)
        """
        
        order = self.orders.get(order_id)
        if not order:
            return False, "Order not found"
            
        if order['status'] not in [OrderStatus.PENDING.value, OrderStatus.OPEN.value]:
            return False, f"Cannot modify order in {order['status']} status"
            
        # Update order fields
        if quantity is not None:
            order['quantity'] = quantity
        if price is not None:
            order['price'] = price
        if trigger_price is not None:
            order['trigger_price'] = trigger_price
            
        order['updated_time'] = datetime.now()
        
        # For live trading, modify through Kite
        if self.mode == "live" and 'kite_order_id' in order:
            try:
                self.kite_client.modify_order(
                    variety="regular",
                    order_id=order['kite_order_id'],
                    quantity=quantity,
                    price=price,
                    trigger_price=trigger_price
                )
            except Exception as e:
                logger.error(f"Error modifying live order: {e}")
                return False, str(e)
                
        logger.info(f"Order modified: {order_id}")
        return True, "Order modified successfully"
        
    def cancel_order(self, order_id: str) -> Tuple[bool, str]:
        """
        Cancel an order
        
        Returns:
            Tuple of (success, message)
        """
        
        order = self.orders.get(order_id)
        if not order:
            return False, "Order not found"
            
        if order['status'] not in [OrderStatus.PENDING.value, OrderStatus.OPEN.value]:
            return False, f"Cannot cancel order in {order['status']} status"
            
        # For live trading, cancel through Kite
        if self.mode == "live" and 'kite_order_id' in order:
            try:
                self.kite_client.cancel_order(
                    variety="regular",
                    order_id=order['kite_order_id']
                )
            except Exception as e:
                logger.error(f"Error cancelling live order: {e}")
                return False, str(e)
                
        # Update order status
        order['status'] = OrderStatus.CANCELLED.value
        order['updated_time'] = datetime.now()
        
        # Update tracking
        if order_id in self.pending_orders:
            self.pending_orders.remove(order_id)
        if order_id in self.open_orders:
            self.open_orders.remove(order_id)
            
        # Update stats
        self.order_stats['cancelled_orders'] += 1
        
        # Add to history
        self.order_history.append(order.copy())
        
        logger.info(f"Order cancelled: {order_id}")
        return True, "Order cancelled successfully"
        
    def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get order details"""
        return self.orders.get(order_id)
        
    def get_orders(self, 
                  status: Optional[OrderStatus] = None,
                  symbol: Optional[str] = None,
                  strategy_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get filtered orders"""
        
        orders = list(self.orders.values())
        
        # Apply filters
        if status:
            orders = [o for o in orders if o['status'] == status.value]
        if symbol:
            orders = [o for o in orders if o['symbol'] == symbol]
        if strategy_id:
            orders = [o for o in orders if o['strategy_id'] == strategy_id]
            
        return orders
        
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get open orders"""
        open_order_ids = self.pending_orders + self.open_orders
        orders = [self.orders[oid] for oid in open_order_ids if oid in self.orders]
        
        if symbol:
            orders = [o for o in orders if o['symbol'] == symbol]
            
        return orders
        
    def get_order_history(self, 
                         symbol: Optional[str] = None,
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get order history with filters"""
        
        history = self.order_history.copy()
        
        # Apply filters
        if symbol:
            history = [o for o in history if o['symbol'] == symbol]
        if start_date:
            history = [o for o in history if o['placed_time'] >= start_date]
        if end_date:
            history = [o for o in history if o['placed_time'] <= end_date]
            
        return history
        
    def get_fills(self, order_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get order fills"""
        
        if order_id:
            order = self.orders.get(order_id)
            return order['fills'] if order else []
            
        # Get all fills
        all_fills = []
        for order in self.orders.values():
            for fill in order['fills']:
                fill_data = fill.copy()
                fill_data['order_id'] = order['order_id']
                fill_data['symbol'] = order['symbol']
                all_fills.append(fill_data)
                
        return all_fills
        
    def update_order_status(self, order_id: str, status: OrderStatus, 
                          fill_data: Optional[Dict[str, Any]] = None):
        """Update order status (for external updates)"""
        
        order = self.orders.get(order_id)
        if not order:
            return
            
        order['status'] = status.value
        order['updated_time'] = datetime.now()
        
        # Handle fill data
        if fill_data:
            order['fills'].append(fill_data)
            order['filled_quantity'] += fill_data['quantity']
            
            # Calculate average fill price
            total_value = sum(f['quantity'] * f['price'] for f in order['fills'])
            total_quantity = sum(f['quantity'] for f in order['fills'])
            order['avg_fill_price'] = total_value / total_quantity if total_quantity > 0 else 0
            
        # Update tracking lists
        if status == OrderStatus.COMPLETE:
            if order_id in self.open_orders:
                self.open_orders.remove(order_id)
            self.order_stats['filled_orders'] += 1
            
        elif status in [OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            if order_id in self.open_orders:
                self.open_orders.remove(order_id)
            if order_id in self.pending_orders:
                self.pending_orders.remove(order_id)
                
        # Add to history if terminal state
        if status in [OrderStatus.COMPLETE, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            self.order_history.append(order.copy())
            
    def _validate_order(self, 
                       symbol: str,
                       side: OrderSide,
                       quantity: int,
                       order_type: OrderType,
                       price: Optional[float]) -> Tuple[bool, Optional[str]]:
        """Validate order parameters"""
        
        # Basic validation
        if quantity <= 0:
            return False, "Quantity must be positive"
            
        if order_type == OrderType.LIMIT and price is None:
            return False, "Limit orders require a price"
            
        if price is not None and price <= 0:
            return False, "Price must be positive"
            
        # Check max orders per symbol
        symbol_orders = [o for o in self.open_orders 
                        if self.orders[o]['symbol'] == symbol]
        if len(symbol_orders) >= self.max_orders_per_symbol:
            return False, f"Maximum {self.max_orders_per_symbol} orders per symbol exceeded"
            
        return True, None
        
    def _check_rate_limit(self) -> bool:
        """Check if rate limit is exceeded"""
        
        # Remove old timestamps
        cutoff_time = datetime.now() - timedelta(minutes=1)
        self.order_timestamps = [ts for ts in self.order_timestamps if ts > cutoff_time]
        
        # Check limit
        return len(self.order_timestamps) < self.max_orders_per_minute
        
    def _map_to_kite_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """Map internal order format to Kite order format"""
        
        # Map order type
        kite_order_type = {
            OrderType.MARKET.value: "MARKET",
            OrderType.LIMIT.value: "LIMIT",
            OrderType.STOP_LOSS.value: "SL",
            OrderType.STOP_LOSS_LIMIT.value: "SL"
        }.get(order['order_type'], "MARKET")
        
        # Map transaction type
        transaction_type = "BUY" if order['side'] == OrderSide.BUY.value else "SELL"
        
        kite_order = {
            'variety': "regular",
            'exchange': "NSE",  # Default, should be determined from symbol
            'tradingsymbol': order['symbol'],
            'transaction_type': transaction_type,
            'quantity': order['quantity'],
            'product': "MIS",  # Intraday, should be configurable
            'order_type': kite_order_type
        }
        
        if order['price']:
            kite_order['price'] = order['price']
            
        if order['trigger_price']:
            kite_order['trigger_price'] = order['trigger_price']
            
        return kite_order
        
    def _update_avg_fill_time(self, fill_time: float):
        """Update average fill time statistic"""
        
        current_avg = self.order_stats['avg_fill_time']
        filled_count = self.order_stats['filled_orders']
        
        # Calculate new average
        self.order_stats['avg_fill_time'] = (
            (current_avg * (filled_count - 1) + fill_time) / filled_count
        )
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get order manager statistics"""
        
        stats = self.order_stats.copy()
        
        # Add current state
        stats['pending_orders'] = len(self.pending_orders)
        stats['open_orders'] = len(self.open_orders)
        
        # Calculate fill rate
        total = stats['total_orders']
        if total > 0:
            stats['fill_rate'] = stats['filled_orders'] / total * 100
            stats['rejection_rate'] = stats['rejected_orders'] / total * 100
            stats['cancellation_rate'] = stats['cancelled_orders'] / total * 100
        
        return stats
        
    def reset(self):
        """Reset order manager state"""
        
        self.orders.clear()
        self.order_history.clear()
        self.pending_orders.clear()
        self.open_orders.clear()
        self.strategy_orders.clear()
        self.order_timestamps.clear()
        
        self.order_stats = {
            'total_orders': 0,
            'filled_orders': 0,
            'rejected_orders': 0,
            'cancelled_orders': 0,
            'avg_fill_time': 0
        }
        
        logger.info("Order manager reset")