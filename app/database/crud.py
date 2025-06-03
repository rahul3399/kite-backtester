# app/database/crud.py
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, func
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import json
import uuid

from . import models

# Strategy CRUD operations

def create_strategy(db: Session, 
                   name: str,
                   class_name: str,
                   module_path: str,
                   description: Optional[str] = None,
                   parameters_schema: Optional[Dict] = None,
                   required_indicators: Optional[List[str]] = None) -> models.Strategy:
    """Create a new strategy"""
    db_strategy = models.Strategy(
        name=name,
        class_name=class_name,
        module_path=module_path,
        description=description,
        parameters_schema=parameters_schema,
        required_indicators=required_indicators or []
    )
    db.add(db_strategy)
    db.commit()
    db.refresh(db_strategy)
    return db_strategy

def get_strategy(db: Session, strategy_id: int) -> Optional[models.Strategy]:
    """Get strategy by ID"""
    return db.query(models.Strategy).filter(models.Strategy.id == strategy_id).first()

def get_strategy_by_name(db: Session, name: str) -> Optional[models.Strategy]:
    """Get strategy by name"""
    return db.query(models.Strategy).filter(models.Strategy.name == name).first()

def get_strategies(db: Session, 
                  skip: int = 0, 
                  limit: int = 100,
                  is_active: Optional[bool] = None) -> List[models.Strategy]:
    """Get list of strategies"""
    query = db.query(models.Strategy)
    
    if is_active is not None:
        query = query.filter(models.Strategy.is_active == is_active)
        
    return query.offset(skip).limit(limit).all()

def update_strategy(db: Session,
                   strategy_id: int,
                   **kwargs) -> Optional[models.Strategy]:
    """Update strategy"""
    strategy = get_strategy(db, strategy_id)
    if not strategy:
        return None
        
    for key, value in kwargs.items():
        if hasattr(strategy, key):
            setattr(strategy, key, value)
            
    db.commit()
    db.refresh(strategy)
    return strategy

def delete_strategy(db: Session, strategy_id: int) -> bool:
    """Delete strategy"""
    strategy = get_strategy(db, strategy_id)
    if not strategy:
        return False
        
    db.delete(strategy)
    db.commit()
    return True

# Strategy Instance CRUD operations

def create_strategy_instance(db: Session,
                           strategy_id: int,
                           name: str,
                           symbols: List[str],
                           parameters: Dict[str, Any],
                           mode: str = "paper",
                           capital_allocation: Optional[float] = None,
                           risk_per_trade: float = 0.02,
                           max_positions: int = 5) -> models.StrategyInstance:
    """Create a new strategy instance"""
    db_instance = models.StrategyInstance(
        instance_id=str(uuid.uuid4()),
        strategy_id=strategy_id,
        name=name,
        symbols=symbols,
        parameters=parameters,
        mode=mode,
        capital_allocation=capital_allocation,
        risk_per_trade=risk_per_trade,
        max_positions=max_positions,
        status="running"
    )
    db.add(db_instance)
    db.commit()
    db.refresh(db_instance)
    return db_instance

def get_strategy_instance(db: Session, instance_id: str) -> Optional[models.StrategyInstance]:
    """Get strategy instance by ID"""
    return db.query(models.StrategyInstance).filter(
        models.StrategyInstance.instance_id == instance_id
    ).first()

def get_strategy_instances(db: Session,
                         skip: int = 0,
                         limit: int = 100,
                         status: Optional[str] = None,
                         mode: Optional[str] = None) -> List[models.StrategyInstance]:
    """Get list of strategy instances"""
    query = db.query(models.StrategyInstance)
    
    if status:
        query = query.filter(models.StrategyInstance.status == status)
    if mode:
        query = query.filter(models.StrategyInstance.mode == mode)
        
    return query.order_by(desc(models.StrategyInstance.started_at)).offset(skip).limit(limit).all()

def update_strategy_instance(db: Session,
                           instance_id: str,
                           **kwargs) -> Optional[models.StrategyInstance]:
    """Update strategy instance"""
    instance = get_strategy_instance(db, instance_id)
    if not instance:
        return None
        
    for key, value in kwargs.items():
        if hasattr(instance, key):
            setattr(instance, key, value)
            
    instance.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(instance)
    return instance

def stop_strategy_instance(db: Session, instance_id: str) -> Optional[models.StrategyInstance]:
    """Stop a strategy instance"""
    return update_strategy_instance(
        db, 
        instance_id,
        status="stopped",
        stopped_at=datetime.utcnow()
    )

# Backtest CRUD operations

def create_backtest(db: Session,
                   strategy_id: int,
                   symbols: List[str],
                   start_date: datetime,
                   end_date: datetime,
                   initial_capital: float,
                   parameters: Dict[str, Any],
                   commission: float = 0.0002,
                   slippage: float = 0.0001) -> models.Backtest:
    """Create a new backtest"""
    db_backtest = models.Backtest(
        backtest_id=str(uuid.uuid4()),
        strategy_id=strategy_id,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        commission=commission,
        slippage=slippage,
        parameters=parameters,
        status="pending"
    )
    db.add(db_backtest)
    db.commit()
    db.refresh(db_backtest)
    return db_backtest

def get_backtest(db: Session, backtest_id: str) -> Optional[models.Backtest]:
    """Get backtest by ID"""
    return db.query(models.Backtest).filter(
        models.Backtest.backtest_id == backtest_id
    ).first()

def get_backtests(db: Session,
                 skip: int = 0,
                 limit: int = 100,
                 strategy_id: Optional[int] = None,
                 status: Optional[str] = None) -> List[models.Backtest]:
    """Get list of backtests"""
    query = db.query(models.Backtest)
    
    if strategy_id:
        query = query.filter(models.Backtest.strategy_id == strategy_id)
    if status:
        query = query.filter(models.Backtest.status == status)
        
    return query.order_by(desc(models.Backtest.created_at)).offset(skip).limit(limit).all()

def update_backtest(db: Session,
                   backtest_id: str,
                   **kwargs) -> Optional[models.Backtest]:
    """Update backtest"""
    backtest = get_backtest(db, backtest_id)
    if not backtest:
        return None
        
    for key, value in kwargs.items():
        if hasattr(backtest, key):
            setattr(backtest, key, value)
            
    db.commit()
    db.refresh(backtest)
    return backtest

def complete_backtest(db: Session,
                     backtest_id: str,
                     final_capital: float,
                     total_return: float,
                     total_trades: int,
                     metrics: Dict[str, float],
                     equity_curve: Optional[List[Dict]] = None,
                     execution_time: Optional[float] = None) -> Optional[models.Backtest]:
    """Mark backtest as completed with results"""
    return update_backtest(
        db,
        backtest_id,
        status="completed",
        completed_at=datetime.utcnow(),
        final_capital=final_capital,
        total_return=total_return,
        total_trades=total_trades,
        metrics=metrics,
        equity_curve=equity_curve,
        execution_time=execution_time,
        progress=100.0
    )

# Order CRUD operations

def create_order(db: Session,
                symbol: str,
                side: str,
                order_type: str,
                quantity: int,
                strategy_instance_id: Optional[int] = None,
                price: Optional[float] = None,
                trigger_price: Optional[float] = None,
                stop_loss: Optional[float] = None,
                take_profit: Optional[float] = None,
                metadata: Optional[Dict] = None) -> models.Order:
    """Create a new order"""
    db_order = models.Order(
        order_id=str(uuid.uuid4()),
        strategy_instance_id=strategy_instance_id,
        symbol=symbol,
        side=side,
        order_type=order_type,
        quantity=quantity,
        price=price,
        trigger_price=trigger_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        metadata=metadata,
        status="PENDING"
    )
    db.add(db_order)
    db.commit()
    db.refresh(db_order)
    return db_order

def get_order(db: Session, order_id: str) -> Optional[models.Order]:
    """Get order by ID"""
    return db.query(models.Order).filter(models.Order.order_id == order_id).first()

def get_orders(db: Session,
              skip: int = 0,
              limit: int = 100,
              symbol: Optional[str] = None,
              status: Optional[str] = None,
              strategy_instance_id: Optional[int] = None,
              start_date: Optional[datetime] = None,
              end_date: Optional[datetime] = None) -> List[models.Order]:
    """Get list of orders with filters"""
    query = db.query(models.Order)
    
    if symbol:
        query = query.filter(models.Order.symbol == symbol)
    if status:
        query = query.filter(models.Order.status == status)
    if strategy_instance_id:
        query = query.filter(models.Order.strategy_instance_id == strategy_instance_id)
    if start_date:
        query = query.filter(models.Order.placed_at >= start_date)
    if end_date:
        query = query.filter(models.Order.placed_at <= end_date)
        
    return query.order_by(desc(models.Order.placed_at)).offset(skip).limit(limit).all()

def update_order(db: Session,
                order_id: str,
                **kwargs) -> Optional[models.Order]:
    """Update order"""
    order = get_order(db, order_id)
    if not order:
        return None
        
    for key, value in kwargs.items():
        if hasattr(order, key):
            setattr(order, key, value)
            
    order.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(order)
    return order

def fill_order(db: Session,
              order_id: str,
              fill_price: float,
              filled_quantity: Optional[int] = None) -> Optional[models.Order]:
    """Mark order as filled"""
    order = get_order(db, order_id)
    if not order:
        return None
        
    return update_order(
        db,
        order_id,
        status="COMPLETE",
        filled_quantity=filled_quantity or order.quantity,
        avg_fill_price=fill_price,
        filled_at=datetime.utcnow()
    )

# Trade CRUD operations

def create_trade(db: Session,
                order_id: int,
                symbol: str,
                side: str,
                quantity: int,
                price: float,
                strategy_instance_id: Optional[int] = None,
                position_id: Optional[int] = None,
                pnl: float = 0.0,
                commission: float = 0.0,
                slippage: float = 0.0) -> models.Trade:
    """Create a new trade record"""
    
    pnl_percentage = 0.0
    if pnl != 0 and price > 0 and quantity > 0:
        # Calculate P&L percentage based on trade value
        trade_value = price * quantity
        pnl_percentage = (pnl / trade_value) * 100
    
    db_trade = models.Trade(
        trade_id=str(uuid.uuid4()),
        order_id=order_id,
        strategy_instance_id=strategy_instance_id,
        position_id=position_id,
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=price,
        pnl=pnl,
        pnl_percentage=pnl_percentage,
        commission=commission,
        slippage=slippage
    )
    db.add(db_trade)
    db.commit()
    db.refresh(db_trade)
    return db_trade

def get_trades(db: Session,
              skip: int = 0,
              limit: int = 100,
              symbol: Optional[str] = None,
              strategy_instance_id: Optional[int] = None,
              start_date: Optional[datetime] = None,
              end_date: Optional[datetime] = None) -> List[models.Trade]:
    """Get list of trades with filters"""
    query = db.query(models.Trade)
    
    if symbol:
        query = query.filter(models.Trade.symbol == symbol)
    if strategy_instance_id:
        query = query.filter(models.Trade.strategy_instance_id == strategy_instance_id)
    if start_date:
        query = query.filter(models.Trade.executed_at >= start_date)
    if end_date:
        query = query.filter(models.Trade.executed_at <= end_date)
        
    return query.order_by(desc(models.Trade.executed_at)).offset(skip).limit(limit).all()

# Position CRUD operations

def create_position(db: Session,
                   symbol: str,
                   side: str,
                   quantity: int,
                   avg_price: float,
                   strategy_instance_id: Optional[int] = None,
                   stop_loss: Optional[float] = None,
                   take_profit: Optional[float] = None) -> models.Position:
    """Create a new position"""
    db_position = models.Position(
        position_id=str(uuid.uuid4()),
        strategy_instance_id=strategy_instance_id,
        symbol=symbol,
        side=side,
        quantity=quantity,
        avg_price=avg_price,
        current_price=avg_price,
        market_value=quantity * avg_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        max_quantity=quantity,
        high_water_mark=avg_price,
        low_water_mark=avg_price
    )
    db.add(db_position)
    db.commit()
    db.refresh(db_position)
    return db_position

def get_position(db: Session, position_id: str) -> Optional[models.Position]:
    """Get position by ID"""
    return db.query(models.Position).filter(
        models.Position.position_id == position_id
    ).first()

def get_open_positions(db: Session,
                      symbol: Optional[str] = None,
                      strategy_instance_id: Optional[int] = None) -> List[models.Position]:
    """Get list of open positions"""
    query = db.query(models.Position).filter(models.Position.is_open == True)
    
    if symbol:
        query = query.filter(models.Position.symbol == symbol)
    if strategy_instance_id:
        query = query.filter(models.Position.strategy_instance_id == strategy_instance_id)
        
    return query.all()

def update_position(db: Session,
                   position_id: str,
                   **kwargs) -> Optional[models.Position]:
    """Update position"""
    position = get_position(db, position_id)
    if not position:
        return None
        
    for key, value in kwargs.items():
        if hasattr(position, key):
            setattr(position, key, value)
            
    position.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(position)
    return position

def close_position(db: Session,
                  position_id: str,
                  exit_price: float,
                  realized_pnl: float) -> Optional[models.Position]:
    """Close a position"""
    return update_position(
        db,
        position_id,
        is_open=False,
        closed_at=datetime.utcnow(),
        current_price=exit_price,
        realized_pnl=realized_pnl,
        unrealized_pnl=0.0
    )

# Performance CRUD operations

def create_performance_snapshot(db: Session,
                              date: datetime,
                              portfolio_value: float,
                              cash_balance: float,
                              positions_value: float,
                              strategy_instance_id: Optional[int] = None,
                              daily_pnl: float = 0.0,
                              total_pnl: float = 0.0,
                              open_positions: int = 0) -> models.Performance:
    """Create daily performance snapshot"""
    
    # Calculate returns
    daily_return = 0.0
    total_return = 0.0
    
    # Get previous snapshot to calculate returns
    if strategy_instance_id:
        prev_snapshot = db.query(models.Performance).filter(
            and_(
                models.Performance.strategy_instance_id == strategy_instance_id,
                models.Performance.date < date
            )
        ).order_by(desc(models.Performance.date)).first()
        
        if prev_snapshot:
            daily_return = ((portfolio_value - prev_snapshot.portfolio_value) / 
                          prev_snapshot.portfolio_value) * 100
    
    db_performance = models.Performance(
        date=date,
        strategy_instance_id=strategy_instance_id,
        portfolio_value=portfolio_value,
        cash_balance=cash_balance,
        positions_value=positions_value,
        daily_pnl=daily_pnl,
        daily_return=daily_return,
        total_pnl=total_pnl,
        total_return=total_return,
        open_positions=open_positions
    )
    
    db.add(db_performance)
    db.commit()
    db.refresh(db_performance)
    return db_performance

def get_performance_history(db: Session,
                          strategy_instance_id: Optional[int] = None,
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None) -> List[models.Performance]:
    """Get performance history"""
    query = db.query(models.Performance)
    
    if strategy_instance_id:
        query = query.filter(models.Performance.strategy_instance_id == strategy_instance_id)
    if start_date:
        query = query.filter(models.Performance.date >= start_date)
    if end_date:
        query = query.filter(models.Performance.date <= end_date)
        
    return query.order_by(models.Performance.date).all()

# Alert CRUD operations

def create_alert(db: Session,
                type: str,
                title: str,
                message: str,
                severity: str = "info",
                strategy_instance_id: Optional[int] = None,
                order_id: Optional[str] = None,
                position_id: Optional[str] = None,
                metadata: Optional[Dict] = None) -> models.Alert:
    """Create a new alert"""
    db_alert = models.Alert(
        alert_id=str(uuid.uuid4()),
        type=type,
        severity=severity,
        title=title,
        message=message,
        strategy_instance_id=strategy_instance_id,
        order_id=order_id,
        position_id=position_id,
        metadata=metadata
    )
    db.add(db_alert)
    db.commit()
    db.refresh(db_alert)
    return db_alert

def get_alerts(db: Session,
              skip: int = 0,
              limit: int = 100,
              is_read: Optional[bool] = None,
              severity: Optional[str] = None,
              type: Optional[str] = None) -> List[models.Alert]:
    """Get list of alerts"""
    query = db.query(models.Alert)
    
    if is_read is not None:
        query = query.filter(models.Alert.is_read == is_read)
    if severity:
        query = query.filter(models.Alert.severity == severity)
    if type:
        query = query.filter(models.Alert.type == type)
        
    return query.order_by(desc(models.Alert.created_at)).offset(skip).limit(limit).all()

def mark_alert_read(db: Session, alert_id: str) -> Optional[models.Alert]:
    """Mark alert as read"""
    alert = db.query(models.Alert).filter(models.Alert.alert_id == alert_id).first()
    if not alert:
        return None
        
    alert.is_read = True
    alert.read_at = datetime.utcnow()
    db.commit()
    db.refresh(alert)
    return alert

# Audit Log CRUD operations

def create_audit_log(db: Session,
                    action: str,
                    entity_type: str,
                    entity_id: Optional[str] = None,
                    user_id: Optional[str] = None,
                    old_values: Optional[Dict] = None,
                    new_values: Optional[Dict] = None,
                    metadata: Optional[Dict] = None) -> models.AuditLog:
    """Create audit log entry"""
    db_audit = models.AuditLog(
        action=action,
        entity_type=entity_type,
        entity_id=entity_id,
        user_id=user_id,
        old_values=old_values,
        new_values=new_values,
        metadata=metadata
    )
    db.add(db_audit)
    db.commit()
    db.refresh(db_audit)
    return db_audit

# Statistics and Analytics

def get_strategy_statistics(db: Session, strategy_instance_id: int) -> Dict[str, Any]:
    """Get comprehensive statistics for a strategy instance"""
    
    # Get instance
    instance = db.query(models.StrategyInstance).filter(
        models.StrategyInstance.id == strategy_instance_id
    ).first()
    
    if not instance:
        return {}
        
    # Get trades
    trades = db.query(models.Trade).filter(
        models.Trade.strategy_instance_id == strategy_instance_id
    ).all()
    
    # Get positions
    open_positions = db.query(models.Position).filter(
        and_(
            models.Position.strategy_instance_id == strategy_instance_id,
            models.Position.is_open == True
        )
    ).all()
    
    # Calculate statistics
    total_trades = len(trades)
    winning_trades = len([t for t in trades if t.pnl > 0])
    losing_trades = len([t for t in trades if t.pnl < 0])
    
    total_pnl = sum(t.pnl for t in trades)
    total_commission = sum(t.commission for t in trades)
    
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    # Get latest performance
    latest_performance = db.query(models.Performance).filter(
        models.Performance.strategy_instance_id == strategy_instance_id
    ).order_by(desc(models.Performance.date)).first()
    
    return {
        "instance_id": instance.instance_id,
        "name": instance.name,
        "status": instance.status,
        "mode": instance.mode,
        "started_at": instance.started_at,
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "total_commission": total_commission,
        "net_pnl": total_pnl - total_commission,
        "open_positions": len(open_positions),
        "portfolio_value": latest_performance.portfolio_value if latest_performance else 0,
        "sharpe_ratio": latest_performance.sharpe_ratio if latest_performance else 0,
        "max_drawdown": latest_performance.max_drawdown if latest_performance else 0
    }