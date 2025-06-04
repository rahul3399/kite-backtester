# app/database/models.py
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, JSON, ForeignKey, Index, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()

class Strategy(Base):
    """Strategy registration and configuration"""
    __tablename__ = "strategies"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    class_name = Column(String(100), nullable=False)
    module_path = Column(String(255), nullable=False)
    description = Column(Text)
    parameters_schema = Column(JSON)  # JSON schema for parameters
    required_indicators = Column(JSON)  # List of required indicators
    version = Column(String(20), default="1.0.0")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    strategy_instances = relationship("StrategyInstance", back_populates="strategy", cascade="all, delete-orphan")
    backtests = relationship("Backtest", back_populates="strategy", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_strategy_active', 'is_active'),
    )

class StrategyInstance(Base):
    """Running strategy instances for paper/live trading"""
    __tablename__ = "strategy_instances"
    
    id = Column(Integer, primary_key=True, index=True)
    instance_id = Column(String(36), unique=True, nullable=False, index=True)  # UUID
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=False)
    name = Column(String(100), nullable=False)
    symbols = Column(JSON, nullable=False)  # List of symbols
    parameters = Column(JSON, nullable=False)  # Strategy parameters
    mode = Column(String(20), nullable=False, default="paper")  # paper/live
    status = Column(String(20), nullable=False, default="running")  # running/stopped/error
    
    # Capital and risk management
    capital_allocation = Column(Float)
    risk_per_trade = Column(Float, default=0.02)
    max_positions = Column(Integer, default=5)
    
    # Performance tracking
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    total_pnl = Column(Float, default=0.0)
    
    # Timestamps
    started_at = Column(DateTime, nullable=False, default=func.now())
    stopped_at = Column(DateTime)
    last_signal_at = Column(DateTime)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    strategy = relationship("Strategy", back_populates="strategy_instances")
    orders = relationship("Order", back_populates="strategy_instance", cascade="all, delete-orphan")
    positions = relationship("Position", back_populates="strategy_instance", cascade="all, delete-orphan")
    trades = relationship("Trade", back_populates="strategy_instance", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_instance_status', 'status'),
        Index('idx_instance_mode', 'mode'),
        Index('idx_instance_started', 'started_at'),
    )

class Backtest(Base):
    """Backtest runs and results"""
    __tablename__ = "backtests"
    
    id = Column(Integer, primary_key=True, index=True)
    backtest_id = Column(String(36), unique=True, nullable=False, index=True)  # UUID
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=False)
    
    # Configuration
    symbols = Column(JSON, nullable=False)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    initial_capital = Column(Float, nullable=False)
    commission = Column(Float, default=0.0002)
    slippage = Column(Float, default=0.0001)
    parameters = Column(JSON, nullable=False)
    
    # Status
    status = Column(String(20), nullable=False, default="pending")  # pending/running/completed/failed
    progress = Column(Float, default=0.0)
    error_message = Column(Text)
    
    # Results
    final_capital = Column(Float)
    total_return = Column(Float)
    total_trades = Column(Integer)
    metrics = Column(JSON)  # All performance metrics
    equity_curve = Column(JSON)  # Compressed equity curve data
    
    # Execution
    execution_time = Column(Float)  # seconds
    bars_processed = Column(Integer)
    
    # Timestamps
    created_at = Column(DateTime, server_default=func.now())
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # Relationships
    strategy = relationship("Strategy", back_populates="backtests")
    backtest_trades = relationship("BacktestTrade", back_populates="backtest", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_backtest_status', 'status'),
        Index('idx_backtest_created', 'created_at'),
    )

class Order(Base):
    """Order records for live and paper trading"""
    __tablename__ = "orders"
    
    id = Column(Integer, primary_key=True, index=True)
    order_id = Column(String(36), unique=True, nullable=False, index=True)  # Internal UUID
    kite_order_id = Column(String(50), index=True)  # External order ID from Kite
    strategy_instance_id = Column(Integer, ForeignKey("strategy_instances.id"))
    
    # Order details
    symbol = Column(String(50), nullable=False, index=True)
    exchange = Column(String(10), default="NSE")
    side = Column(String(10), nullable=False)  # BUY/SELL
    order_type = Column(String(20), nullable=False)  # MARKET/LIMIT/STOP_LOSS
    quantity = Column(Integer, nullable=False)
    price = Column(Float)
    trigger_price = Column(Float)
    
    # Status
    status = Column(String(20), nullable=False, default="PENDING")
    filled_quantity = Column(Integer, default=0)
    avg_fill_price = Column(Float)
    
    # Risk management
    stop_loss = Column(Float)
    take_profit = Column(Float)
    
    # Metadata
    tag = Column(String(50))
    meta_info = Column(JSON)
    
    # Timestamps
    placed_at = Column(DateTime, nullable=False, default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    filled_at = Column(DateTime)
    cancelled_at = Column(DateTime)
    
    # Relationships
    strategy_instance = relationship("StrategyInstance", back_populates="orders")
    trades = relationship("Trade", back_populates="order", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_order_status', 'status'),
        Index('idx_order_symbol', 'symbol'),
        Index('idx_order_placed', 'placed_at'),
    )

class Trade(Base):
    """Executed trades (fills)"""
    __tablename__ = "trades"
    
    id = Column(Integer, primary_key=True, index=True)
    trade_id = Column(String(36), unique=True, nullable=False, index=True)
    order_id = Column(Integer, ForeignKey("orders.id"), nullable=False)
    strategy_instance_id = Column(Integer, ForeignKey("strategy_instances.id"))
    position_id = Column(Integer, ForeignKey("positions.id"))
    
    # Trade details
    symbol = Column(String(50), nullable=False, index=True)
    side = Column(String(10), nullable=False)
    quantity = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)
    
    # P&L
    pnl = Column(Float, default=0.0)
    pnl_percentage = Column(Float, default=0.0)
    commission = Column(Float, default=0.0)
    slippage = Column(Float, default=0.0)
    
    # Timestamps
    executed_at = Column(DateTime, nullable=False, default=func.now())
    
    # Relationships
    order = relationship("Order", back_populates="trades")
    strategy_instance = relationship("StrategyInstance", back_populates="trades")
    position = relationship("Position", back_populates="trades")
    
    __table_args__ = (
        Index('idx_trade_symbol', 'symbol'),
        Index('idx_trade_executed', 'executed_at'),
    )

class Position(Base):
    """Current and historical positions"""
    __tablename__ = "positions"
    
    id = Column(Integer, primary_key=True, index=True)
    position_id = Column(String(36), unique=True, nullable=False, index=True)
    strategy_instance_id = Column(Integer, ForeignKey("strategy_instances.id"))
    
    # Position details
    symbol = Column(String(50), nullable=False, index=True)
    side = Column(String(10), nullable=False)  # LONG/SHORT
    quantity = Column(Integer, nullable=False)
    avg_price = Column(Float, nullable=False)
    
    # Current state
    is_open = Column(Boolean, default=True, index=True)
    current_price = Column(Float)
    market_value = Column(Float)
    
    # P&L
    unrealized_pnl = Column(Float, default=0.0)
    realized_pnl = Column(Float, default=0.0)
    total_commission = Column(Float, default=0.0)
    
    # Risk management
    stop_loss = Column(Float)
    take_profit = Column(Float)
    max_quantity = Column(Integer)  # Maximum position size reached
    
    # Performance
    high_water_mark = Column(Float)
    low_water_mark = Column(Float)
    max_unrealized_profit = Column(Float, default=0.0)
    max_unrealized_loss = Column(Float, default=0.0)
    
    # Timestamps
    opened_at = Column(DateTime, nullable=False, default=func.now())
    closed_at = Column(DateTime)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    strategy_instance = relationship("StrategyInstance", back_populates="positions")
    trades = relationship("Trade", back_populates="position")
    
    __table_args__ = (
        Index('idx_position_open', 'is_open'),
        Index('idx_position_symbol', 'symbol'),
        Index('idx_position_opened', 'opened_at'),
    )

class MarketData(Base):
    """Historical market data cache"""
    __tablename__ = "market_data"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(50), nullable=False)
    instrument_token = Column(Integer, nullable=False)
    interval = Column(String(20), nullable=False)  # minute/5minute/day etc
    
    # OHLCV data
    timestamp = Column(DateTime, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False)
    oi = Column(Integer)  # Open interest for F&O
    
    # Timestamps
    created_at = Column(DateTime, server_default=func.now())
    
    __table_args__ = (
        UniqueConstraint('symbol', 'interval', 'timestamp', name='uq_market_data'),
        Index('idx_market_data_symbol_interval_time', 'symbol', 'interval', 'timestamp'),
    )

class Performance(Base):
    """Daily performance snapshots"""
    __tablename__ = "performance"
    
    id = Column(Integer, primary_key=True, index=True)
    date = Column(DateTime, nullable=False)
    strategy_instance_id = Column(Integer, ForeignKey("strategy_instances.id"))
    
    # Portfolio metrics
    portfolio_value = Column(Float, nullable=False)
    cash_balance = Column(Float, nullable=False)
    positions_value = Column(Float, nullable=False)
    
    # Daily metrics
    daily_pnl = Column(Float, default=0.0)
    daily_return = Column(Float, default=0.0)
    daily_trades = Column(Integer, default=0)
    
    # Cumulative metrics
    total_pnl = Column(Float, default=0.0)
    total_return = Column(Float, default=0.0)
    win_rate = Column(Float)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    
    # Position counts
    open_positions = Column(Integer, default=0)
    total_positions = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime, server_default=func.now())
    
    __table_args__ = (
        UniqueConstraint('date', 'strategy_instance_id', name='uq_performance_date'),
        Index('idx_performance_date', 'date'),
        Index('idx_performance_instance', 'strategy_instance_id'),
    )

class Alert(Base):
    """System alerts and notifications"""
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    alert_id = Column(String(36), unique=True, nullable=False, index=True)
    
    # Alert details
    type = Column(String(50), nullable=False)  # order_filled/stop_loss_hit/error etc
    severity = Column(String(20), nullable=False, default="info")  # info/warning/error/critical
    title = Column(String(200), nullable=False)
    message = Column(Text, nullable=False)
    
    # Related entities
    strategy_instance_id = Column(Integer, ForeignKey("strategy_instances.id"))
    order_id = Column(String(36))
    position_id = Column(String(36))
    
    # Status
    is_read = Column(Boolean, default=False, index=True)
    is_resolved = Column(Boolean, default=False)
    
    # Metadata
    meta_info = Column(JSON)
    
    # Timestamps
    created_at = Column(DateTime, server_default=func.now())
    read_at = Column(DateTime)
    resolved_at = Column(DateTime)
    
    __table_args__ = (
        Index('idx_alert_type', 'type'),
        Index('idx_alert_severity', 'severity'),
        Index('idx_alert_read', 'is_read'),
        Index('idx_alert_created', 'created_at'),
    )

class BacktestTrade(Base):
    """Trade records from backtests"""
    __tablename__ = "backtest_trades"
    
    id = Column(Integer, primary_key=True, index=True)
    backtest_id = Column(Integer, ForeignKey("backtests.id"), nullable=False)
    
    # Trade details
    symbol = Column(String(50), nullable=False)
    side = Column(String(10), nullable=False)
    quantity = Column(Integer, nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float)
    entry_time = Column(DateTime, nullable=False)
    exit_time = Column(DateTime)
    
    # P&L
    pnl = Column(Float)
    pnl_percentage = Column(Float)
    commission = Column(Float, default=0.0)
    slippage = Column(Float, default=0.0)
    
    # Metadata
    meta_info = Column(JSON)
    
    # Relationships
    backtest = relationship("Backtest", back_populates="backtest_trades")
    
    __table_args__ = (
        Index('idx_backtest_trade_symbol', 'symbol'),
        Index('idx_backtest_trade_time', 'entry_time'),
    )

class AuditLog(Base):
    """Audit trail for all system actions"""
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Action details
    action = Column(String(100), nullable=False, index=True)
    entity_type = Column(String(50), nullable=False)  # order/trade/position etc
    entity_id = Column(String(36))
    
    # User info (if applicable)
    user_id = Column(String(50))
    ip_address = Column(String(45))
    
    # Change details
    old_values = Column(JSON)
    new_values = Column(JSON)
    
    # Metadata
    meta_info = Column(JSON)
    
    # Timestamp
    created_at = Column(DateTime, server_default=func.now(), index=True)
    
    __table_args__ = (
        Index('idx_audit_action', 'action'),
        Index('idx_audit_entity', 'entity_type', 'entity_id'),
    )