import uuid
from typing import Dict, Optional, List, Tuple
from datetime import datetime
import logging
from ..models.trading import Trade, TradeSignal, TradeAction, SpreadData
from ..utils.json_audit import JsonAuditLogger
from .spread_calculator import SpreadCalculator

logger = logging.getLogger(__name__)

class PairTradingStrategy:
    def __init__(self, 
                 spread_calculator: SpreadCalculator,
                 audit_logger: JsonAuditLogger,
                 entry_z_score: float = 2.0,
                 exit_z_score: float = 0.5,
                 stop_loss_z_score: float = 3.0):
        
        self.spread_calculator = spread_calculator
        self.audit_logger = audit_logger
        self.entry_z_score = entry_z_score
        self.exit_z_score = exit_z_score
        self.stop_loss_z_score = stop_loss_z_score
        
        self.active_trades: Dict[str, Trade] = {}
        self.is_active = False
        self.current_pair: Optional[Tuple[str, str]] = None
        self.total_pnl = 0.0
        
    def process_tick(self, spread_data: SpreadData) -> Optional[TradeSignal]:
        """Process new spread data and generate trading signals"""
        
        if not self.is_active or spread_data.z_score is None:
            return None
        
        signal = self._generate_signal(spread_data)
        
        if signal and signal.action != TradeAction.NONE:
            self._execute_signal(signal, spread_data)
            
        # Log spread calculation
        self.audit_logger.log_spread_calculation(spread_data)
        
        return signal
    
    def _generate_signal(self, spread_data: SpreadData) -> Optional[TradeSignal]:
        """Generate trading signal based on spread data"""
        
        z_score = spread_data.z_score
        pair_key = f"{spread_data.instrument1}_{spread_data.instrument2}"
        
        # Check if we have an open trade
        if pair_key in self.active_trades:
            trade = self.active_trades[pair_key]
            
            # Exit conditions
            if trade.trade_type == "LONG_SPREAD" and z_score <= self.exit_z_score:
                return TradeSignal(
                    timestamp=datetime.now(),
                    action=TradeAction.CLOSE,
                    instrument1=spread_data.instrument1,
                    instrument2=spread_data.instrument2,
                    spread=spread_data.spread,
                    entry_reason="Z-score returned to mean"
                )
            elif trade.trade_type == "SHORT_SPREAD" and z_score >= -self.exit_z_score:
                return TradeSignal(
                    timestamp=datetime.now(),
                    action=TradeAction.CLOSE,
                    instrument1=spread_data.instrument1,
                    instrument2=spread_data.instrument2,
                    spread=spread_data.spread,
                    entry_reason="Z-score returned to mean"
                )
            
            # Stop loss
            elif abs(z_score) >= self.stop_loss_z_score:
                return TradeSignal(
                    timestamp=datetime.now(),
                    action=TradeAction.CLOSE,
                    instrument1=spread_data.instrument1,
                    instrument2=spread_data.instrument2,
                    spread=spread_data.spread,
                    entry_reason="Stop loss triggered"
                )
        
        else:
            # Entry conditions
            if z_score >= self.entry_z_score:
                return TradeSignal(
                    timestamp=datetime.now(),
                    action=TradeAction.SELL,  # Sell the spread (short spread trade)
                    instrument1=spread_data.instrument1,
                    instrument2=spread_data.instrument2,
                    spread=spread_data.spread,
                    entry_reason=f"Z-score above threshold: {z_score:.2f}"
                )
            elif z_score <= -self.entry_z_score:
                return TradeSignal(
                    timestamp=datetime.now(),
                    action=TradeAction.BUY,  # Buy the spread (long spread trade)
                    instrument1=spread_data.instrument1,
                    instrument2=spread_data.instrument2,
                    spread=spread_data.spread,
                    entry_reason=f"Z-score below threshold: {z_score:.2f}"
                )
        
        return TradeSignal(
            timestamp=datetime.now(),
            action=TradeAction.NONE,
            instrument1=spread_data.instrument1,
            instrument2=spread_data.instrument2,
            spread=spread_data.spread,
            entry_reason="No signal"
        )
    
    def _execute_signal(self, signal: TradeSignal, spread_data: SpreadData):
        """Execute trading signal"""
        pair_key = f"{signal.instrument1}_{signal.instrument2}"
        
        if signal.action in [TradeAction.BUY, TradeAction.SELL]:
            # Open new trade
            trade = Trade(
                trade_id=str(uuid.uuid4()),
                entry_time=datetime.now(),
                instrument1=signal.instrument1,
                instrument2=signal.instrument2,
                entry_spread=signal.spread,
                trade_type="LONG_SPREAD" if signal.action == TradeAction.BUY else "SHORT_SPREAD"
            )
            
            self.active_trades[pair_key] = trade
            self.audit_logger.log_trade_entry(trade, signal)
            logger.info(f"Opened {trade.trade_type} trade: {trade.trade_id}")
            
        elif signal.action == TradeAction.CLOSE and pair_key in self.active_trades:
            # Close existing trade
            trade = self.active_trades[pair_key]
            trade.exit_time = datetime.now()
            trade.exit_spread = signal.spread
            trade.status = "CLOSED"
            
            # Calculate PnL
            if trade.trade_type == "LONG_SPREAD":
                trade.pnl = (trade.exit_spread - trade.entry_spread) * 75  # lot size
            else:  # SHORT_SPREAD
                trade.pnl = (trade.entry_spread - trade.exit_spread) * 75
            
            self.total_pnl += trade.pnl
            
            self.audit_logger.log_trade_exit(trade, signal)
            logger.info(f"Closed trade {trade.trade_id} with PnL: {trade.pnl}")
            
            del self.active_trades[pair_key]
    
    def start_strategy(self, instrument1: str, instrument2: str):
        """Start the trading strategy"""
        self.is_active = True
        self.current_pair = (instrument1, instrument2)
        logger.info(f"Strategy started for pair: {instrument1} - {instrument2}")
    
    def stop_strategy(self):
        """Stop the trading strategy"""
        self.is_active = False
        
        # Close all open trades
        for pair_key, trade in list(self.active_trades.items()):
            trade.exit_time = datetime.now()
            trade.status = "CLOSED"
            trade.pnl = 0  # Force close with no PnL
            self.audit_logger.log_trade_exit(trade, TradeSignal(
                timestamp=datetime.now(),
                action=TradeAction.CLOSE,
                instrument1=trade.instrument1,
                instrument2=trade.instrument2,
                spread=0,
                entry_reason="Strategy stopped"
            ))
        
        self.active_trades.clear()
        logger.info("Strategy stopped")
    
    def get_status(self) -> Dict:
        """Get current strategy status"""
        return {
            "is_active": self.is_active,
            "current_pair": self.current_pair,
            "open_trades": list(self.active_trades.values()),
            "total_pnl": self.total_pnl,
            "trades_count": len(self.active_trades)
        }