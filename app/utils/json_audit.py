import json
import aiofiles
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import logging
from ..models.trading import Trade, TradeSignal, SpreadData

logger = logging.getLogger(__name__)

class JsonAuditLogger:
    def __init__(self, log_file_path: str = "logs/trades_audit.json"):
        self.log_file_path = Path(log_file_path)
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize log file if it doesn't exist
        if not self.log_file_path.exists():
            self._initialize_log_file()
    
    def _initialize_log_file(self):
        """Initialize the JSON log file with empty structure"""
        initial_structure = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "version": "1.0"
            },
            "trades": [],
            "spread_calculations": [],
            "signals": []
        }
        
        with open(self.log_file_path, 'w') as f:
            json.dump(initial_structure, f, indent=2)
    
    async def _append_to_log(self, section: str, data: Dict[Any, Any]):
        """Append data to specific section of the log file"""
        try:
            async with aiofiles.open(self.log_file_path, 'r') as f:
                content = await f.read()
                log_data = json.loads(content)
            
            log_data[section].append(data)
            
            async with aiofiles.open(self.log_file_path, 'w') as f:
                await f.write(json.dumps(log_data, indent=2, default=str))
                
        except Exception as e:
            logger.error(f"Failed to write to audit log: {e}")
    
    def log_trade_entry(self, trade: Trade, signal: TradeSignal):
        """Log trade entry"""
        entry_data = {
            "event_type": "TRADE_ENTRY",
            "timestamp": datetime.now().isoformat(),
            "trade_id": trade.trade_id,
            "instrument1": trade.instrument1,
            "instrument2": trade.instrument2,
            "entry_spread": trade.entry_spread,
            "trade_type": trade.trade_type,
            "signal_reason": signal.entry_reason,
            "z_score": signal.z_score if hasattr(signal, 'z_score') else None
        }
        
        # Sync write for simplicity (can be made async)
        self._sync_append_to_log("trades", entry_data)
    
    def log_trade_exit(self, trade: Trade, signal: TradeSignal):
        """Log trade exit"""
        exit_data = {
            "event_type": "TRADE_EXIT",
            "timestamp": datetime.now().isoformat(),
            "trade_id": trade.trade_id,
            "instrument1": trade.instrument1,
            "instrument2": trade.instrument2,
            "entry_spread": trade.entry_spread,
            "exit_spread": trade.exit_spread,
            "pnl": trade.pnl,
            "trade_duration_minutes": (trade.exit_time - trade.entry_time).total_seconds() / 60 if trade.exit_time else 0,
            "exit_reason": signal.entry_reason
        }
        
        self._sync_append_to_log("trades", exit_data)
    
    def log_spread_calculation(self, spread_data: SpreadData):
        """Log spread calculation"""
        calc_data = {
            "timestamp": spread_data.timestamp.isoformat(),
            "instrument1": spread_data.instrument1,
            "instrument2": spread_data.instrument2,
            "price1": spread_data.price1,
            "price2": spread_data.price2,
            "spread": spread_data.spread,
            "spread_percentage": spread_data.spread_percentage,
            "ma_20": spread_data.ma_20,
            "ma_50": spread_data.ma_50,
            "z_score": spread_data.z_score
        }
        
        # Log only every 10th calculation to avoid huge files
        import random
        if random.random() < 0.1:  # 10% sampling
            self._sync_append_to_log("spread_calculations", calc_data)
    
    def _sync_append_to_log(self, section: str, data: Dict[Any, Any]):
        """Synchronous version of append_to_log"""
        try:
            with open(self.log_file_path, 'r') as f:
                log_data = json.load(f)
            
            log_data[section].append(data)
            
            with open(self.log_file_path, 'w') as f:
                json.dump(log_data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to write to audit log: {e}")