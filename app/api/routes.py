from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List, Optional  # Add Optional here

from ..models.trading import StrategyStatus
from ..services.trading_strategy import PairTradingStrategy
from ..services.websocket_manager import WebSocketManager
from ..services.kite_client import KiteClient
from ..services.instrument_lookup import InstrumentLookup
from .dependencies import get_strategy, get_ws_manager, get_kite_client
from ..services.backtesting_service import BacktestingService, BacktestConfig
from datetime import datetime, timedelta

router = APIRouter()


def get_instrument_lookup(kite_client: KiteClient = Depends(get_kite_client)) -> InstrumentLookup:
    """Get instrument lookup service"""
    return InstrumentLookup(kite_client)

@router.post("/strategy/start/v2")
async def start_strategy_v2(
    instrument1: str,  # Now accepts symbol like "NIFTY24JAN25000CE"
    instrument2: str,  # Now accepts symbol like "NIFTY24JAN25000PE"
    strategy: PairTradingStrategy = Depends(get_strategy),
    ws_manager: WebSocketManager = Depends(get_ws_manager),
    instrument_lookup: InstrumentLookup = Depends(get_instrument_lookup)
):
    """Start the pair trading strategy using instrument symbols"""
    try:
        # Look up instrument tokens
        token1 = instrument_lookup.get_instrument_token(instrument1)
        token2 = instrument_lookup.get_instrument_token(instrument2)
        
        if not token1:
            raise HTTPException(status_code=404, detail=f"Instrument {instrument1} not found")
        if not token2:
            raise HTTPException(status_code=404, detail=f"Instrument {instrument2} not found")
        
        # Subscribe to instruments
        ws_manager.subscribe([token1, token2])
        
        # Start strategy
        strategy.start_strategy(instrument1, instrument2)
        
        return {
            "status": "success",
            "message": f"Strategy started for {instrument1} - {instrument2}",
            "instruments": {
                instrument1: {"token": token1},
                instrument2: {"token": token2}
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/strategy/start")
async def start_strategy(
    instrument1: str,
    instrument2: str,
    instrument1_token: int,
    instrument2_token: int,
    strategy: PairTradingStrategy = Depends(get_strategy),
    ws_manager: WebSocketManager = Depends(get_ws_manager)
):
    """Start the pair trading strategy"""
    try:
        # Subscribe to instruments
        ws_manager.subscribe([instrument1_token, instrument2_token])
        
        # Start strategy
        strategy.start_strategy(instrument1, instrument2)
        
        return {
            "status": "success",
            "message": f"Strategy started for {instrument1} - {instrument2}",
            "tokens": [instrument1_token, instrument2_token]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/strategy/stop")
async def stop_strategy(
    strategy: PairTradingStrategy = Depends(get_strategy)
):
    """Stop the pair trading strategy"""
    try:
        strategy.stop_strategy()
        return {
            "status": "success",
            "message": "Strategy stopped",
            "final_pnl": strategy.total_pnl
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/strategy/status", response_model=Dict)
async def get_strategy_status(
    strategy: PairTradingStrategy = Depends(get_strategy)
):
    """Get current strategy status"""
    return strategy.get_status()

@router.post("/strategy/change-pair")
async def change_trading_pair(
    instrument1: str,
    instrument2: str,
    instrument1_token: int,
    instrument2_token: int,
    strategy: PairTradingStrategy = Depends(get_strategy),
    ws_manager: WebSocketManager = Depends(get_ws_manager)
):
    """Change the trading pair"""
    try:
        # Stop current strategy
        strategy.stop_strategy()
        
        # Unsubscribe from old instruments (if any)
        # Subscribe to new instruments
        ws_manager.subscribe([instrument1_token, instrument2_token])
        
        # Start with new pair
        strategy.start_strategy(instrument1, instrument2)
        
        return {
            "status": "success",
            "message": f"Switched to new pair: {instrument1} - {instrument2}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/spread/current")
async def get_current_spread(
    strategy: PairTradingStrategy = Depends(get_strategy)
):
    """Get current spread for active pair"""
    if not strategy.is_active or not strategy.current_pair:
        raise HTTPException(status_code=400, detail="No active trading pair")
    
    instrument1, instrument2 = strategy.current_pair
    stats = strategy.spread_calculator.get_spread_statistics(instrument1, instrument2)
    
    return {
        "pair": f"{instrument1} - {instrument2}",
        "statistics": stats
    }

@router.get("/spread/history")
async def get_spread_history(
    instrument1: str,
    instrument2: str,
    limit: int = 100,
    strategy: PairTradingStrategy = Depends(get_strategy)
):
    """Get historical spread data"""
    pair_key = f"{instrument1}_{instrument2}"
    
    if pair_key not in strategy.spread_calculator.spread_history:
        raise HTTPException(status_code=404, detail="No history for this pair")
    
    history = list(strategy.spread_calculator.spread_history[pair_key])[-limit:]
    
    return {
        "pair": f"{instrument1} - {instrument2}",
        "data_points": len(history),
        "spreads": history
    }


@router.get("/instruments/search")
async def search_instruments(
    query: str,
    exchange: str = "NFO",
    instrument_lookup: InstrumentLookup = Depends(get_instrument_lookup)
):
    """Search for instruments by symbol"""
    try:
        results = instrument_lookup.search_instruments(query, exchange)
        return {
            "query": query,
            "exchange": exchange,
            "count": len(results),
            "results": results[:10]  # Limit display
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/instruments/nifty-options")
async def get_nifty_options(
    strikes_around_atm: int = 5,
    instrument_lookup: InstrumentLookup = Depends(get_instrument_lookup),
    kite_client: KiteClient = Depends(get_kite_client)
):
    """Get current Nifty options with ATM strikes"""
    try:
        # Get Nifty spot price
        ltp_data = kite_client.get_ltp(["NSE:NIFTY 50"])
        spot_price = ltp_data["NSE:NIFTY 50"]["last_price"]
        
        # Get ATM strikes
        strikes = instrument_lookup.get_atm_strikes(spot_price, strikes_around_atm)
        
        # Get option pairs
        pairs = instrument_lookup.get_option_pairs("NIFTY", strikes)
        
        return {
            "spot_price": spot_price,
            "atm_strike": round(spot_price / 50) * 50,
            "strikes": strikes,
            "option_pairs": pairs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/strategy/start/auto")
async def start_strategy_auto(
    strike: int,
    option_types: str = "CE-PE",  # Can be "CE-PE", "CE-CE", "PE-PE"
    expiry_offset: int = 0,  # 0 for current expiry, 1 for next, etc.
    strategy: PairTradingStrategy = Depends(get_strategy),
    ws_manager: WebSocketManager = Depends(get_ws_manager),
    instrument_lookup: InstrumentLookup = Depends(get_instrument_lookup)
):
    """Start strategy with automatic instrument selection"""
    try:
        types = option_types.split("-")
        if len(types) != 2:
            raise HTTPException(status_code=400, detail="option_types must be like 'CE-PE'")
        
        # Get tokens
        token1 = instrument_lookup.get_nifty_option_token(strike, types[0])
        token2 = instrument_lookup.get_nifty_option_token(strike, types[1])
        
        if not token1 or not token2:
            raise HTTPException(status_code=404, detail=f"Options not found for strike {strike}")
        
        # Get instrument details
        inst1 = instrument_lookup._instruments_cache['NFO']['by_token'][token1]
        inst2 = instrument_lookup._instruments_cache['NFO']['by_token'][token2]
        
        # Subscribe and start
        ws_manager.subscribe([token1, token2])
        strategy.start_strategy(inst1['tradingsymbol'], inst2['tradingsymbol'])
        
        return {
            "status": "success",
            "message": f"Strategy started for strike {strike}",
            "instruments": {
                "instrument1": {
                    "symbol": inst1['tradingsymbol'],
                    "token": token1,
                    "type": types[0]
                },
                "instrument2": {
                    "symbol": inst2['tradingsymbol'], 
                    "token": token2,
                    "type": types[1]
                }
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/instruments/lookup/{symbol}")
async def lookup_instrument(
    symbol: str,
    exchange: str = "NFO",
    instrument_lookup: InstrumentLookup = Depends(get_instrument_lookup)
):
    """Look up a single instrument by symbol"""
    instrument = instrument_lookup.get_instrument_by_symbol(symbol.upper(), exchange)
    
    if not instrument:
        # Try parsing as option symbol
        parsed = instrument_lookup.parse_option_symbol(symbol)
        if parsed:
            raise HTTPException(
                status_code=404, 
                detail=f"Instrument not found. Parsed as: {parsed}"
            )
        raise HTTPException(status_code=404, detail=f"Instrument {symbol} not found")
    
    return {
        "symbol": instrument['tradingsymbol'],
        "token": instrument['instrument_token'],
        "exchange": instrument['exchange'],
        "instrument_type": instrument.get('instrument_type'),
        "strike": instrument.get('strike'),
        "expiry": instrument.get('expiry')
    }

def get_backtesting_service(kite_client: KiteClient = Depends(get_kite_client)) -> BacktestingService:
    """Get backtesting service instance"""
    return BacktestingService(kite_client)

@router.post("/backtest/run")
async def run_backtest(
    instrument1: str,
    instrument2: str,
    days_back: int = 30,
    entry_z_score: float = 2.0,
    exit_z_score: float = 0.5,
    stop_loss_z_score: float = 3.0,
    lookback_period: int = 20,
    backtesting_service: BacktestingService = Depends(get_backtesting_service),
    instrument_lookup: InstrumentLookup = Depends(get_instrument_lookup)
):
    """Run backtest for a pair of instruments"""
    try:
        # Get instrument tokens
        token1 = instrument_lookup.get_instrument_token(instrument1)
        token2 = instrument_lookup.get_instrument_token(instrument2)
        
        if not token1 or not token2:
            raise HTTPException(
                status_code=404, 
                detail=f"Instruments not found: {instrument1} or {instrument2}"
            )
        
        # Set date range
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days_back)
        
        # Configure backtest
        config = BacktestConfig(
            entry_z_score=entry_z_score,
            exit_z_score=exit_z_score,
            stop_loss_z_score=stop_loss_z_score,
            lookback_period=lookback_period
        )
        
        # Run backtest
        results = await backtesting_service.run_backtest(
            instrument1=instrument1,
            instrument2=instrument2,
            token1=token1,
            token2=token2,
            from_date=from_date,
            to_date=to_date,
            config=config
        )
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/backtest/quick")
async def quick_backtest(
    strike: int,
    option_type: str = "CE-PE",
    days_back: int = 30,
    backtesting_service: BacktestingService = Depends(get_backtesting_service),
    instrument_lookup: InstrumentLookup = Depends(get_instrument_lookup)
):
    """Quick backtest for NIFTY options at given strike"""
    try:
        # Parse option types
        types = option_type.split("-")
        if len(types) != 2:
            raise HTTPException(status_code=400, detail="Invalid option_type format")
        
        # Get tokens
        token1 = instrument_lookup.get_nifty_option_token(strike, types[0])
        token2 = instrument_lookup.get_nifty_option_token(strike, types[1])
        
        if not token1 or not token2:
            raise HTTPException(status_code=404, detail=f"Options not found for strike {strike}")
        
        # Get instrument names
        inst1 = instrument_lookup._instruments_cache['NFO']['by_token'][token1]
        inst2 = instrument_lookup._instruments_cache['NFO']['by_token'][token2]
        
        # Run backtest with default parameters
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days_back)
        
        results = await backtesting_service.run_backtest(
            instrument1=inst1['tradingsymbol'],
            instrument2=inst2['tradingsymbol'],
            token1=token1,
            token2=token2,
            from_date=from_date,
            to_date=to_date
        )
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/backtest/trades/{backtest_id}")
async def get_backtest_trades(
    backtest_id: str,
    limit: int = 50,
    status: Optional[str] = None
):
    """Get list of trades from a backtest"""
    try:
        # Load backtest results
        filename = f"logs/backtest_{backtest_id}.json"
        
        with open(filename, 'r') as f:
            results = json.load(f)
        
        trades = results.get('trades', [])
        
        # Filter by status if provided
        if status == "winning":
            trades = [t for t in trades if t['pnl'] > 0]
        elif status == "losing":
            trades = [t for t in trades if t['pnl'] < 0]
        
        # Apply limit
        trades = trades[:limit]
        
        return {
            "backtest_id": backtest_id,
            "total_trades": len(results.get('trades', [])),
            "filtered_trades": len(trades),
            "trades": trades
        }
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Backtest results not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/backtest/summary/{backtest_id}")
async def get_backtest_summary(backtest_id: str):
    """Get summary of backtest results"""
    try:
        filename = f"logs/backtest_{backtest_id}.json"
        
        with open(filename, 'r') as f:
            results = json.load(f)
        
        return {
            "backtest_id": backtest_id,
            "period": results.get('period'),
            "instruments": results.get('instruments'),
            "config": results.get('config'),
            "metrics": results.get('metrics'),
            "monthly_returns": results.get('monthly_returns')
        }
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Backtest results not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/backtest/compare")
async def compare_strategies(
    instrument1: str,
    instrument2: str,
    days_back: int = 30,
    configs: List[Dict] = None,
    backtesting_service: BacktestingService = Depends(get_backtesting_service),
    instrument_lookup: InstrumentLookup = Depends(get_instrument_lookup)
):
    """Compare multiple strategy configurations"""
    try:
        if configs is None:
            # Default configurations to compare
            configs = [
                {"entry_z_score": 1.5, "exit_z_score": 0.5, "stop_loss_z_score": 2.5},
                {"entry_z_score": 2.0, "exit_z_score": 0.5, "stop_loss_z_score": 3.0},
                {"entry_z_score": 2.5, "exit_z_score": 0.75, "stop_loss_z_score": 3.5}
            ]
        
        # Get tokens
        token1 = instrument_lookup.get_instrument_token(instrument1)
        token2 = instrument_lookup.get_instrument_token(instrument2)
        
        if not token1 or not token2:
            raise HTTPException(status_code=404, detail="Instruments not found")
        
        # Set date range
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days_back)
        
        # Run backtests for each config
        comparison_results = []
        
        for i, config_dict in enumerate(configs):
            config = BacktestConfig(**config_dict)
            
            results = await backtesting_service.run_backtest(
                instrument1=instrument1,
                instrument2=instrument2,
                token1=token1,
                token2=token2,
                from_date=from_date,
                to_date=to_date,
                config=config
            )
            
            comparison_results.append({
                "config_id": i,
                "config": config_dict,
                "metrics": results['metrics']
            })
        
        # Find best configuration
        best_config = max(comparison_results, key=lambda x: x['metrics']['total_pnl'])
        
        return {
            "comparison": comparison_results,
            "best_config": best_config,
            "instruments": {
                "instrument1": instrument1,
                "instrument2": instrument2
            },
            "period": {
                "from": from_date.isoformat(),
                "to": to_date.isoformat(),
                "days": days_back
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))