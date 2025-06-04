import logging
import sys
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pythonjsonlogger import jsonlogger

def setup_logging(log_level: str = "INFO", log_dir: str = "logs"):
    """Setup logging configuration for the application"""
    
    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Configure log format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    json_format = "%(timestamp)s %(level)s %(name)s %(message)s"
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler with standard format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(log_format)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler for general logs
    file_handler = RotatingFileHandler(
        log_path / "app.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(log_format)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # JSON file handler for structured logs
    json_handler = RotatingFileHandler(
        log_path / "app.json",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5
    )
    json_handler.setLevel(logging.DEBUG)
    json_formatter = jsonlogger.JsonFormatter(json_format)
    json_handler.setFormatter(json_formatter)
    root_logger.addHandler(json_handler)
    
    # Error file handler
    error_handler = RotatingFileHandler(
        log_path / "errors.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    error_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    )
    error_handler.setFormatter(error_formatter)
    root_logger.addHandler(error_handler)
    
    # Trading specific logger
    trading_logger = logging.getLogger("trading")
    trading_handler = RotatingFileHandler(
        log_path / f"trading_{datetime.now().strftime('%Y%m%d')}.log",
        maxBytes=50 * 1024 * 1024,  # 50 MB
        backupCount=10
    )
    trading_handler.setLevel(logging.DEBUG)
    trading_formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    trading_handler.setFormatter(trading_formatter)
    trading_logger.addHandler(trading_handler)
    trading_logger.setLevel(logging.DEBUG)
    trading_logger.propagate = False
    
    # WebSocket specific logger
    ws_logger = logging.getLogger("websocket")
    ws_handler = RotatingFileHandler(
        log_path / "websocket.log",
        maxBytes=20 * 1024 * 1024,  # 20 MB
        backupCount=3
    )
    ws_handler.setLevel(logging.DEBUG)
    ws_formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    ws_handler.setFormatter(ws_formatter)
    ws_logger.addHandler(ws_handler)
    ws_logger.setLevel(logging.DEBUG)
    ws_logger.propagate = False
    
    # Suppress noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("kiteconnect").setLevel(logging.INFO)
    
    logging.info(f"Logging setup complete. Log level: {log_level}")

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance"""
    return logging.getLogger(name)

def log_performance(func):
    """Decorator to log function performance"""
    import functools
    import time
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            elapsed_time = (time.time() - start_time) * 1000  # ms
            logger.debug(f"{func.__name__} completed in {elapsed_time:.2f}ms")
            return result
        
        except Exception as e:
            elapsed_time = (time.time() - start_time) * 1000  # ms
            logger.error(f"{func.__name__} failed after {elapsed_time:.2f}ms: {str(e)}")
            raise
    
    return wrapper