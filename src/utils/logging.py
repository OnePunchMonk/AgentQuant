import logging
import sys
from src.utils.config import config

def setup_logging():
    """
    Configures the root logger for the application.
    """
    log_level = config.get("log_level", "INFO").upper()
    
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Avoid adding duplicate handlers if this function is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()
        
    # Create a handler to print to the console (stderr)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    
    # Create a formatter and add it to the handler
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    
    # Add the handler to the logger
    logger.addHandler(handler)

    logging.info(f"Logging initialized with level {log_level}")