import logging
import os

def get_logger(name="flight_data_logger", log_file="logs/app.log"):
    """Return a reusable file-only logger."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers to prevent duplicates or console output
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # File handler only
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Do NOT add StreamHandler -> no console output
    return logger
