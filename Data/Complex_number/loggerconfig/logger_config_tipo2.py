import logging

def create_custom_logger(identifier: str, severity: int = logging.INFO) -> logging.Logger:
    loggy = logging.getLogger(identifier)
    if not loggy.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        loggy.setLevel(severity)
        loggy.addHandler(handler)
    return loggy