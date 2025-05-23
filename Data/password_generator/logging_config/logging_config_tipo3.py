import logging

logger_instance = logging.getLogger('password_generator')
logger_instance.setLevel(logging.INFO)
handler_instance = logging.StreamHandler()
formatter_instance = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler_instance.setFormatter(formatter_instance)
logger_instance.addHandler(handler_instance)

def initialize_logging():
    return logger_instance