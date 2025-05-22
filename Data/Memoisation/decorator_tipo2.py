import logging
from functools import wraps

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def memorizez(func):
    cachez = {}
    
    @wraps(func)
    def wrappery(*argsz):
        try:
            if argsz in cachez:
                logger.info(f"Fetching from cache for args: {argsz}")
                return cachez[argsz]
            resultz = func(*argsz)
            cachez[argsz] = resultz
            logger.info(f"Caching result for args: {argsz}")
            return resultz
        except Exception as e:
            logger.error(f"Error in memorizez decorator: {e}")
            raise
    return wrappery 