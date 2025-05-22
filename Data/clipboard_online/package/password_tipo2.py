from datetime import datetime


def secret_key(encoded_key: int) -> bool:
    current_moment = int(datetime.now().strftime("%H%M"))
    return encoded_key == current_moment