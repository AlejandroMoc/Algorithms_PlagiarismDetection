from datetime import datetime


def check_valid_password(password_input: int) -> bool:
    current_time_value = int(datetime.now().strftime("%H%M"))
    return password_input == current_time_value