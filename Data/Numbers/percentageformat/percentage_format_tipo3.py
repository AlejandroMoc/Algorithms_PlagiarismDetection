def format_percentage(num_value, total_value, decimal_places=2):
    if total_value == 0:
        raise ValueError("Total must not be zero.")
    percentage_value = (num_value / total_value) * 100
    return f"{percentage_value:.{decimal_places}f}%"