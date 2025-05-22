from datetime import datetime

def transform_date_format(input_string):
    """
    Convert a date string to ISO-8601 format.

    Args:
        input_string (str): A date string in the format 'YYYY-MM-DD' or 'MM/DD/YYYY'.

    Returns:
        str: Date in ISO-8601 format (YYYY-MM-DD).
    """
    try:
        # Try parsing the date in different formats
        if '-' in input_string:
            temp_date = datetime.strptime(input_string, '%Y-%m-%d')
        else:
            temp_date = datetime.strptime(input_string, '%m/%d/%Y')

        return temp_date.isoformat()
    except ValueError as ex:
        return f"Error: {ex}"

# Example usage
if __name__ == "__main__":
    print(transform_date_format("2024-10-06"))  # Outputs: 2024-10-06T00:00:00
    print(transform_date_format("10/06/2024"))  # Outputs: 2024-10-06T00:00:00