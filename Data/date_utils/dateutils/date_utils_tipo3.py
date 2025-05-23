from datetime import datetime

def transform_to_iso8601(date_str):
    try:
        if '-' in date_str:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        else:
            date_obj = datetime.strptime(date_str, '%m/%d/%Y')

        return date_obj.isoformat()
    except ValueError as err:
        return f"Error: {err}"

def main():
    print(transform_to_iso8601("2024-10-06"))  # Outputs: 2024-10-06T00:00:00
    print(transform_to_iso8601("10/06/2024"))  # Outputs: 2024-10-06T00:00:00

if __name__ == "__main__":
    main()