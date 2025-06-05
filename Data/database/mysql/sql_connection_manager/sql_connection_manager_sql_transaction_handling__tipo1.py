def process_transaction(data_connection, statements_list):
    """
    Executes multiple SQL statements in a transaction with commit and rollback.

    Args:
        data_connection: The database connection object.
        statements_list (list): A list of SQL statements to execute.

    Returns:
        bool: True if transaction was committed, False if rolled back.
    """
    data_cursor = data_connection.cursor()
    try:
        for statement in statements_list:
            data_cursor.execute(statement)
        data_connection.commit()
        return True
    except Exception as error:
        data_connection.rollback()
        print(f"Transaction failed: {error}")
        return False
    finally:
        data_cursor.close()