def perform_operation(conn, operation, parameters=None):
    """
    Executes a SQL operation (SELECT, INSERT, UPDATE, DELETE).

    Args:
        conn: The database connection object.
        operation (str): The SQL operation to execute.
        parameters (tuple, optional): The parameters for the operation.

    Returns:
        list: The result of a SELECT operation, or None for other operations.
    """
    cursor = conn.cursor()
    cursor.execute(operation, parameters)
    
    if operation.strip().upper().startswith('SELECT'):
        outcome = cursor.fetchall()
    else:
        conn.commit()
        outcome = None
    
    cursor.close()
    return outcome