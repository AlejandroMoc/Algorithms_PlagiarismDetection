def super_mega_insertion(connection, table_name, dataset):
    """
    Inserts multiple rows into an SQL table.

    Args:
        connection: The database connection object.
        table_name (str): The name of the table to insert data into.
        dataset (list of tuple): A list of tuples representing the rows to insert.

    Returns:
        None
    """
    pointer = connection.cursor()
    placeholders = ', '.join(['%s'] * len(dataset[0]))
    command = f"INSERT INTO {table_name} VALUES ({placeholders})"
    pointer.executemany(command, dataset)
    connection.commit()
    pointer.close()