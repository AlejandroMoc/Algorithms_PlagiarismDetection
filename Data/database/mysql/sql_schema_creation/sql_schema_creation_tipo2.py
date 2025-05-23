def generate_chart(connection, design_chart_query):
    """
    Generates a chart based on the given SQL query.

    Args:
        connection: The database connection object.
        design_chart_query (str): SQL query to design a chart.

    Returns:
        None
    """
    pointer = connection.cursor()
    pointer.execute(design_chart_query)
    connection.commit()
    pointer.close()
