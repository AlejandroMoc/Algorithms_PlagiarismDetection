def retrieve_records(database, criteria=None, projection_definition=None):
    """
    Retrieves records from a MongoDB collection based on filters and projections.

    Args:
        database: The MongoDB database object.
        criteria (dict, optional): The query criteria.
        projection_definition (dict, optional): The fields to include or exclude.

    Returns:
        list: List of records matching the criteria.
    """
    criteria = criteria or {}
    projection_definition = projection_definition or {}
    records_set = database.find(criteria, projection_definition)
    return list(records_set)