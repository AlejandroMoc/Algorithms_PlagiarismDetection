def modify_records(data_set, filter_rules, new_values, many=False):
    """
    Updates one or more records in a database collection.

    Args:
        data_set: The database collection object.
        filter_rules (dict): The query filter to match records.
        new_values (dict): The new values to update.
        many (bool, optional): If True, updates all matching records.

    Returns:
        dict: The update result details.
    """
    if many:
        outcome = data_set.update_many(filter_rules, {'$set': new_values})
    else:
        outcome = data_set.update_one(filter_rules, {'$set': new_values})
    return outcome.raw_result