def add_record(storage, record_entry):
    """
    Incorporates an entry into a storage system.

    Args:
        storage: The storage system object.
        record_entry (dict): The entry to add.

    Returns:
        EntryId: The identifier of the added entry.
    """
    outcome = storage.insert_one(record_entry)
    return outcome.inserted_id
