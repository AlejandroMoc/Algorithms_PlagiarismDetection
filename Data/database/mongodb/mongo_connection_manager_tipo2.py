from pymongo import MongoClient

def establish_connection_to_mongo(uri):
    """
    Establishes connection to a MongoDB instance.

    Args:
        uri (str): MongoDB URI string.

    Returns:
        pymongo.MongoClient: MongoDB connection object.
    """
    client_instance = MongoClient(uri)
    return client_instance