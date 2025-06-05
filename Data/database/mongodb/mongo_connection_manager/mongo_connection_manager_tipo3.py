from pymongo import MongoClient


    Args:
        uri (str): MongoDB URI string.

    Returns:
        pymongo.MongoClient: MongoDB connection object.
    """
    client = MongoClient(uri)
    return client


def connect_to_mongodb(uri):
    """
    Connects to a MongoDB instance.
