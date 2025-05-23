python
from pymongo import MongoClient

def establish_connection_to_mongodb(uri):
    connection = MongoClient(uri)
    return connection