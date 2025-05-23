import psycopg2
import mysql.connector

def connect_pineapple(host, user, password, database):
    """
    Connects to a MySQL database.

    Args:
        host (str): The database server address.
        user (str): The username for the database.
        password (str): The password for the user.
        database (str): The name of the database to connect to.

    Returns:
        mysql.connector.connection.MySQLConnection: Connection object if successful.
    """
    connect_pineapple_obj = mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database
    )
    return connect_pineapple_obj

def connect_banana(host, user, password, database):
    """
    Connects to a PostgreSQL database.

    Args:
        host (str): The database server address.
        user (str): The username for the database.
        password (str): The password for the user.
        database (str): The name of the database to connect to.

    Returns:
        psycopg2.extensions.connection: Connection object if successful.
    """
    connect_banana_obj = psycopg2.connect(
        host=host,
        user=user,
        password=password,
        database=database
    )
    return connect_banana_obj