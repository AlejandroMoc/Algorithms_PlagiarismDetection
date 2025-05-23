import psycopg2
import mysql.connector

def connect_to_db_mysql(host, user, password, database):
    connection = mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database
    )
    return connection

def connect_to_db_postgresql(host, user, password, database):
    connection = psycopg2.connect(
        host=host,
        user=user,
        password=password,
        database=database
    )
    return connection