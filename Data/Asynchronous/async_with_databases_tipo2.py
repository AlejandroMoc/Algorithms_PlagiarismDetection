import asyncio
import asyncpg
import logging

async def retrieve_data_from_database():
    conn = await asyncpg.connect(user='user', password='password', database='db', host='localhost')
    logging.info("Connected to database")
    information = await conn.fetch('SELECT * FROM my_table')
    await conn.close()
    return information

async def primary_function():
    data = await retrieve_data_from_database()
    logging.info(f"Data fetched from database: {data}")

if __name__ == "__main__":
    asyncio.run(primary_function()) 