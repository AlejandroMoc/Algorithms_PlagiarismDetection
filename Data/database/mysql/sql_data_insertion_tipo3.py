def insert_bulk(conn, tbl, dat):
    cursor = conn.cursor()
    placeholders = ', '.join(['%s'] * len(dat[0]))
    query = f"INSERT INTO {tbl} VALUES ({placeholders})"
    cursor.executemany(query, dat)
    conn.commit()
    cursor.close()