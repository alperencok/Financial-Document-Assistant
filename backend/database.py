import sqlite3
import os

def init_db(db_path="../data/rag.db"):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT,
            page INTEGER,
            content TEXT,
            embedding TEXT
        )
    """)
    conn.commit()
    return conn

def clear_chunks(conn):
    conn.execute("DELETE FROM chunks")
    conn.commit()