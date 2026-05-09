import sqlite3
import os
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv

load_dotenv()

def setup_memory():
    # 1. Ensure the directory exists
    db_dir = "Databases"

    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
        print(f"Directory created: {db_dir}")

    # 2. Connect to SQLite
    db_path = os.path.join(db_dir, "petesinn.sqlite")

    sqlite_conn = sqlite3.connect(
        db_path,
        check_same_thread=False
    )

    # 3. Performance Optimizations
    sqlite_conn.execute("PRAGMA journal_mode=WAL;")
    sqlite_conn.execute("PRAGMA synchronous=NORMAL;")

    # 4. Return plain SqliteSaver (NO encryption)
    return SqliteSaver(sqlite_conn)

memory = setup_memory()