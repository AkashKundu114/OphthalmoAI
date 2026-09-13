import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sqlite3
from backend.db import Base, engine

def sync_schema():
    db_path = "ophthalmoai.db"
    if not os.path.exists(db_path):
        print(f"{db_path} does not exist. Creating tables...")
        Base.metadata.create_all(bind=engine)
        print("Done.")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Ensure all tables defined in metadata exist
    Base.metadata.create_all(bind=engine)

    # For each table in metadata, inspect columns and add missing ones
    for table_name, table in Base.metadata.tables.items():
        cursor.execute(f"PRAGMA table_info({table_name});")
        existing_cols = {row[1]: row for row in cursor.fetchall()}
        
        for column in table.columns:
            if column.name not in existing_cols:
                col_type = str(column.type)
                # Map column type for SQLite
                if "FLOAT" in col_type.upper() or "REAL" in col_type.upper():
                    sql_type = "REAL"
                elif "INT" in col_type.upper() or "BOOL" in col_type.upper():
                    sql_type = "INTEGER"
                else:
                    sql_type = "TEXT"
                
                default_clause = ""
                if column.default is not None and column.default.is_scalar:
                    val = column.default.arg
                    default_clause = f" DEFAULT '{val}'" if isinstance(val, str) else f" DEFAULT {val}"

                alter_stmt = f"ALTER TABLE {table_name} ADD COLUMN {column.name} {sql_type}{default_clause};"
                print(f"Executing: {alter_stmt}")
                try:
                    cursor.execute(alter_stmt)
                    conn.commit()
                except Exception as ex:
                    print(f"Error altering {table_name}.{column.name}: {ex}")

    conn.close()
    print("Database schema successfully synchronized!")

if __name__ == "__main__":
    sync_schema()
