# test_connection.py
import psycopg2
from config import DB_HOST, DB_NAME, DB_USER, DB_PASS, DB_PORT

try:
    conn = psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        port=DB_PORT
    )
    print("Database connection successful!")
except Exception as e:
    print(f"Database connection failed: {e}")
finally:
    if conn:
        conn.close()
