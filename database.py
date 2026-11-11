import sqlite3
import pandas as pd

# Database connection
def connect_db():
    return sqlite3.connect("reviewmind.db")


# Create database table
def create_table():

    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            review TEXT NOT NULL,
            sentiment TEXT NOT NULL,
            confidence REAL,
            date TEXT
        )
    """)

    conn.commit()
    conn.close()


# Insert customer review
def add_review(review, sentiment, confidence, date):

    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO reviews
        (review, sentiment, confidence, date)
        VALUES (?, ?, ?, ?)
    """, (review, sentiment, confidence, date))

    conn.commit()
    conn.close()


# Retrieve all reviews
def get_reviews():

    conn = connect_db()

    df = pd.read_sql_query(
        "SELECT * FROM reviews ORDER BY id DESC",
        conn
    )

    conn.close()

    return df


# Delete all reviews
def clear_reviews():

    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM reviews")

    conn.commit()
    conn.close()