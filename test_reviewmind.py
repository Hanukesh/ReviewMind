import os
import joblib
import sqlite3

from database import get_reviews

print("\n===== REVIEWMIND TESTING =====\n")

# 1. Check ML model
if os.path.exists("sentiment_model.pkl"):

    model = joblib.load("sentiment_model.pkl")

    print("PASS: ML model loaded successfully")

else:

    print("FAIL: ML model not found")
    exit()


# 2. Test predictions
test_reviews = [
    "I love this product. Excellent quality!",
    "Very bad product. Waste of money.",
    "The product arrived yesterday."
]

for review in test_reviews:

    prediction = model.predict([review])[0]

    print("\nReview:", review)
    print("Prediction:", prediction)


# 3. Check database
if os.path.exists("reviewmind.db"):

    conn = sqlite3.connect("reviewmind.db")

    cursor = conn.cursor()

    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='reviews'
    """)

    result = cursor.fetchone()

    conn.close()

    if result:

        print("\nPASS: Database table exists")

    else:

        print("\nFAIL: Database table not found")

else:

    print("\nFAIL: Database not found")


# 4. Check stored reviews
try:

    df = get_reviews()

    print("\nTotal Stored Reviews:", len(df))

    print("PASS: Database records retrieved")

except Exception as error:

    print("\nFAIL: Database retrieval error")
    print(error)


print("\n===== TESTING COMPLETED =====")