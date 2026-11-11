import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# 1. Load cleaned dataset
data = pd.read_csv("cleaned_dataset.csv")

X = data["cleaned_review"]
y = data["sentiment"]

print("Dataset loaded successfully!")
print("Total reviews:", len(data))

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Create algorithms
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),

    "Naive Bayes": MultinomialNB()
}

results = {}

# 4. Train and evaluate models
for name, classifier in models.items():

    print("\nTraining:", name)

    model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000)),
        ("classifier", classifier)
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test, predictions, average="weighted", zero_division=0
    )

    recall = recall_score(
        y_test, predictions, average="weighted", zero_division=0
    )

    f1 = f1_score(
        y_test, predictions, average="weighted", zero_division=0
    )

    results[name] = {
        "model": model,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    print("\nClassification Report:")
    print(classification_report(
        y_test, predictions, zero_division=0
    ))

# 5. Compare results
print("\n========== MODEL COMPARISON ==========")

for name, result in results.items():

    print("\nModel:", name)
    print("Accuracy:", round(result["accuracy"] * 100, 2), "%")
    print("Precision:", round(result["precision"], 2))
    print("Recall:", round(result["recall"], 2))
    print("F1 Score:", round(result["f1_score"], 2))

# 6. Select best model using F1 score
best_name = max(
    results,
    key=lambda name: results[name]["f1_score"]
)

best_model = results[best_name]["model"]

# 7. Save best model
joblib.dump(best_model, "sentiment_model.pkl")

print("\nBest Model:", best_name)
print("Best model saved successfully!")