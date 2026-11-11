import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score
)

# 1. Load dataset
data = pd.read_csv("cleaned_dataset.csv")

X = data["review"].fillna("")
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

# 3. Create improved model
model = Pipeline([

    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=30000,
        min_df=2,
        sublinear_tf=True
    )),

    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        C=2.0
    ))

])

# 4. Train model
print("\nTraining improved model...")

model.fit(X_train, y_train)

# 5. Predictions
predictions = model.predict(X_test)

# 6. Evaluate model
accuracy = accuracy_score(y_test, predictions)

f1 = f1_score(
    y_test,
    predictions,
    average="weighted"
)

print("\n========== MODEL RESULTS ==========")

print("Accuracy:", round(accuracy * 100, 2), "%")
print("F1 Score:", round(f1, 3))

print("\nClassification Report:")

print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

# 7. Confusion Matrix
labels = ["negative", "neutral", "positive"]

cm = confusion_matrix(
    y_test,
    predictions,
    labels=labels
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)

display.plot()
plt.title("ReviewMind - Improved Model")
plt.tight_layout()
plt.show()

# 8. Train final model using complete dataset
print("\nTraining final model...")

model.fit(X, y)

# 9. Save model
joblib.dump(model, "sentiment_model.pkl")

print("\nImproved model saved successfully!")