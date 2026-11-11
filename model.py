
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
    ConfusionMatrixDisplay
)

# 1. Load dataset
data = pd.read_csv("dataset.csv")

X = data["review"]
y = data["sentiment"]

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# 3. Create ML model
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000))
])

# 4. Train model
model.fit(X_train, y_train)

print("Model training completed!")

# 5. Make predictions
predictions = model.predict(X_test)

# 6. Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", accuracy * 100, "%")

# 7. Classification report
print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

# 8. Create confusion matrix
labels = ["negative", "neutral", "positive"]

cm = confusion_matrix(
    y_test,
    predictions,
    labels=labels
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Negative", "Neutral", "Positive"]
)

# 9. Display graph
display.plot()
plt.title("ReviewMind - Confusion Matrix")
plt.tight_layout()
plt.show()

# 10. Train final model using complete dataset
model.fit(X, y)

# 11. Save trained model
joblib.dump(model, "sentiment_model.pkl")

print("\nFinal model saved successfully!")
