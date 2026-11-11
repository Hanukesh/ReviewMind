import pandas as pd
import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Load real-world dataset
data = pd.read_csv("dataset_large.csv")

print("Original Dataset Loaded!")
print("Total Reviews:", len(data))


# Preserve important negation words
negations = {"not", "no", "never", "nor", "neither"}

stop_words = ENGLISH_STOP_WORDS - negations


# Text cleaning function
def clean_text(text):

    # Convert to lowercase
    text = str(text).lower()

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove special characters and numbers
    text = re.sub(r"[^a-z\s]", " ", text)

    # Tokenization
    words = text.split()

    # Remove stopwords
    words = [
        word for word in words
        if word not in stop_words
    ]

    # Join words
    return " ".join(words)


# Apply preprocessing
data["cleaned_review"] = data["review"].apply(clean_text)


# Remove empty reviews
data = data[data["cleaned_review"].str.strip() != ""]


# Save cleaned dataset
data.to_csv("cleaned_dataset.csv", index=False)


print("\nText preprocessing completed!")

print("\nOriginal Review:")
print(data["review"].iloc[0])

print("\nCleaned Review:")
print(data["cleaned_review"].iloc[0])

print("\nCleaned dataset saved successfully!")