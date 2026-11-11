
import pandas as pd
from datasets import load_dataset

print("Downloading Yelp dataset...")

# Load real-world dataset
data = load_dataset(
    "Yelp/yelp_review_full",
    split="train",
    streaming=True
)

# Select 20,000 reviews
reviews = list(data.take(20000))

# Convert into DataFrame
df = pd.DataFrame(reviews)

# Convert star ratings into sentiments
def classify_sentiment(rating):


    if rating <= 1:
        return "negative"

    elif rating == 2:
        return "neutral"

    else:
        return "positive"


# Prepare final dataset
df["sentiment"] = df["label"].apply(classify_sentiment)

df = df.rename(columns={"text": "review"})

df = df[["review", "sentiment"]]

# Remove empty and duplicate reviews
df = df.dropna()
df = df.drop_duplicates()

# Save dataset
df.to_csv("dataset_large.csv", index=False)

print("\nDataset preparation completed!")

print("\nTotal Reviews:", len(df))

print("\nSentiment Distribution:")
print(df["sentiment"].value_counts())

print("\nNew dataset saved as dataset_large.csv")
