import pandas as pd
import numpy as np
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "data/raw/train.csv")
TEST_PATH = os.path.join(BASE_DIR, "data/raw/test.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data/processed/test_predictions.csv")

def train_and_predict():
    """Trains a model using train.csv and generates predictions for test.csv."""
    
    # Load datasets
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # Ensure required columns exist
    if "id" not in train_df.columns or "content" not in train_df.columns or "target" not in train_df.columns:
        raise ValueError("Train data must contain 'id', 'content', and 'target' columns.")
    if "id" not in test_df.columns or "content" not in test_df.columns:
        raise ValueError("Test data must contain 'id' and 'content' columns.")

    # Handle missing values
    train_df["content"] = train_df["content"].fillna("")
    test_df["content"] = test_df["content"].fillna("")

    # Convert text into TF-IDF features
    vectorizer = TfidfVectorizer(max_features=5000)  # Use top 5000 features
    X_train = vectorizer.fit_transform(train_df["content"])
    y_train = train_df["target"]

    # Train a Logistic Regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Transform test data and make predictions
    X_test = vectorizer.transform(test_df["content"])
    predictions = model.predict(X_test)

    # Save predictions in required format
    output_df = pd.DataFrame({"id": test_df["id"], "target": predictions})
    output_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Predictions saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    train_and_predict()
