import joblib
import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score

# Ensure NumPy compatibility
np.load.__defaults__ = (None, True, True, 'ASCII')

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "mental_health_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")
TRAIN_DATA_PATH = os.path.join(BASE_DIR, "data/raw/train.csv")

# Load model and vectorizer
def load_model():
    """Loads the trained ML model and vectorizer."""
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        return model, vectorizer
    except Exception as e:
        print(f"Error loading model or vectorizer: {e}")
        return None, None

def evaluate_model():
    """Evaluates the model on the training dataset."""
    # Load training data
    train_df = pd.read_csv(TRAIN_DATA_PATH)

    # Ensure required columns exist
    if "id" not in train_df.columns or "content" not in train_df.columns or "target" not in train_df.columns:
        raise ValueError("Train data must contain 'id', 'content', and 'target' columns.")

    # Handle missing values in 'content'
    train_df["content"] = train_df["content"].fillna("")

    # Load model & vectorizer
    model, vectorizer = load_model()
    if model is None or vectorizer is None:
        return

    # Transform content and make predictions
    X_train = vectorizer.transform(train_df["content"])
    predictions = model.predict(X_train)

    # Compute accuracy
    accuracy = accuracy_score(train_df["target"], predictions)

    # Print accuracy
    print(f"Model Accuracy on Training Set: {accuracy:.4f}")

if __name__ == "__main__":
    evaluate_model()
