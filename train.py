import pandas as pd
import numpy as np
import os
import joblib
import re
import string
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "data/raw/train.csv")
MODEL_PATH = os.path.join(BASE_DIR, "mental_health_xgb_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")

def clean_text(text):
    """Preprocess text: removes special characters, lowercases, and strips spaces."""
    if pd.isna(text):
        return ""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

def train_model():
    """Trains an optimized XGBoost model with reduced memory usage."""
    
    # Load training data
    train_df = pd.read_csv(TRAIN_PATH)

    # Ensure required columns exist
    if "id" not in train_df.columns or "content" not in train_df.columns or "target" not in train_df.columns:
        raise ValueError("Train data must contain 'id', 'content', and 'target' columns.")

    # Handle missing values and clean text
    train_df["content"] = train_df["content"].fillna("").apply(clean_text)

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        train_df["content"], train_df["target"], test_size=0.2, random_state=42, stratify=train_df["target"]
    )

    # Convert text into TF-IDF features with optimized parameters
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words="english", dtype=np.float32)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)

    # Encode string labels into numerical values
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)

    # Apply SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_tfidf_resampled, y_train_resampled = smote.fit_resample(X_train_tfidf, y_train_encoded)

    # Train XGBoost Model (Optimized)
    model = XGBClassifier(
        n_estimators=100,  # Reduce from 200 → 100
        learning_rate=0.1,
        max_depth=4,  # Reduce from 6 → 4
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",  # Uses optimized histogram-based algorithm
        random_state=42
    )
    model.fit(X_train_tfidf_resampled, y_train_resampled)

    # Evaluate on validation set
    val_predictions_encoded = model.predict(X_val_tfidf)
    val_predictions = label_encoder.inverse_transform(val_predictions_encoded)

    accuracy = accuracy_score(y_val, val_predictions)
    class_report = classification_report(y_val, val_predictions, digits=4)
    conf_matrix = confusion_matrix(y_val, val_predictions)

    # Save the trained model, vectorizer, and label encoder
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)

    # Print accuracy details
    print("\n=== Improved Model Training Complete! ===")
    print(f"Validation Accuracy: {accuracy:.4f}\n")
    print("=== Classification Report ===")
    print(class_report)

    # Plot confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    train_model()
