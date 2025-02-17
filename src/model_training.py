import pandas as pd
import pickle
import scipy.sparse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load data
X_train = scipy.sparse.load_npz("data/processed/X_train.npz")
y_train = pd.read_csv("data/processed/y_train.csv")["target"]

# Split into train and validation sets
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train_split, y_train_split)

# Evaluate model
y_pred = model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred))

# Save model
with open("models/logistic_regression.pkl", "wb") as f:
    pickle.dump(model, f)
