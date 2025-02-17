import pandas as pd
import pickle
import scipy.sparse

# Load test data and trained model
X_test = scipy.sparse.load_npz("data/processed/X_test.npz")

with open("models/logistic_regression.pkl", "rb") as f:
    model = pickle.load(f)

# Make predictions
test_predictions = model.predict(X_test)

# Load test file IDs
test_df = pd.read_csv("data/raw/test.csv")
test_df["target"] = test_predictions

# Save submission file
test_df[["id", "target"]].to_csv("outputs/submission.csv", index=False)
