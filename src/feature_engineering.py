import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Load cleaned data
train_df = pd.read_csv("data/processed/train_clean.csv")
test_df = pd.read_csv("data/processed/test_clean.csv")

vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_df["clean_text"])
X_test = vectorizer.transform(test_df["clean_text"])

# Save TF-IDF model
with open("../models/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# Save transformed data
import scipy.sparse
scipy.sparse.save_npz("data/processed/X_train.npz", X_train)
scipy.sparse.save_npz("data/processed/X_test.npz", X_test)
train_df["target"].to_csv("data/processed/y_train.csv", index=False)
