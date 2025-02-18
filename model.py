import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the training data
train_df = pd.read_csv('data/processed/train_clean.csv')

# Check for NaN values in the 'clean_text' column and drop rows with NaN
missing_train = train_df[train_df['clean_text'].isna()]

if not missing_train.empty:
    print("Missing entries in training data:")
    print(missing_train[['id', 'clean_text']])

train_df = train_df.dropna(subset=['clean_text'])

# Ensure 'clean_text' and 'target' columns exist in the training data
X_train = train_df['clean_text']
y_train = train_df['target']

# Initialize the TfidfVectorizer and Logistic Regression Model
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
model = LogisticRegression(C=10, solver='saga', max_iter=1000)

# Transform the training data with the vectorizer
X_train_tfidf = vectorizer.fit_transform(X_train)

# Train the model
model.fit(X_train_tfidf, y_train)

# Evaluate the model on training data
train_predictions = model.predict(X_train_tfidf)
train_accuracy = accuracy_score(y_train, train_predictions)

# Print Training Accuracy and Classification Report
print(f'Training Accuracy: {train_accuracy:.2f}')
print('Classification Report on Train Set:')
print(classification_report(y_train, train_predictions))

# Load the test data (the cleaned version)
test_df = pd.read_csv('data/processed/test_clean.csv')

# Check for NaN values in the 'clean_text' column and drop rows with NaN in the test set
missing_test = test_df[test_df['clean_text'].isna()]

if not missing_test.empty:
    print("Missing entries in test data:")
    print(missing_test[['id', 'clean_text']])

test_df = test_df.dropna(subset=['clean_text'])

# Ensure 'clean_text' column exists in the test data
X_test = test_df['clean_text']

# Transform the test data with the same vectorizer
X_test_tfidf = vectorizer.transform(X_test)

# Make predictions on the test set
test_predictions = model.predict(X_test_tfidf)

# Create a DataFrame for the test predictions
test_results = pd.DataFrame({
    'id': test_df['id'],  # Assuming 'id' is the identifier column in the test set
    'target': test_predictions
})

# Save the updated predictions to the existing CSV file in the processed data folder
test_results.to_csv('data/processed/test_predictions.csv', index=False)

print('Test predictions updated in data/processed/test_predictions.csv')
