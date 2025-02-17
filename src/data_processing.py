import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    """
    Preprocess the input text by lowering the case and handling non-string values.
    
    Args:
        text (str): Input text to preprocess.
        
    Returns:
        str: Processed text in lowercase.
    """
    if not isinstance(text, str):
        return ""
    return text.lower()

def load_data(train_path, test_path):
    """
    Load the training and test data from the specified paths.
    
    Args:
        train_path (str): Path to the training dataset.
        test_path (str): Path to the test dataset.
        
    Returns:
        pd.DataFrame, pd.DataFrame: Loaded training and test data.
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def save_data(train_df, test_df, train_path, test_path):
    """
    Save the processed training and test data to specified paths.
    
    Args:
        train_df (pd.DataFrame): Processed training dataset.
        test_df (pd.DataFrame): Processed test dataset.
        train_path (str): Path to save the processed training data.
        test_path (str): Path to save the processed test data.
    """
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

def main():
    # File paths
    train_file = "data/processed/train_clean.csv"
    test_file = "data/raw/test.csv"
    train_cleaned_file = "data/processed/train_clean.csv"
    test_cleaned_file = "data/processed/test_clean.csv"

    # Load data
    train_df, test_df = load_data(train_file, test_file)

    # Apply preprocessing
    train_df["clean_text"] = train_df["clean_text"].fillna("").apply(preprocess_text)
    test_df["clean_text"] = test_df["content"].apply(preprocess_text)

    # Save processed data
    save_data(train_df, test_df, train_cleaned_file, test_cleaned_file)

if __name__ == "__main__":
    main()
