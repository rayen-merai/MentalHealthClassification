import os

print("Preprocessing text data...")
os.system("python src/data_processing.py")

print("Extracting features...")
os.system("python src/feature_engineering.py")

print("Training the model...")
os.system("python src/model_training.py")

print("Running inference on test data...")
os.system("python src/inference.py")

print("Pipeline completed! Check 'outputs/submission.csv'.")
