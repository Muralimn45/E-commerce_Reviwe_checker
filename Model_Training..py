import pandas as pd
import nltk
import re
import pickle
import os
import sys
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# --- Configuration ---
# NOTE: This file (amazon_reviews_2019.csv) must exist in the same directory as this script.
EXTERNAL_FILE = 'amazon_reviews_2019.csv'
MODELS_DIR = 'Models'
MODEL_PATH = os.path.join(MODELS_DIR, 'best_model.pkl')
VECTORIZER_PATH = os.path.join(MODELS_DIR, 'count_vectorizer.pkl')

# --- 1. Initialization and Preprocessing Setup ---

try:
    print("Initializing NLTK resources...")
    # Attempt to download stopwords quietly
    nltk.download('stopwords', quiet=True)
except Exception as e:
    print(f"Warning: Could not download NLTK resources. Error: {e}")

# Global Preprocessing objects
STOP_WORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()


def preprocess_text(text):
    """
    Cleans text, removes stopwords, and performs stemming.
    Must be IDENTICAL to the function used in app.py.
    """
    if not isinstance(text, str):
        # Handle non-string types gracefully
        return ""

    # Remove non-alphabetic characters and numbers, preserving spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I | re.A)
    text = text.lower()
    words = text.split()

    # Remove stopwords and apply stemming
    words = [STEMMER.stem(word) for word in words if word not in STOP_WORDS]

    return " ".join(words)


# --- 2. Main Training Logic ---

def train_and_save_model():
    """Loads data, trains the model, and saves artifacts."""

    try:
        # Load the dataset
        # FIX: Added encoding='latin1' to handle non-UTF-8 characters in the CSV
        df = pd.read_csv(EXTERNAL_FILE, encoding='latin1')
    except FileNotFoundError:
        print(f"❌ ERROR: Training file '{EXTERNAL_FILE}' not found. Please place it in the script directory.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"❌ ERROR: Training file '{EXTERNAL_FILE}' is empty.")
        sys.exit(1)

    # Map 'True'/'False' from the CSV to 1 (Legitimate) and 0 (Fake)
    # This assumes 'verified_purchase' is the label source, where 'True' = Legitimate.
    df['label'] = df['verified_purchase'].apply(lambda x: 1 if x else 0)

    # Select features (Review Text) and labels
    X = df['review_text'].astype(str)
    y = df['label']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Dataset loaded. Total: {len(df)}. Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    # Apply preprocessing to training data
    print("Starting text preprocessing...")
    X_train = X_train.apply(preprocess_text)
    X_test = X_test.apply(preprocess_text)
    print("Preprocessing complete.")

    # Feature Extraction (CountVectorizer with N-grams)
    print("Starting feature extraction (CountVectorizer)...")
    count_vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=5000)
    X_train_c = count_vectorizer.fit_transform(X_train)
    X_test_c = count_vectorizer.transform(X_test)
    print(f"Feature extraction complete. Feature space size: {X_train_c.shape[1]}")

    # Model Training (Logistic Regression)
    model = LogisticRegression(solver='liblinear', random_state=42, max_iter=1000)
    model.fit(X_train_c, y_train)
    print("✅ Logistic Regression model trained successfully.")

    # Evaluation
    y_pred = model.predict(X_test_c)
    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    # Display performance metrics
    print(classification_report(y_test, y_pred, target_names=['Fake (0)', 'Legitimate (1)']))

    # Save Model and Vectorizer
    try:
        os.makedirs(MODELS_DIR, exist_ok=True)
        pickle.dump(model, open(MODEL_PATH, 'wb'))
        pickle.dump(count_vectorizer, open(VECTORIZER_PATH, 'wb'))
        print(f"✅ Model and Vectorizer successfully saved to '{MODELS_DIR}/'.")
    except Exception as e:
        print(f"❌ Error saving models: {e}")


# --- Execution Entry Point ---
if __name__ == "__main__":
    train_and_save_model()