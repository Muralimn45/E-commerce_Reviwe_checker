import streamlit as st
import subprocess
import pickle
import nltk
import pandas as pd
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import os
import sys

# --- CRITICAL CONFIGURATION: MUST BE FIRST ---\
st.set_page_config(page_title="🛡️ Fake Review Detector", layout="wide")
# --- END CONFIGURATION ---

# --- CONFIGURATION CONSTANTS ---\
MODEL_PATH = 'Models/best_model.pkl'
VECTORIZER_PATH = 'Models/count_vectorizer.pkl'
SCRAPED_FILE = 'scraped_reviews.csv'
SENTIMENT_THRESHOLD = 0.05
# Placeholder for the user's Amazon link (or a generic example)
DEFAULT_AMAZON_URL = "https://www.amazon.in/Sony-WH-1000XM5-Wireless-Cancelling-Headphones/dp/B0B4PR5C5F"


# --- 1. INITIALIZATION AND MODEL LOADING ---

@st.cache_resource
def load_resources():
    """Loads the model, vectorizer, and NLTK resources once for the session."""
    try:
        # Load the trained model and feature vectorizer
        model = pickle.load(open(MODEL_PATH, 'rb'))
        vectorizer = pickle.load(open(VECTORIZER_PATH, 'rb'))

        # Download NLTK resources if not present
        nltk.download('stopwords', quiet=True)
        sw = set(stopwords.words('english'))
        stemmer = PorterStemmer()

        return model, vectorizer, sw, stemmer
    except FileNotFoundError:
        st.error(
            f"❌ ERROR: Model files not found. Ensure '{MODEL_PATH}' and '{VECTORIZER_PATH}' exist. Run Model_Training..py first."
        )
        # Exit the application gracefully
        st.stop()
    except Exception as e:
        st.error(f"❌ An error occurred during resource loading: {e}")
        st.stop()


# Load all resources
model, vectorizer, STOP_WORDS, STEMMER = load_resources()


# --- 2. PREPROCESSING AND CLASSIFICATION FUNCTIONS ---

def preprocess_text(text):
    """
    Cleans text, removes stopwords, and performs stemming.
    Must be IDENTICAL to the function used in Model_Training..py.
    """
    if not isinstance(text, str):
        return ""

    # Remove non-alphabetic characters and numbers, preserving spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I | re.A)
    text = text.lower()
    words = text.split()

    # Remove stopwords and apply stemming
    words = [STEMMER.stem(word) for word in words if word not in STOP_WORDS]

    return " ".join(words)


def classify_review(text):
    """Predicts legitimacy, probability, and sentiment for a single review."""
    # 1. Preprocessing and Vectorization
    processed_text = preprocess_text(text)
    # The model expects a list of strings
    vectorized_text = vectorizer.transform([processed_text])

    # 2. Classification
    prediction = model.predict(vectorized_text)[0]  # 1 (Legitimate) or 0 (Fake)
    probabilities = model.predict_proba(vectorized_text)[0]
    prob_fake = probabilities[0]
    prob_legit = probabilities[1]

    # 3. Sentiment Analysis (using original text for better context)
    sentiment = TextBlob(text).sentiment.polarity

    # Determine final label
    label = "✅ Legitimate" if prediction == 1 else "🚨 Fake"

    # Format probabilities to 2 decimal places
    prob_legit_formatted = f"{prob_legit * 100:.2f}%"
    prob_fake_formatted = f"{prob_fake * 100:.2f}%"

    return label, prob_legit_formatted, prob_fake_formatted, sentiment


def display_classification_result(text, label, prob_legit, prob_fake, sentiment):
    """Displays the classification result in Streamlit."""

    if label.startswith("🚨"):
        st.error(f"**Final Classification: {label}**")
    else:
        st.success(f"**Final Classification: {label}**")

    col1, col2, col3 = st.columns(3)
    col1.metric("Legitimate Probability", prob_legit)
    col2.metric("Fake Probability", prob_fake)

    sentiment_color = "green" if sentiment > SENTIMENT_THRESHOLD else (
        "red" if sentiment < -SENTIMENT_THRESHOLD else "gray")
    col3.markdown(f"**Sentiment Polarity (TextBlob):** <span style='color:{sentiment_color}'>{sentiment:.4f}</span>",
                  unsafe_allow_html=True)
    st.caption(f"Review Text: {text}")


# --- 3. STREAMLIT APPLICATION LAYOUT ---

def main():
    st.title("🛡️ E-Commerce Fake Review Detector")
    st.markdown("A tool to classify reviews as **Legitimate** or **Fake** using a **Logistic Regression** classifier.")

    st.sidebar.title("App Modes")
    mode = st.sidebar.radio(
        "Select Classification Input",
        ("Test Single Review", "🕸️ Web Scraper and Classification", "Classify Uploaded CSV"),
        index=1  # Set default mode to scraper
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "The model determines legitimacy based on features like word frequency (N-grams) and other text characteristics.")

    if mode == "Test Single Review":
        st.header("Test Single Review")
        review_input = st.text_area(
            "Paste a single review here:",
            "This product is outstanding, works exactly as described and arrived quickly. Highly recommended.",
            height=150
        )

        if st.button("Classify Review"):
            if review_input:
                label, prob_legit, prob_fake, sentiment = classify_review(review_input)
                display_classification_result(review_input, label, prob_legit, prob_fake, sentiment)
            else:
                st.warning("Please enter a review to classify.")

    elif mode == "🕸️ Web Scraper and Classification":
        st.header("🕸️ Web Scraper and Classification (Selenium-Based)")
        st.markdown(
            "🚨 **CRITICAL NOTE**: The scraper now uses **Selenium** with **advanced stealth options** to bypass Amazon's anti-bot system. Success is not guaranteed, but the performance is optimized.")
        st.markdown("Enter an **Amazon product URL** to scrape the visible reviews and classify them.")
        st.markdown(
            "⚠️ If the scraper fails due to a block, a mock dataset will be used to demonstrate the classification feature.")

        url_input = st.text_input(
            "Enter Review URL:",
            DEFAULT_AMAZON_URL
        )

        if st.button("Start Scrape and Classify"):
            if url_input:
                st.info(f"Starting dynamic crawler for: **{url_input}**. This may take up to 60 seconds per page.")

                # Use spinner to show activity
                with st.spinner('Scraping in progress... (using Selenium/Chrome)'):
                    try:
                        # Call the external Python script (review_crawler.py)
                        # The scraper is now updated to handle multiple pages
                        result = subprocess.run(
                            [sys.executable, 'review_crawler.py', url_input],
                            capture_output=True,
                            text=True,
                            check=True,
                            timeout=300 # Increased timeout for multi-page scrape
                        )
                        st.success("✅ Scrape process complete. Starting classification...")

                    except subprocess.CalledProcessError as e:
                        st.error("❌ Scraping Failed!")
                        st.warning(
                            "The scraper script terminated with an error. Check the terminal for 'FATAL ERROR' messages from Selenium.")
                        st.code(f"Output (Stdout):\n{e.stdout}\n\nError (Stderr):\n{e.stderr}")
                        return
                    except subprocess.TimeoutExpired:
                        st.error("❌ Scraping Failed! The scraper took too long and timed out.")
                        return
                    except FileNotFoundError:
                        st.error("❌ Scraping Failed! Ensure 'review_crawler.py' exists in the same directory.")
                        return

                # Load the data generated by the crawler
                try:
                    df_scraped = pd.read_csv(SCRAPED_FILE)
                    if df_scraped.empty:
                         raise pd.errors.EmptyDataError("Scraped data is empty.")
                except FileNotFoundError:
                    st.error(
                        f"❌ Classification failed: The scraper did not produce the expected file: '{SCRAPED_FILE}'")
                    return
                except pd.errors.EmptyDataError:
                    # --- CRITICAL FIX: IF SCRAPE IS EMPTY, USE MOCK DATA ---
                    st.warning("⚠️ Scraper returned an empty dataset (Amazon block detected or no reviews).")
                    st.info(
                        "💡 **Displaying Mock Classification Output:** Using sample data to demonstrate the final result structure.")

                    mock_data = {
                        'Review Text': [
                            "This product is outstanding, works exactly as described and arrived quickly. Highly recommended.",
                            "Great value for money. Best headphones I have ever bought, sound is clear and noise cancelling works flawlessly.",
                            "BUY NOW! Five stars, five stars, five stars! Everyone must buy this, superb product!!",
                            "I am very disappointed with the battery life. It barely lasts three hours, and the volume is too low.",
                            "This headset is amazing! I bought ten of these for my family. The best ever product I used, must buy."
                        ],
                        'Rating': [5.0, 5.0, 5.0, 2.0, 5.0]
                    }
                    df_scraped = pd.DataFrame(mock_data)
                    # --- END MOCK DATA ---

                # Perform classification on the scraped data (or mock data)
                # The 'Review Text' column name is now enforced in the scraper script
                if 'Review Text' in df_scraped.columns:

                    # Apply classification and split the returned tuple into new columns
                    df_scraped[['Final Label', 'Prob_Legit', 'Prob_Fake', 'Sentiment']] = df_scraped[
                        'Review Text'].apply(
                        lambda x: pd.Series(classify_review(x))
                    )

                    # Display summary statistics
                    st.subheader(f"Classification Results for {len(df_scraped)} Reviews")

                    # Count and Summary Metrics
                    count_df = df_scraped['Final Label'].value_counts().reset_index()
                    count_df.columns = ['Label', 'Count']

                    legit_count = count_df[count_df['Label'].str.startswith('✅')]['Count'].sum()
                    fake_count = count_df[count_df['Label'].str.startswith('🚨')]['Count'].sum()

                    col_sum1, col_sum2, col_sum3 = st.columns(3)
                    col_sum1.metric("Total Reviews Scraped", len(df_scraped))
                    col_sum2.metric("Legitimate Count", legit_count)
                    col_sum3.metric("Fake Count", fake_count)

                    st.markdown("---")
                    st.subheader("Classified Scraped Reviews")

                    # Display classified data table in the exact format requested
                    st.dataframe(
                        df_scraped[
                            ['Review Text', 'Rating', 'Final Label', 'Prob_Legit', 'Prob_Fake', 'Sentiment']].head(
                            100).style.format({'Sentiment': '{:.4f}'}),
                        use_container_width=True
                    )
                else:
                    st.error("❌ Scraped file is missing the 'Review Text' column. Check scraper output.")

            else:
                st.warning("Please enter a valid URL.")

    elif mode == "Classify Uploaded CSV":
        st.header("Classify Uploaded CSV")
        st.markdown("Upload a CSV file containing reviews and select the column with the text.")

        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file is not None:
            try:
                df_uploaded = pd.read_csv(uploaded_file)

                # Let the user select the text column
                text_column = st.selectbox(
                    "Select the column containing the review text:",
                    options=df_uploaded.columns.tolist()
                )

                if st.button("Start Classification"):
                    st.info(f"Classifying reviews in column: **{text_column}**...")

                    # Ensure the selected column is not empty
                    if df_uploaded[text_column].isnull().all():
                        st.error(f"❌ The selected column '{text_column}' is empty.")
                        return

                    # Apply classification and split the returned tuple into new columns
                    # The slice [:4] is redundant here as classify_review returns 4 values, but kept for safety.
                    df_uploaded[['Final Label', 'Prob_Legit', 'Prob_Fake', 'Sentiment']] = df_uploaded[
                        text_column].apply(
                        lambda x: pd.Series(classify_review(x)[:4])
                    )

                    st.success("🎉 Classification complete!")

                    # Display summary statistics
                    st.subheader("Summary")

                    # Count and Summary Metrics
                    count_df = df_uploaded['Final Label'].value_counts().reset_index()
                    count_df.columns = ['Label', 'Count']

                    legit_count = count_df[count_df['Label'].str.startswith('✅')]['Count'].sum()
                    fake_count = count_df[count_df['Label'].str.startswith('🚨')]['Count'].sum()

                    col_sum1, col_sum2, col_sum3 = st.columns(3)
                    col_sum1.metric("Total Reviews Classified", len(df_uploaded))
                    col_sum2.metric("Legitimate Count", legit_count)
                    col_sum3.metric("Fake Count", fake_count)

                    # Display classified data table
                    st.markdown("---")
                    st.subheader("Classified Uploaded Reviews (Top 100)")
                    st.dataframe(
                        df_uploaded[[text_column, 'Final Label', 'Prob_Legit', 'Prob_Fake', 'Sentiment']].head(100),
                        use_container_width=True)

            except Exception as e:
                st.error(f"❌ Error processing uploaded file: {e}")


# --- EXECUTION ENTRY POINT ---
if __name__ == "__main__":
    main()