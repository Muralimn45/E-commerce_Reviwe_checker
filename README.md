# E-commerce_Reviwe_checker

Here is a comprehensive README for your GitHub repository, based on the analysis of your "E-Commerce Fake Review Detector" project files.

-----

# 🛡️ E-Commerce Fake Review Detector

This project is a sophisticated **Natural Language Processing (NLP)** application built using **Streamlit** that classifies product reviews as **Legitimate** or **Fake**. It features a custom **Selenium-based web scraper** to pull live product reviews from Amazon, which are then analyzed using a pre-trained **Logistic Regression** model.

## ✨ Features

  * **Review Classification:** Predicts if a review is **Legitimate (✅)** or **Fake (🚨)** using a trained Logistic Regression model.
  * **Live Web Scraping:** Uses a **stealthy, dynamic Selenium crawler** (`review_crawler.py`) to scrape multiple pages of reviews directly from Amazon product URLs.
  * **Sentiment Analysis:** Provides additional context by calculating **TextBlob Sentiment Polarity** for each review.
  * **Flexible Input:** Supports three modes of operation:
    1.  Testing a **single review** text input.
    2.  Scraping and classifying reviews from an **Amazon URL**.
    3.  Classifying reviews from an **uploaded CSV file**.
  * **Model Persistence:** The trained classification model and vectorizer are saved using `pickle` for quick loading via Streamlit's `@st.cache_resource`.

-----

## 💻 Project Structure

The project is organized into three main Python scripts and a resource directory:

```
.
├── app.py                      # Streamlit frontend & Classification logic
├── Model_Training..py          # Script for model training and saving
├── review_crawler.py           # Selenium-based Amazon web scraper
├── amazon_reviews_2019.csv     # Sample dataset for model training (Required)
├── requirements.txt            # Project dependencies
└── Models/                     # Directory for persistence artifacts
    ├── best_model.pkl          # Trained Logistic Regression classifier
    └── count_vectorizer.pkl    # Fitted CountVectorizer
```

-----

## 🛠️ Setup and Installation

### Prerequisites

You need **Python 3.8+** installed on your system. The web scraping component requires a Chrome browser installation as `review_crawler.py` uses `webdriver-manager` to automatically handle the ChromeDriver setup.

### 1\. Clone the repository

```bash
git clone <repository-url>
cd e-commerce-fake-review-detector
```

### 2\. Install Dependencies

Install all necessary Python packages using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 3\. Obtain Training Data

Ensure you have a review dataset named `amazon_reviews_2019.csv` in the root directory. This dataset is required by `Model_Training..py`. *Note: The size and columns of this dataset should match the script's expectations (specifically, a `verified_purchase` column for labeling and a `review_text` column).*

### 4\. Train the Model

You must run the training script once to generate the required model and vectorizer files in the `Models/` directory.

```bash
python Model_Training..py
```

This script will print the model's performance metrics (Accuracy, Classification Report) and confirm the successful saving of the `.pkl` files.

-----

## 🚀 Running the Application

Start the Streamlit application from the terminal:

```bash
streamlit run app.py
```

A web browser will automatically open, displaying the **Fake Review Detector** interface.

-----

## ⚙️ Core Implementation Details

### Model Pipeline

The machine learning classification uses a standard NLP pipeline:

1.  **Preprocessing:** Text is cleaned (special characters removed), converted to lowercase, stop words are removed, and the remaining words are **stemmed** (`PorterStemmer`).
2.  **Feature Extraction:** A **CountVectorizer** extracts features, considering **Unigrams and Bigrams** (`ngram_range=(1, 2)`), limited to the top 5000 features.
3.  **Classification:** A **Logistic Regression** model is trained on these features, mapping the `verified_purchase` status (True/False) to the Legitimate/Fake label.

### Web Scraper (`review_crawler.py`)

The scraper is designed to be robust against Amazon's anti-bot measures:

  * It uses **Selenium** with a **headless Chrome** instance.
  * It disables images and CSS for **faster loading**.
  * It uses **stealth techniques** like manipulating the `navigator.webdriver` property and custom user agents to mimic human browsing behavior.
  * It handles **pagination** to collect reviews across multiple pages (up to `MAX_PAGES`).

-----

I can also help you draft the content for the `amazon_reviews_2019.csv` file if you need sample data or have any questions about the data format.

RESULTS : <img width="1917" height="860" alt="Image" src="https://github.com/user-attachments/assets/013991b2-236e-40d4-836b-88520aa95500" />
