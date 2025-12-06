import sys
import os
import re
import pandas as pd
from urllib.parse import urlparse

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

# ------------------------- CONFIG --------------------------
SCRAPED_FILE = "scraped_reviews.csv"
LOAD_TIMEOUT = 30  # more stable for Amazon
MAX_PAGES = 5  # Maximum number of review pages to scrape (e.g., 50 reviews)


# -----------------------------------------------------------


def extract_asin(url: str):
    """
    Extracts ASIN from any Amazon product URL.
    Works with dp/, gp/, product/, and review URLs.
    """
    patterns = [
        r"/dp/([A-Z0-9]{10})",
        r"/gp/product/([A-Z0-9]{10})",
        r"/product-reviews/([A-Z0-9]{10})",
        r"/gp/aw/d/([A-Z0-9]{10})",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None


def setup_driver():
    """
    Configures a stealthy, optimized Selenium driver.
    """
    chrome_options = Options()

    # Disable images/CSS for faster load
    prefs = {
        "profile.managed_default_content_settings.images": 2,
        "profile.managed_default_content_settings.stylesheet": 2,
    }
    chrome_options.add_experimental_option("prefs", prefs)

    # Anti-detection
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option("useAutomationExtension", False)
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")

    # Headless session
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")

    # Realistic user agent
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)

        # Hide automation flag
        driver.execute_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )
        return driver

    except Exception as e:
        print(f"FATAL ERROR: Selenium driver failed to start: {e}")
        sys.exit(1)


def parse_html_for_reviews(html):
    """Parses HTML content to extract review text and rating."""
    soup = BeautifulSoup(html, "lxml")
    containers = soup.select('div[data-hook="review"]')
    page_reviews = []

    for block in containers:
        review_text = ""
        rating_value = None

        # Review Text
        text_elem = block.select_one('span[data-hook="review-body"]')
        if not text_elem:
            text_elem = block.select_one(".review-text")

        if text_elem:
            review_text = " ".join(text_elem.stripped_strings)

        # Rating
        rating_elem = block.select_one('i[data-hook="review-star-rating"]')
        if rating_elem and rating_elem.get("class"):
            rating_match = re.search(r"(\d+\.?\d*)", rating_elem.get_text())
            if rating_match:
                rating_value = float(rating_match.group(1))

        if review_text:
            page_reviews.append({
                "Review Text": review_text,  # Corrected column name to match app.py's expectation
                "Rating": rating_value
            })

    return page_reviews


def scrape_amazon_reviews(url: str):
    """
    Scrapes multiple pages of Amazon reviews using a stealth Selenium browser.
    Returns: list of review dictionaries.
    """
    asin = extract_asin(url)
    if not asin:
        print("ERROR: Could not detect a valid ASIN in this URL.")
        return []

    print(f"INFO: Detected ASIN: {asin}")

    parsed = urlparse(url)
    base_url = f"{parsed.scheme}://{parsed.netloc}/product-reviews/{asin}/?ie=UTF8"

    driver = setup_driver()
    reviews_list = []

    try:
        driver.set_page_load_timeout(LOAD_TIMEOUT)

        for page in range(1, MAX_PAGES + 1):
            page_url = f"{base_url}&pageNumber={page}"
            print(f"INFO: Scraping page {page}: {page_url}")
            driver.get(page_url)

            # Wait for at least one review block to load
            WebDriverWait(driver, LOAD_TIMEOUT).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'div[data-hook="review"]'))
            )

            # Get page source and parse reviews
            html = driver.page_source
            page_reviews = parse_html_for_reviews(html)

            if not page_reviews:
                print(f"WARNING: No reviews found on page {page}. Stopping pagination.")
                break

            reviews_list.extend(page_reviews)
            print(f"INFO: Extracted {len(page_reviews)} reviews from page {page}. Total: {len(reviews_list)}")

    except Exception as e:
        print("ERROR: Page did not load correctly or Amazon blocked the request.")
        print(f"Reason: {e}")
        try:
            print("Page Title:", driver.title)
        except:
            pass

    finally:
        driver.quit()

    print(f"SUCCESS: Total of {len(reviews_list)} reviews extracted across all pages.")
    return reviews_list


# ---------------- EXECUTION ENTRY ----------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scraper.py <amazon_product_url>")
        sys.exit(1)

    url = sys.argv[1]

    reviews = scrape_amazon_reviews(url)
    df = pd.DataFrame(reviews)

    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(SCRAPED_FILE) or ".", exist_ok=True)
        # Use .copy() to prevent SettingWithCopyWarning if dataframe manipulation was needed
        df.to_csv(SCRAPED_FILE, index=False)
        print(f"Saved to: {SCRAPED_FILE}")
        sys.exit(0)

    except Exception as e:
        print(f"ERROR: Could not save CSV file: {e}")
        sys.exit(1)