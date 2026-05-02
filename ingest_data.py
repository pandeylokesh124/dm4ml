import pandas as pd
import requests
import logging
import os
import sys
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True,
    handlers=[
        logging.FileHandler("ingestion.log", mode='a'),
        logging.StreamHandler(sys.stdout)  # This ensures you see it in the terminal
    ]
)


def ingest_interactions(file_path):
    try:
        # Check if file exists first to avoid silent failures
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            return

        df = pd.read_csv(file_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        save_path = f"data/raw/interactions/{timestamp}_interactions.csv"

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)

        # This will now appear in both the file and console
        logging.info(f"Successfully ingested interactions to {save_path}")

    except Exception as e:
        logging.error(f"Failed to ingest CSV: {e}")


def ingest_product_api(api_url):
    try:
        # Sixmulation check: if it's a fake URL, it will trigger the except block
        response = requests.get(api_url, timeout=5)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        save_path = f"data/raw/products/{timestamp}_products.json"

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_json(save_path)
        logging.info(f"Successfully ingested API data to {save_path}")

    except Exception as e:
        logging.error(f"Failed to ingest API: {str(e)}")


if __name__ == "__main__":
    # Ensure the local sample file exists for testing
    # If the file doesn't exist, the logger will now tell you why!
    ingest_interactions('sample_interactions.csv')
