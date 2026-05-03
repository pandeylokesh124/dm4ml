import pandas as pd
import requests
import logging
import os
import sys
from datetime import datetime


class DataIngestor:
    def __init__(self, log_file="ingestion.log"):
        self._setup_logging(log_file)

    def _setup_logging(self, log_file):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            force=True,
            handlers=[
                logging.FileHandler(log_file, mode='a'),
                logging.StreamHandler(sys.stdout)
            ]
        )

    def ingest_interactions(self, file_path, output_dir="data/raw/interactions"):
        try:
            if not os.path.exists(file_path):
                logging.error(f"File not found: {file_path}")
                return None

            df = pd.read_csv(file_path)

            save_path = os.path.join(output_dir, "sample_interactions.csv")

            os.makedirs(output_dir, exist_ok=True)
            df.to_csv(save_path, index=False)

            logging.info(f"Successfully ingested interactions to {save_path}")
            return save_path

        except Exception as e:
            logging.error(f"Failed to ingest CSV: {e}")
            return None

    def ingest_product_api(self, api_url, output_dir="data/raw/products"):
        try:
            response = requests.get(api_url, timeout=5)
            response.raise_for_status()

            data = response.json()
            df = pd.DataFrame(data)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            save_path = os.path.join(output_dir, f"{timestamp}_products.json")

            os.makedirs(output_dir, exist_ok=True)
            df.to_json(save_path)

            logging.info(f"Successfully ingested API data to {save_path}")
            return save_path

        except Exception as e:
            logging.error(f"Failed to ingest API: {str(e)}")
            return None