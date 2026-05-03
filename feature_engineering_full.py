import json
import logging
import os
from pathlib import Path

import pandas as pd

INTERACTIONS_PATH = "data/prepared/clean_interactions.csv"
DEFAULT_PRODUCTS_PATH = "data/raw/products/latest.json"
FALLBACK_PRODUCTS_PATH = "sample_products.json"
OUTPUT_DIR = Path("data/transformed")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,
)


def _load_products(products_path: str) -> pd.DataFrame:
    if not os.path.exists(products_path):
        logging.warning("Product metadata not found at %s", products_path)
        if os.path.exists(FALLBACK_PRODUCTS_PATH):
            logging.info("Falling back to %s", FALLBACK_PRODUCTS_PATH)
            products_path = FALLBACK_PRODUCTS_PATH
        else:
            logging.warning("No product metadata available; using empty product feature frame.")
            return pd.DataFrame(columns=["product_id", "category", "price", "stock_status"])

    with open(products_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    df = pd.DataFrame(data)
    expected_columns = ["product_id", "category", "price", "stock_status"]
    for column in expected_columns:
        if column not in df.columns:
            df[column] = pd.NA
    return df[expected_columns]


def engineer_features(
    interactions_path: str = INTERACTIONS_PATH,
    products_path: str = DEFAULT_PRODUCTS_PATH,
):
    logging.info("Loading cleaned interactions from %s", interactions_path)
    interactions = pd.read_csv(interactions_path)
    products = _load_products(products_path)

    interactions["timestamp"] = pd.to_datetime(
        interactions["timestamp"],
        dayfirst=True,
        errors="coerce",
    )
    interactions["rating"] = pd.to_numeric(interactions["rating"], errors="coerce")

    transformed = interactions.copy()
    transformed["interaction_date"] = transformed["timestamp"].dt.date.astype("string")
    transformed["interaction_hour"] = transformed["timestamp"].dt.hour.fillna(-1).astype(int)
    transformed["interaction_dayofweek"] = transformed["timestamp"].dt.day_name().fillna("Unknown")
    transformed["is_weekend"] = transformed["timestamp"].dt.dayofweek.ge(5).fillna(False).astype(int)

    user_features = transformed.groupby("user_id").agg(
        total_interactions=("item_id", "count"),
        avg_user_rating=("rating", "mean"),
        unique_items_rated=("item_id", "nunique"),
        max_user_rating=("rating", "max"),
        min_user_rating=("rating", "min"),
    ).reset_index()
    user_features["avg_user_rating"] = user_features["avg_user_rating"].round(4)

    item_features = transformed.groupby("item_id").agg(
        popularity_score=("user_id", "count"),
        avg_item_rating=("rating", "mean"),
        rating_count=("rating", "count"),
        max_item_rating=("rating", "max"),
        min_item_rating=("rating", "min"),
    ).reset_index()
    item_features["avg_item_rating"] = item_features["avg_item_rating"].round(4)

    prepared_products = products.copy()
    prepared_products["price"] = pd.to_numeric(prepared_products["price"], errors="coerce")
    prepared_products["in_stock_flag"] = (
        prepared_products["stock_status"].fillna("").str.lower().eq("in stock").astype(int)
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    transformed_path = OUTPUT_DIR / "interaction_features.csv"
    user_features_path = OUTPUT_DIR / "user_features.csv"
    item_features_path = OUTPUT_DIR / "item_features.csv"
    product_features_path = OUTPUT_DIR / "product_features.csv"

    transformed.to_csv(transformed_path, index=False)
    user_features.to_csv(user_features_path, index=False)
    item_features.to_csv(item_features_path, index=False)
    prepared_products.to_csv(product_features_path, index=False)

    logging.info("Saved interaction features to %s", transformed_path)
    logging.info("Saved user features to %s", user_features_path)
    logging.info("Saved item features to %s", item_features_path)
    logging.info("Saved product features to %s", product_features_path)

    return {
        "interaction_features": str(transformed_path),
        "user_features": str(user_features_path),
        "item_features": str(item_features_path),
        "product_features": str(product_features_path),
    }


if __name__ == "__main__":
    engineer_features()
