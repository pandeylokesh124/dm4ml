# Feature Engineering and Feature Store

## 1. SQL Schema
- `features.sql` defines feature creation SQL for:
  - `user_features` with `total_interactions`, `avg_user_rating`
  - `item_features` with `popularity_score`, `avg_product_rating`

## 2. Transformation Scripts
- `feature_engineering_full.py` generates transformed feature tables and saves output to `data/transformed/`:
  - `interaction_features.csv`
  - `user_features.csv`
  - `item_features.csv`
  - `product_features.csv`
- It loads cleaned interaction data from `data/prepared/clean_interactions.csv` and product metadata from `data/raw/products/latest.json`, falling back to `sample_products.json`.

## 3. Feature Logic Summary
- User activity frequency:
  - `total_interactions` = count of item interactions per user
  - `unique_items_rated` = unique items rated per user
- Rating behavior:
  - `avg_user_rating`, `max_user_rating`, `min_user_rating`
  - `avg_item_rating`, `max_item_rating`, `min_item_rating`
- Item popularity:
  - `popularity_score` = count of distinct users per item
  - `rating_count` = count of ratings per item
- Product metadata features:
  - `price`, `in_stock_flag`

## 4. Feature Store Code
- `feature_store.py` contains a custom SQLite-backed feature store with:
  - `feature_registry` metadata table
  - `lineage_log` audit trail
  - `user_features_v<N>` and `item_features_v<N>` versioned tables
- `feature_store_full.py` materializes transformed features from `data/transformed/` and supports versioned retrieval.

## 5. Feature Metadata Documentation
- Registered features include:
  - feature name
  - entity type (`user` or `item`)
  - entity key (`user_id` or `item_id`)
  - source table
  - transformation description

## 6. Sample Commands
```bash
python feature_engineering_full.py
python feature_store_full.py --version 1
python feature_store.py --data data/prepared/clean_interactions.csv --version 1
```

## 7. Storage
- Transformed CSV output: `data/transformed/`
- SQLite feature stores:
  - `feature_store/feature_store_full.db`
  - `feature_store/feature_store.db`
