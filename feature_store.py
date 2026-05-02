"""
RecoMart Custom Feature Store
==============================
A lightweight feature store built on SQLite that provides:
  - Feature registration with metadata (source, version, description, entity type)
  - Versioned feature materialization from clean_interactions.csv
  - Point-in-time correct retrieval for both training and inference
  - Full lineage tracking (every write records its source + transformation)

Architecture
------------
  feature_store.db  (SQLite)
    ├── feature_registry   — metadata for every registered feature
    ├── feature_store_v*   — one table per feature set version (user / item)
    └── lineage_log        — append-only audit trail of every materialization

Usage
-----
  python feature_store.py               # materialize + demo retrieval
  python feature_store.py --version 2   # materialize as v2
"""

import sqlite3
import pandas as pd
import json
import hashlib
import argparse
from datetime import datetime, timezone
from pathlib import Path

# ─── Config ──────────────────────────────────────────────────────────────────

DB_PATH = "feature_store/feature_store.db"
DEFAULT_DATA_PATH = "data/prepared/clean_interactions.csv"


# ─── Core FeatureStore class ──────────────────────────────────────────────────

class FeatureStore:
    """
    A file-backed feature store using SQLite as the offline store.

    Concepts:
      - Feature View  : a logical group of features sharing an entity key
                        (e.g. 'user_features', 'item_features')
      - Version       : integer that increments each time features are
                        re-materialised from a new data snapshot
      - Registry      : a catalog row per feature with name, dtype, source,
                        transformation logic, and entity type
      - Lineage Log   : an append-only record of every materialization event,
                        including data hash, row count, and timestamp
    """

    def __init__(self, db_path: str = DB_PATH):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._bootstrap()

    # ── Schema bootstrap ──────────────────────────────────────────────────────

    def _bootstrap(self):
        cur = self.conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS feature_registry (
                feature_name    TEXT NOT NULL,
                entity_type     TEXT NOT NULL,        -- 'user' | 'item'
                entity_key      TEXT NOT NULL,        -- column that is the PK
                dtype           TEXT NOT NULL,
                source_table    TEXT NOT NULL,        -- logical origin
                transformation  TEXT NOT NULL,        -- plain-English description
                registered_at   TEXT NOT NULL,
                PRIMARY KEY (feature_name, entity_type)
            );

            CREATE TABLE IF NOT EXISTS lineage_log (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                version         INTEGER NOT NULL,
                feature_view    TEXT NOT NULL,        -- 'user_features' | 'item_features'
                source_path     TEXT NOT NULL,
                source_hash     TEXT NOT NULL,        -- MD5 of source file
                row_count       INTEGER NOT NULL,
                materialized_at TEXT NOT NULL
            );
        """)
        self.conn.commit()

    # ── Registry ──────────────────────────────────────────────────────────────

    def register_features(self):
        """Declare all features and their metadata in the registry."""
        features = [
            # User features
            ("total_interactions",  "user", "user_id", "INTEGER",
             "raw_interactions", "COUNT(interaction_id) per user_id"),
            ("avg_user_rating",     "user", "user_id", "FLOAT",
             "raw_interactions", "AVG(rating) per user_id"),
            ("unique_items_rated",  "user", "user_id", "INTEGER",
             "raw_interactions", "COUNT(DISTINCT item_id) per user_id"),
            ("max_user_rating",     "user", "user_id", "FLOAT",
             "raw_interactions", "MAX(rating) per user_id"),
            ("min_user_rating",     "user", "user_id", "FLOAT",
             "raw_interactions", "MIN(rating) per user_id"),
            ("user_rating_stddev",  "user", "user_id", "FLOAT",
             "raw_interactions", "STD(rating) per user_id — measures taste consistency"),
            # Item features
            ("popularity_score",    "item", "item_id", "INTEGER",
             "raw_interactions", "COUNT(user_id) per item_id — number of raters"),
            ("avg_item_rating",     "item", "item_id", "FLOAT",
             "raw_interactions", "AVG(rating) per item_id"),
            ("rating_count",        "item", "item_id", "INTEGER",
             "raw_interactions", "COUNT(rating) per item_id"),
            ("max_item_rating",     "item", "item_id", "FLOAT",
             "raw_interactions", "MAX(rating) per item_id"),
            ("min_item_rating",     "item", "item_id", "FLOAT",
             "raw_interactions", "MIN(rating) per item_id"),
            ("item_rating_stddev",  "item", "item_id", "FLOAT",
             "raw_interactions", "STD(rating) per item_id — measures rating consistency"),
        ]
        now = datetime.now(timezone.utc).isoformat()
        cur = self.conn.cursor()
        cur.executemany("""
            INSERT OR REPLACE INTO feature_registry
              (feature_name, entity_type, entity_key, dtype,
               source_table, transformation, registered_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [(n, et, ek, dt, src, tf, now) for n, et, ek, dt, src, tf in features])
        self.conn.commit()
        print(f"[Registry] Registered {len(features)} features.")

    def list_features(self) -> pd.DataFrame:
        return pd.read_sql("SELECT * FROM feature_registry ORDER BY entity_type, feature_name",
                           self.conn)

    # ── Materialisation ───────────────────────────────────────────────────────

    def _md5(self, path: str) -> str:
        h = hashlib.md5()
        with open(path, "rb") as f:
            h.update(f.read())
        return h.hexdigest()

    def materialize(self, data_path: str = DEFAULT_DATA_PATH, version: int = 1):
        """
        Read clean_interactions.csv, compute all features, write two versioned
        tables (user_features_v<N> and item_features_v<N>) and record lineage.
        """
        print(f"\n[Materialize] Loading data from: {data_path}")
        df = pd.read_csv(data_path)
        src_hash = self._md5(data_path)
        print(f"[Materialize] {len(df)} rows | MD5: {src_hash}")

        # ── User features ──────────────────────────────────────────────────
        user_feats = df.groupby("user_id").agg(
            total_interactions=("item_id", "count"),
            avg_user_rating=("rating", "mean"),
            unique_items_rated=("item_id", "nunique"),
            max_user_rating=("rating", "max"),
            min_user_rating=("rating", "min"),
            user_rating_stddev=("rating", "std"),
        ).reset_index()
        user_feats["user_rating_stddev"] = user_feats["user_rating_stddev"].fillna(0)
        user_feats["avg_user_rating"] = user_feats["avg_user_rating"].round(4)
        user_feats["user_rating_stddev"] = user_feats["user_rating_stddev"].round(4)
        user_feats["feature_version"] = version
        user_feats["materialized_at"] = datetime.now(timezone.utc).isoformat()

        # ── Item features ──────────────────────────────────────────────────
        item_feats = df.groupby("item_id").agg(
            popularity_score=("user_id", "count"),
            avg_item_rating=("rating", "mean"),
            rating_count=("rating", "count"),
            max_item_rating=("rating", "max"),
            min_item_rating=("rating", "min"),
            item_rating_stddev=("rating", "std"),
        ).reset_index()
        item_feats["item_rating_stddev"] = item_feats["item_rating_stddev"].fillna(0)
        item_feats["avg_item_rating"] = item_feats["avg_item_rating"].round(4)
        item_feats["item_rating_stddev"] = item_feats["item_rating_stddev"].round(4)
        item_feats["feature_version"] = version
        item_feats["materialized_at"] = datetime.now(timezone.utc).isoformat()

        # ── Write versioned tables ─────────────────────────────────────────
        user_table = f"user_features_v{version}"
        item_table = f"item_features_v{version}"

        user_feats.to_sql(user_table, self.conn, if_exists="replace", index=False)
        item_feats.to_sql(item_table, self.conn, if_exists="replace", index=False)
        print(f"[Materialize] Written: {user_table} ({len(user_feats)} users)")
        print(f"[Materialize] Written: {item_table} ({len(item_feats)} items)")

        # ── Lineage log ────────────────────────────────────────────────────
        now = datetime.now(timezone.utc).isoformat()
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO lineage_log
              (version, feature_view, source_path, source_hash, row_count, materialized_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (version, "user_features", data_path, src_hash, len(user_feats), now))
        cur.execute("""
            INSERT INTO lineage_log
              (version, feature_view, source_path, source_hash, row_count, materialized_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (version, "item_features", data_path, src_hash, len(item_feats), now))
        self.conn.commit()
        print(f"[Lineage]    Recorded materialization v{version} at {now}")

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def get_user_features(self, user_ids: list, version: int = 1) -> pd.DataFrame:
        """
        Retrieve user features for a list of user_ids at a specific version.
        Intended for both training (bulk) and inference (single entity).
        """
        table = f"user_features_v{version}"
        placeholders = ",".join("?" * len(user_ids))
        query = f"SELECT * FROM {table} WHERE user_id IN ({placeholders})"
        return pd.read_sql(query, self.conn, params=user_ids)

    def get_item_features(self, item_ids: list, version: int = 1) -> pd.DataFrame:
        """Retrieve item features for a list of item_ids at a specific version."""
        table = f"item_features_v{version}"
        placeholders = ",".join("?" * len(item_ids))
        query = f"SELECT * FROM {table} WHERE item_id IN ({placeholders})"
        return pd.read_sql(query, self.conn, params=item_ids)

    def get_training_dataset(self, version: int = 1) -> dict:
        """
        Return the full user and item feature tables for model training.
        Returns a dict with keys 'user_features' and 'item_features'.
        """
        user_df = pd.read_sql(f"SELECT * FROM user_features_v{version}", self.conn)
        item_df = pd.read_sql(f"SELECT * FROM item_features_v{version}", self.conn)
        return {"user_features": user_df, "item_features": item_df}

    def get_lineage(self) -> pd.DataFrame:
        """Return the full lineage log."""
        return pd.read_sql("SELECT * FROM lineage_log ORDER BY id", self.conn)

    def list_versions(self) -> list:
        """Return all available feature versions from the lineage log."""
        cur = self.conn.cursor()
        cur.execute("SELECT DISTINCT version FROM lineage_log ORDER BY version")
        return [r[0] for r in cur.fetchall()]

    def close(self):
        self.conn.close()


# ─── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RecoMart Feature Store")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH,
                        help="Path to clean_interactions.csv")
    parser.add_argument("--version", type=int, default=1,
                        help="Feature set version to materialize (default: 1)")
    args = parser.parse_args()

    fs = FeatureStore()

    # 1. Register feature definitions
    print("\n" + "="*60)
    print("STEP 1: Registering features in registry")
    print("="*60)
    fs.register_features()

    # 2. Materialize features from source data
    print("\n" + "="*60)
    print(f"STEP 2: Materializing features (version={args.version})")
    print("="*60)
    fs.materialize(data_path=args.data, version=args.version)

    # 3. Demo: point-in-time retrieval for training
    print("\n" + "="*60)
    print("STEP 3: Training dataset retrieval demo")
    print("="*60)
    training_data = fs.get_training_dataset(version=args.version)
    print(f"\nUser features ({len(training_data['user_features'])} users):")
    print(training_data["user_features"].head(5).to_string(index=False))
    print(f"\nItem features ({len(training_data['item_features'])} items):")
    print(training_data["item_features"].head(5).to_string(index=False))

    # 4. Demo: inference retrieval (single entity lookup)
    print("\n" + "="*60)
    print("STEP 4: Inference retrieval demo (single entity lookup)")
    print("="*60)
    sample_users = pd.read_csv(args.data)["user_id"].unique()[:3].tolist()
    sample_items = pd.read_csv(args.data)["item_id"].unique()[:3].tolist()

    print(f"\nFetching features for user_ids: {sample_users}")
    print(fs.get_user_features(sample_users, version=args.version).to_string(index=False))

    print(f"\nFetching features for item_ids: {sample_items}")
    print(fs.get_item_features(sample_items, version=args.version).to_string(index=False))

    # 5. Show registry
    print("\n" + "="*60)
    print("STEP 5: Feature Registry")
    print("="*60)
    registry = fs.list_features()
    print(registry[["feature_name", "entity_type", "entity_key",
                     "dtype", "transformation"]].to_string(index=False))

    # 6. Show lineage
    print("\n" + "="*60)
    print("STEP 6: Lineage Log")
    print("="*60)
    print(fs.get_lineage().to_string(index=False))

    # 7. Show available versions
    print(f"\nAvailable versions: {fs.list_versions()}")

    fs.close()
    print("\n[Done] Feature store is at:", DB_PATH)