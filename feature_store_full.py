import argparse
import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DB_PATH = "feature_store/feature_store_full.db"
DEFAULT_USER_FEATURES_PATH = "data/transformed/user_features.csv"
DEFAULT_ITEM_FEATURES_PATH = "data/transformed/item_features.csv"


class FeatureStoreFull:
    def __init__(self, db_path: str = DB_PATH):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._bootstrap()

    def _bootstrap(self):
        cur = self.conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS feature_registry (
                feature_name    TEXT NOT NULL,
                entity_type     TEXT NOT NULL,
                entity_key      TEXT NOT NULL,
                dtype           TEXT NOT NULL,
                source_table    TEXT NOT NULL,
                transformation  TEXT NOT NULL,
                registered_at   TEXT NOT NULL,
                PRIMARY KEY (feature_name, entity_type)
            );

            CREATE TABLE IF NOT EXISTS lineage_log (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                version         INTEGER NOT NULL,
                feature_view    TEXT NOT NULL,
                source_path     TEXT NOT NULL,
                source_hash     TEXT NOT NULL,
                row_count       INTEGER NOT NULL,
                materialized_at TEXT NOT NULL
            );
        """
        )
        self.conn.commit()

    def register_features(self):
        features = [
            ("total_interactions", "user", "user_id", "INTEGER", "transformed_user_features", "Precomputed total interactions per user"),
            ("avg_user_rating", "user", "user_id", "FLOAT", "transformed_user_features", "Precomputed average rating per user"),
            ("unique_items_rated", "user", "user_id", "INTEGER", "transformed_user_features", "Precomputed unique items rated per user"),
            ("max_user_rating", "user", "user_id", "FLOAT", "transformed_user_features", "Precomputed maximum rating per user"),
            ("min_user_rating", "user", "user_id", "FLOAT", "transformed_user_features", "Precomputed minimum rating per user"),
            ("popularity_score", "item", "item_id", "INTEGER", "transformed_item_features", "Precomputed interaction count per item"),
            ("avg_item_rating", "item", "item_id", "FLOAT", "transformed_item_features", "Precomputed average rating per item"),
            ("rating_count", "item", "item_id", "INTEGER", "transformed_item_features", "Precomputed rating count per item"),
            ("max_item_rating", "item", "item_id", "FLOAT", "transformed_item_features", "Precomputed maximum rating per item"),
            ("min_item_rating", "item", "item_id", "FLOAT", "transformed_item_features", "Precomputed minimum rating per item"),
        ]
        now = datetime.now(timezone.utc).isoformat()
        cur = self.conn.cursor()
        cur.executemany(
            """
            INSERT OR REPLACE INTO feature_registry
              (feature_name, entity_type, entity_key, dtype,
               source_table, transformation, registered_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            [(n, et, ek, dt, src, tf, now) for n, et, ek, dt, src, tf in features],
        )
        self.conn.commit()
        print(f"[Registry] Registered {len(features)} transformed features.")

    def _md5(self, path: str) -> str:
        h = hashlib.md5()
        with open(path, "rb") as f:
            h.update(f.read())
        return h.hexdigest()

    def materialize(
        self,
        user_features_path: str = DEFAULT_USER_FEATURES_PATH,
        item_features_path: str = DEFAULT_ITEM_FEATURES_PATH,
        version: int = 1,
    ):
        print(f"\n[Materialize] Loading transformed user features from: {user_features_path}")
        print(f"[Materialize] Loading transformed item features from: {item_features_path}")

        user_feats = pd.read_csv(user_features_path)
        item_feats = pd.read_csv(item_features_path)

        user_hash = self._md5(user_features_path)
        item_hash = self._md5(item_features_path)

        user_feats["feature_version"] = version
        user_feats["materialized_at"] = datetime.now(timezone.utc).isoformat()
        item_feats["feature_version"] = version
        item_feats["materialized_at"] = datetime.now(timezone.utc).isoformat()

        user_table = f"user_features_v{version}"
        item_table = f"item_features_v{version}"

        user_feats.to_sql(user_table, self.conn, if_exists="replace", index=False)
        item_feats.to_sql(item_table, self.conn, if_exists="replace", index=False)
        print(f"[Materialize] Written: {user_table} ({len(user_feats)} users)")
        print(f"[Materialize] Written: {item_table} ({len(item_feats)} items)")

        now = datetime.now(timezone.utc).isoformat()
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO lineage_log
              (version, feature_view, source_path, source_hash, row_count, materialized_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (version, "user_features", user_features_path, user_hash, len(user_feats), now),
        )
        cur.execute(
            """
            INSERT INTO lineage_log
              (version, feature_view, source_path, source_hash, row_count, materialized_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (version, "item_features", item_features_path, item_hash, len(item_feats), now),
        )
        self.conn.commit()
        print(f"[Lineage] Recorded materialization v{version} at {now}")

    def close(self):
        self.conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RecoMart Full Feature Store")
    parser.add_argument("--user-features", default=DEFAULT_USER_FEATURES_PATH)
    parser.add_argument("--item-features", default=DEFAULT_ITEM_FEATURES_PATH)
    parser.add_argument("--version", type=int, default=1)
    args = parser.parse_args()

    fs = FeatureStoreFull()
    fs.register_features()
    fs.materialize(
        user_features_path=args.user_features,
        item_features_path=args.item_features,
        version=args.version,
    )
    fs.close()
    print("\n[Done] Feature store is at:", DB_PATH)
