import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class DataPreparationPipeline:

    def __init__(self, base_dir=None):
        self.base_dir = base_dir or os.getcwd()

        self.plots_dir = os.path.join(self.base_dir, 'plots')
        self.interactions_path = os.path.join(self.base_dir, 'sample_interactions.csv')
        self.products_path = os.path.join(self.base_dir, 'sample_products.json')
        self.prepared_path = os.path.join(self.base_dir, 'data', 'prepared', 'prepared_interactions.csv')

        os.makedirs(self.plots_dir, exist_ok=True)

    # -------------------- LOAD --------------------
    def load_data(self):
        interactions = pd.read_csv(self.interactions_path, skipinitialspace=True)
        with open(self.products_path, 'r', encoding='utf-8') as f:
            products = pd.read_json(f)
        return interactions, products

    # -------------------- CLEAN --------------------
    def clean_interactions(self, df):
        df = df.copy()
        df.columns = df.columns.str.strip()
        df = df.dropna(how='all')

        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', dayfirst=True)

        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

        missing_values = df.isnull().sum().to_dict()

        if missing_values.get('rating', 0) > 0:
            df['rating'] = df['rating'].fillna(df['rating'].mean())

        if 'timestamp' in df.columns and missing_values.get('timestamp', 0) > 0:
            df['timestamp'] = df['timestamp'].fillna(method='ffill').fillna(method='bfill')

        duplicate_count = df.duplicated(subset=['user_id', 'item_id']).sum()
        if duplicate_count > 0:
            df = df.drop_duplicates(subset=['user_id', 'item_id'], keep='last')

        return df

    # -------------------- MERGE --------------------
    def merge_with_product_features(self, interactions, products):
        products = products.rename(columns={'product_id': 'item_id'})
        merged = interactions.merge(products, on='item_id', how='left')

        merged['category'] = merged['category'].fillna('Unknown')
        merged['price'] = pd.to_numeric(merged['price'], errors='coerce').fillna(merged['price'].mean())
        merged['stock'] = pd.to_numeric(merged['stock'], errors='coerce').fillna(0).astype(int)

        return merged

    # -------------------- FEATURE ENGINEERING --------------------
    def encode_categorical_columns(self, df, categorical_columns=None):
        df = df.copy()

        if categorical_columns is None:
            categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

        encoded_features = {}
        for column in categorical_columns:
            if column in df.columns:
                df[f'{column}_encoded'] = pd.factorize(df[column])[0]
                encoded_features[column] = df[f'{column}_encoded'].nunique()

        return df, encoded_features

    def normalize_numerical_columns(self, df, numeric_columns=None):
        df = df.copy()

        if numeric_columns is None:
            numeric_columns = df.select_dtypes(include=['number']).columns.tolist()

        normalized_columns = {}
        for column in numeric_columns:
            values = df[column].astype(float)
            min_val = values.min()
            max_val = values.max()

            if pd.isna(min_val) or pd.isna(max_val) or min_val == max_val:
                df[f'{column}_normalized'] = 0.0
            else:
                df[f'{column}_normalized'] = (values - min_val) / (max_val - min_val)

            normalized_columns[column] = (min_val, max_val)

        return df, normalized_columns

    def prepare_timestamp_features(self, df):
        df = df.copy()

        if 'timestamp' in df.columns:
            df['interaction_hour'] = df['timestamp'].dt.hour
            df['interaction_dayofweek'] = df['timestamp'].dt.day_name()
            df['timestamp_numeric'] = df['timestamp'].astype('int64') // 10**9

        return df

    # -------------------- PLOTTING --------------------
    def plot_histogram(self, df, column, output_path, title=None):
        plt.figure(figsize=(8, 5))
        sns.histplot(df[column].dropna(), kde=False, bins=10)
        plt.title(title or f'Distribution of {column}')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def plot_bar(self, df, column, output_path, title=None, top_n=20):
        counts = df[column].value_counts().head(top_n)

        plt.figure(figsize=(10, 6))
        plt.barh(counts.index.astype(str), counts.values)
        plt.gca().invert_yaxis()
        plt.title(title or f'{column} Popularity')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def plot_sparsity_heatmap(self, df, output_path, max_users=40, max_items=40):
        pivot = df.pivot(index='user_id', columns='item_id', values='rating').fillna(0)

        if pivot.shape[0] > max_users or pivot.shape[1] > max_items:
            pivot = pivot.iloc[:max_users, :max_items]

        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot != 0, cbar=False)
        plt.title('Interaction Sparsity Pattern')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    # -------------------- SAVE --------------------
    def save_prepared_dataset(self, df):
        os.makedirs(os.path.dirname(self.prepared_path), exist_ok=True)
        df.to_csv(self.prepared_path, index=False)

    # -------------------- MAIN PIPELINE --------------------
    def run(self):
        interactions, products = self.load_data()

        interactions = self.clean_interactions(interactions)
        merged = self.merge_with_product_features(interactions, products)
        merged = self.prepare_timestamp_features(merged)

        merged, encoding_info = self.encode_categorical_columns(merged, ['category'])
        merged, normalization_info = self.normalize_numerical_columns(
            merged, ['price', 'timestamp_numeric']
        )

        print('--- Data Preparation Summary ---')
        print(f'Raw rows: {len(interactions)}')
        print(f'Unique users: {interactions["user_id"].nunique()}')
        print(f'Unique items: {interactions["item_id"].nunique()}')
        print(f'Encoding info: {encoding_info}')
        print(f'Normalization ranges: {normalization_info}')

        self.save_prepared_dataset(merged)

        # plots
        self.plot_histogram(merged, 'rating', os.path.join(self.plots_dir, 'rating_distribution.png'))
        self.plot_bar(merged, 'item_id', os.path.join(self.plots_dir, 'item_popularity.png'))
        self.plot_bar(merged, 'category', os.path.join(self.plots_dir, 'category_distribution.png'))
        self.plot_histogram(merged, 'price', os.path.join(self.plots_dir, 'price_distribution.png'))
        self.plot_sparsity_heatmap(merged, os.path.join(self.plots_dir, 'interaction_sparsity.png'))

        print(f'Prepared dataset saved to: {self.prepared_path}')
        print(f'Plots saved to: {self.plots_dir}')

        return merged