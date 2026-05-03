import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_absolute_error


class RecommenderTrainer:

    def __init__(self, project_path=None, tracking_uri=None):
        self.project_path = project_path or os.getcwd()

        # Priority: tracking_uri > project_path DB
        if tracking_uri:
            self.tracking_uri = tracking_uri
        else:
            db_path = os.path.join(self.project_path, "mlflow.db")
            self.tracking_uri = f"sqlite:///{db_path}"

        mlflow.set_tracking_uri(self.tracking_uri)
        print(f"FORCING metadata to: {mlflow.get_tracking_uri()}")

    # -------------------- METRICS --------------------
    def calculate_precision_recall_at_k(self, actual, predicted, k=5, threshold=3.5):
        act_relevant = {i for i, r in enumerate(actual) if r >= threshold}
        pred_top_k = set(np.argsort(predicted)[-k:])

        true_pos = len(act_relevant.intersection(pred_top_k))

        precision = true_pos / k
        recall = true_pos / len(act_relevant) if len(act_relevant) > 0 else 0

        return precision, recall

    # -------------------- TRAIN --------------------
    def train_recommender(self, data_path, n_components=10):
        # Load data
        df = pd.read_csv(data_path)

        # Create user-item matrix
        user_item_matrix = df.pivot(
            index='user_id',
            columns='item_id',
            values='rating'
        ).fillna(0)

        # Safe component handling
        n_components = min(n_components, user_item_matrix.shape[1] - 1)

        with mlflow.start_run():

            # 1. Train model
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            user_features = svd.fit_transform(user_item_matrix)

            # 2. Reconstruct predictions
            predicted_matrix = np.dot(user_features, svd.components_)

            # 3. Evaluate only known ratings
            mask = user_item_matrix.values > 0
            actual_ratings = user_item_matrix.values[mask]
            predicted_ratings = predicted_matrix[mask]

            # 4. Metrics
            mae = mean_absolute_error(actual_ratings, predicted_ratings)
            precision, recall = self.calculate_precision_recall_at_k(
                actual_ratings, predicted_ratings, k=5
            )

            # 5. MLflow logging
            mlflow.log_param("n_components", n_components)
            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("Precision_at_5_pct", precision * 100)
            mlflow.log_metric("Recall_at_5_pct", recall * 100)

            mlflow.sklearn.log_model(svd, "svd_model")

            # Output
            print("Model Training complete.")
            print(f"MAE: {mae:.4f} (Avg. star rating error)")
            print(f"Precision@5: {precision * 100:.2f}%")
            print(f"Recall@5: {recall * 100:.2f}%")

        return svd