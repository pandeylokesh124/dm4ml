import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_absolute_error
import mlflow
import os
import mlflow.sklearn


def calculate_precision_recall_at_k(actual, predicted, k=5, threshold=3.5):
    """Calculate Precision and Recall at K based on a relevance threshold."""
    # Binary relevance: rating >= threshold
    act_relevant = {i for i, r in enumerate(actual) if r >= threshold}
    # Top K predicted items
    pred_top_k = set(np.argsort(predicted)[-k:])

    true_pos = len(act_relevant.intersection(pred_top_k))

    precision = true_pos / k
    recall = true_pos / len(act_relevant) if len(act_relevant) > 0 else 0
    return precision, recall


def train_recommender(data_path):
    # Load and prepare data
    df = pd.read_csv(data_path)
    # Create user-item matrix (users as rows, items as columns)
    user_item_matrix = df.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
    # Replace this with your ACTUAL project folder path from your screenshot
    # Note: Use forward slashes / even on Windows to avoid errors
    project_path = "C:/Users/hp/Desktop/BITS/Semester 2/Data Management for ML/Assignment"
    db_path = os.path.join(project_path, "mlflow.db")

    # This is the "Force" command
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")

    print(f"FORCING metadata to: {mlflow.get_tracking_uri()}")
    with mlflow.start_run():
        # 1. Train Model (SVD)
        svd = TruncatedSVD(n_components=min(10, user_item_matrix.shape[1] - 1), random_state=42)
        user_features = svd.fit_transform(user_item_matrix)

        # 2. Reconstruct matrix to get predicted ratings
        predicted_matrix = np.dot(user_features, svd.components_)

        # 3. Flatten for Evaluation (Compare only known ratings)
        actual_ratings = user_item_matrix.values[user_item_matrix.values > 0]
        predicted_ratings = predicted_matrix[user_item_matrix.values > 0]

        # 4. Calculate Metrics
        mae = mean_absolute_error(actual_ratings, predicted_ratings)
        precision, recall = calculate_precision_recall_at_k(actual_ratings, predicted_ratings, k=20)

        # 5. Log to MLflow (Note: It is best practice to log the raw floats to MLflow)
        mlflow.log_param("n_components", 10)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("Precision_at_5_pct", precision * 100)  # Logged as percentage
        mlflow.log_metric("Recall_at_5_pct", recall * 100)  # Logged as percentage
        mlflow.sklearn.log_model(svd, "svd_model")

        print(f"Model Training complete.")
        # Printing as percentages for your report
        print(f"MAE: {mae:.4f} (Avg. star rating error)")
        print(f"Precision@5: {precision * 100:.2f}%")
        print(f"Recall@5: {recall * 100:.2f}%")


if __name__ == "__main__":
    train_recommender('data/prepared/clean_interactions.csv')
