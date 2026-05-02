import sys
from datetime import datetime
from unittest.mock import MagicMock

# 1. FIX: Mocking 'pwd' to stop the Windows POSIX error
if sys.platform == "win32":
    sys.modules["pwd"] = MagicMock()

try:
    from airflow import DAG
    # 2. FIX: Using the new recommended import path to stop the DeprecationWarning
    from airflow.providers.standard.operators.python import PythonOperator
except (ImportError, ModuleNotFoundError):
    # Fallback for older versions or if imports fail on Windows
    from airflow.models import DAG
    from airflow.operators.python import PythonOperator


def dummy_task():
    print("Task executed!")


def run_feature_store():
    from feature_store import FeatureStore
    fs = FeatureStore()
    fs.register_features()
    fs.materialize(data_path='data/prepared/clean_interactions.csv', version=1)
    fs.close()


# 3. FIX: Changed 'schedule_interval' to 'schedule' (Airflow 2.0+ standard)
with DAG(
        dag_id='recomart_pipeline',
        start_date=datetime(2023, 1, 1),
        schedule='@daily',  # Fixed keyword argument
        catchup=False
) as dag:
    task_ingest = PythonOperator(
        task_id='ingest',
        python_callable=dummy_task
    )

    task_validate = PythonOperator(
        task_id='validate',
        python_callable=dummy_task
    )

    task_feature_store = PythonOperator(
        task_id='feature_store',
        python_callable=run_feature_store
    )

    task_train = PythonOperator(
        task_id='train',
        python_callable=dummy_task
    )

    task_ingest >> task_validate >> task_feature_store >> task_train
