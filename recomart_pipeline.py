import os
import sys
import logging
from datetime import datetime


# -------------------- CONFIG --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "sample_interactions.csv")
CLEAN_DATA_PATH = os.path.join(BASE_DIR, "data", "prepared", "clean_interactions.csv")


# -------------------- LOGGER --------------------
def setup_logger():
    logger = logging.getLogger("recomart_pipeline")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)

    file_handler = logging.FileHandler("pipeline.log")
    file_handler.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(file_handler)

    return logger


logger = setup_logger()


# -------------------- TASKS --------------------
def run_data_ingestion():
    from ingest_data import DataIngestor

    logger.info("Starting ingestion...")
    ingestor = DataIngestor()

    output_path = ingestor.ingest_interactions(DATA_PATH)

    if not output_path:
        raise RuntimeError("Ingestion failed")

    logger.info(f"Ingestion complete: {output_path}")


def run_data_validation():
    from validate_data import DataValidator

    logger.info("Starting validation...")
    validator = DataValidator()

    df = validator.validate_and_clean(DATA_PATH)

    if df is None or df.empty:
        raise RuntimeError("Validation failed")

    logger.info(f"Validation complete: {len(df)} rows")


def run_preparation():
    from data_preparation_eda import DataPreparationPipeline

    logger.info("Starting preparation...")
    pipeline = DataPreparationPipeline(base_dir=BASE_DIR)

    df = pipeline.run()

    if df is None or df.empty:
        raise RuntimeError("Preparation failed")

    logger.info(f"Preparation complete: {len(df)} rows")


def run_feature_store():
    from feature_store import FeatureStore

    logger.info("Starting feature store...")
    fs = FeatureStore()

    fs.register_features()
    fs.materialize(data_path=CLEAN_DATA_PATH, version=1)
    fs.close()

    logger.info("Feature store complete")


def run_train_model():
    from train_model import RecommenderTrainer

    logger.info("Starting training...")

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    trainer = RecommenderTrainer(tracking_uri=tracking_uri)
    model = trainer.train_recommender(CLEAN_DATA_PATH)

    if model is None:
        raise RuntimeError("Training failed")

    logger.info("Training complete")


# -------------------- PIPELINE EXECUTION --------------------
PIPELINE_STEPS = [
    ("ingest", run_data_ingestion),
    ("validate", run_data_validation),
    ("prepare", run_preparation),
    ("feature_store", run_feature_store),
    ("train", run_train_model),
]


def run_pipeline(selected_steps=None):
    logger.info("=== Starting Recomart Pipeline ===")

    for step_name, step_func in PIPELINE_STEPS:
        if selected_steps and step_name not in selected_steps:
            continue

        logger.info(f"\n--- Running step: {step_name} ---")

        try:
            start_time = datetime.now()
            step_func()
            duration = (datetime.now() - start_time).total_seconds()

            logger.info(f"Step '{step_name}' completed in {duration:.2f}s")

        except Exception as e:
            logger.exception(f"Step '{step_name}' FAILED")
            logger.info("Pipeline stopped due to failure.")
            return

    logger.info("=== Pipeline completed successfully ===")


# -------------------- ENTRY POINT --------------------
if __name__ == "__main__":
    # Run full pipeline
    run_pipeline()

    # OR run specific steps:
    # run_pipeline(selected_steps=["ingest", "validate"])