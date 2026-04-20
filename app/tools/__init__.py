from app.tools.loader import load_dataset
from app.tools.eda import run_eda
from app.tools.trainer import train_model
from app.tools.evaluator import evaluate_model
from app.tools.mlflow_logger import log_to_mlflow
from app.tools.reporter import generate_report

ALL_TOOLS = [
    load_dataset,
    run_eda,
    train_model,
    evaluate_model,
    log_to_mlflow,
    generate_report,
]
