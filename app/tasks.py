"""
Celery task definitions.
The worker picks up run_pipeline_task and executes the agent asynchronously.
"""

from celery import Celery
from app.config import CELERY_BROKER_URL, CELERY_RESULT_BACKEND

celery_app = Celery(
    "autonomous_ml_agent",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
)


@celery_app.task(bind=True, name="run_pipeline_task")
def run_pipeline_task(self, job_id: str, target_column: str = "auto") -> dict:
    """
    Celery task — runs the full autonomous ML agent pipeline.
    Executed asynchronously by the Celery worker.
    """
    from app.agent import run_agent
    return run_agent(job_id=job_id, target_column=target_column)
