"""
FastAPI backend for the Autonomous ML Pipeline Agent.

Endpoints:
  POST /jobs            — upload CSV + start agent job
  GET  /jobs/{job_id}   — get job status
  GET  /jobs/{job_id}/metrics  — get model metrics
  GET  /jobs/{job_id}/report   — view HTML report
  GET  /health          — health check
"""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from app.config import JOBS_DIR, UPLOADS_DIR

app = FastAPI(
    title="Autonomous ML Pipeline Agent",
    description=(
        "An AI agent (Claude Sonnet + LangGraph) that autonomously runs EDA, "
        "selects models, trains, evaluates, and generates reports — "
        "given any CSV dataset."
    ),
    version="1.0.0",
)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@app.get("/health")
def health():
    return {"status": "ok", "service": "autonomous-ml-agent"}


@app.post("/jobs")
async def create_job(
    file: UploadFile = File(...),
    target_column: str = Form(default="auto"),
):
    """
    Upload a CSV dataset and start the autonomous ML pipeline.

    - **file**: CSV file to analyse.
    - **target_column**: Name of the target column (leave blank to auto-detect).

    Returns job_id to poll for status.
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    job_id  = str(uuid.uuid4())[:8]
    job_upload_dir = UPLOADS_DIR / job_id
    job_upload_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded file
    dest = job_upload_dir / "dataset.csv"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Enqueue Celery task
    from app.tasks import run_pipeline_task
    task = run_pipeline_task.delay(job_id, target_column or "auto")

    # Save initial status
    status_path = JOBS_DIR / job_id / "status.json"
    (JOBS_DIR / job_id).mkdir(exist_ok=True)
    with open(status_path, "w") as f:
        json.dump({
            "status":        "queued",
            "message":       "Job queued, waiting for worker.",
            "celery_task_id": task.id,
            "target_column": target_column or "auto",
        }, f)

    return {
        "job_id":         job_id,
        "celery_task_id": task.id,
        "status":         "queued",
        "message":        "Job created. Poll GET /jobs/{job_id} for status.",
    }


@app.get("/jobs/{job_id}")
def get_job_status(job_id: str):
    """Get the current status of a pipeline job."""
    status = _read_json(JOBS_DIR / job_id / "status.json")
    if not status:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return status


@app.get("/jobs/{job_id}/metrics")
def get_job_metrics(job_id: str):
    """Get model evaluation metrics for a completed job."""
    metrics = _read_json(JOBS_DIR / job_id / "metrics.json")
    if not metrics:
        raise HTTPException(
            status_code=404,
            detail="No metrics yet. Job may still be running."
        )
    return metrics


@app.get("/jobs/{job_id}/report", response_class=HTMLResponse)
def get_job_report(job_id: str):
    """View the HTML evaluation report for a completed job."""
    report_path = JOBS_DIR / job_id / "report.html"
    if not report_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No report yet. Job may still be running."
        )
    return HTMLResponse(content=report_path.read_text(encoding="utf-8"))


@app.get("/jobs")
def list_jobs():
    """List all jobs and their statuses."""
    jobs = []
    for job_dir in sorted(JOBS_DIR.iterdir()):
        if job_dir.is_dir():
            status = _read_json(job_dir / "status.json")
            meta   = _read_json(job_dir / "metadata.json")
            jobs.append({
                "job_id":       job_dir.name,
                "status":       status.get("status", "unknown"),
                "problem_type": meta.get("problem_type", "unknown"),
                "target":       meta.get("target_column", "unknown"),
                "rows":         meta.get("shape", [0])[0],
            })
    return {"jobs": jobs}
