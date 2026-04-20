"""
Central configuration for the Autonomous ML Pipeline Agent.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_DIR  = Path(__file__).parent.parent
UPLOADS_DIR  = PROJECT_DIR / "uploads"
JOBS_DIR     = PROJECT_DIR / "jobs"
RESULTS_DIR  = PROJECT_DIR / "results"

for _d in [UPLOADS_DIR, JOBS_DIR, RESULTS_DIR]:
    _d.mkdir(exist_ok=True)

# ── API Keys ───────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# ── Models ─────────────────────────────────────────────────────────────────────
AGENT_MODEL  = "claude-sonnet-4-6"   # agent reasoning — needs strong capability
REPORT_MODEL = "claude-sonnet-4-6"   # report narrative

# ── MLflow ─────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5002")
MLFLOW_EXPERIMENT   = "autonomous-ml-agent"

# ── Celery / Redis ─────────────────────────────────────────────────────────────
REDIS_URL            = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CELERY_BROKER_URL    = REDIS_URL
CELERY_RESULT_BACKEND = REDIS_URL

# ── ML ─────────────────────────────────────────────────────────────────────────
RANDOM_SEED    = 42
TEST_SIZE      = 0.2
IMBALANCE_THR  = 0.15   # minority class ratio below this triggers SMOTE
