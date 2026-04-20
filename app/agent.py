"""
Autonomous ML Pipeline Agent.

Uses LangGraph's ReAct agent with Claude Sonnet as the reasoning engine.
The agent is given a job_id and target column, then autonomously:
  1. Loads the dataset
  2. Runs EDA
  3. Trains multiple models (with SMOTE if imbalanced)
  4. Evaluates and compares models
  5. Logs best model to MLflow
  6. Generates the final report
"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent

from app.config import AGENT_MODEL, ANTHROPIC_API_KEY, JOBS_DIR
from app.tools import ALL_TOOLS

SYSTEM_PROMPT = """\
You are an autonomous ML pipeline agent. Your job is to fully automate the \
machine learning workflow for a given dataset.

You have access to these tools:
- load_dataset: Load and summarize the dataset. Always call this first.
- run_eda: Run exploratory data analysis. Call after load_dataset.
- train_model: Train a ML model. Available models: logistic_regression, \
random_forest, xgboost (and ridge for regression).
- evaluate_model: Evaluate a trained model. Call after each train_model.
- log_to_mlflow: Log the best model to MLflow. Call once after all evaluations.
- generate_report: Generate the final HTML report. Call this last.

WORKFLOW — follow this order strictly:
1. Call load_dataset to understand the data
2. Call run_eda for deeper analysis
3. Decide which models to train based on problem type and imbalance:
   - Always train at least: logistic_regression and random_forest
   - If XGBoost makes sense (imbalanced data, complex patterns), train xgboost too
   - Use use_smote=True if class imbalance was detected
4. Call evaluate_model for EACH model you trained
5. Compare results, identify the best model
6. Call log_to_mlflow with the best model name
7. Call generate_report as the final step

Be systematic. After each tool call, reason about what you learned and \
what to do next. Always complete the full workflow — do not stop early.
"""


def run_agent(job_id: str, target_column: str = "auto") -> dict:
    """
    Run the autonomous ML agent for a given job.

    Args:
        job_id:        The job identifier (dataset must be in uploads/{job_id}/).
        target_column: The target column name, or 'auto' to detect.

    Returns:
        dict with 'status', 'messages', and 'final_output'.
    """
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(exist_ok=True)

    # Save job status
    _update_status(job_id, "running", "Agent started")

    try:
        llm = ChatAnthropic(
            model=AGENT_MODEL,
            api_key=ANTHROPIC_API_KEY,
            max_tokens=4096,
        )

        agent = create_react_agent(llm, ALL_TOOLS, prompt=SYSTEM_PROMPT)

        user_message = (
            f"Please run the full ML pipeline for job_id='{job_id}' "
            f"with target_column='{target_column}'. "
            "Follow all steps: load, EDA, train multiple models, evaluate, "
            "log to MLflow, generate report."
        )

        # Collect agent messages for logging
        log_messages = []
        final_output = ""

        result = agent.invoke({"messages": [("user", user_message)]})

        for msg in result["messages"]:
            role    = getattr(msg, "type", "unknown")
            content = msg.content if isinstance(msg.content, str) \
                      else json.dumps(msg.content)
            log_messages.append({"role": role, "content": content})
            if role == "ai":
                final_output = content

        # Save log
        with open(job_dir / "agent_log.json", "w") as f:
            json.dump(log_messages, f, indent=2)

        _update_status(job_id, "complete", "Pipeline finished successfully")
        return {
            "status":       "complete",
            "messages":     log_messages,
            "final_output": final_output,
        }

    except Exception as e:
        _update_status(job_id, "failed", str(e))
        return {"status": "failed", "error": str(e)}


def _update_status(job_id: str, status: str, message: str) -> None:
    """Write job status to a JSON file for the API to poll."""
    status_path = JOBS_DIR / job_id / "status.json"
    status_path.parent.mkdir(exist_ok=True)
    with open(status_path, "w") as f:
        json.dump({"status": status, "message": message}, f)
