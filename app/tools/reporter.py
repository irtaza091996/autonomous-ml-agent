"""
Tool: generate_report
Uses Claude Sonnet to write a narrative report, then renders full HTML.
"""

from __future__ import annotations

import json
from pathlib import Path

import anthropic
from langchain_core.tools import tool

from app.config import ANTHROPIC_API_KEY, JOBS_DIR, REPORT_MODEL

REPORT_PROMPT = """\
You are an expert ML engineer and data scientist writing a portfolio evaluation report.

Below are the results of an autonomous ML pipeline run on a dataset.

DATASET INFO:
{metadata}

EDA FINDINGS:
{eda_findings}

MODEL RESULTS:
{model_metrics}

Write a concise, professional report with these sections:
1. Executive Summary (2-3 sentences — what was the task, what's the headline result)
2. Dataset Overview (key characteristics, any data quality issues)
3. EDA Key Findings (what patterns were discovered)
4. Model Comparison (compare all models trained, explain why the best one won)
5. Feature Analysis (which features drive predictions and why)
6. Limitations & Next Steps (honest assessment + improvements)

Be specific with numbers. Write in plain English suitable for a technical portfolio.
"""


@tool
def generate_report(job_id: str) -> str:
    """
    Generate a comprehensive HTML evaluation report using Claude Sonnet.
    Combines all metrics, EDA findings, and model results into a polished report.
    Call this as the final step after all models have been evaluated.

    Args:
        job_id: Unique job identifier.

    Returns:
        Confirmation that the report was saved, with the file path.
    """
    job_dir = JOBS_DIR / job_id
    metadata_path = job_dir / "metadata.json"
    metrics_path  = job_dir / "metrics.json"

    if not metadata_path.exists():
        return "ERROR: Run load_dataset first."
    if not metrics_path.exists():
        return "ERROR: No model metrics found. Train and evaluate models first."

    with open(metadata_path) as f:
        meta = json.load(f)
    with open(metrics_path) as f:
        all_metrics = json.load(f)

    # Load EDA summary if available
    eda_summary_path = job_dir / "eda" / "summary.json"
    eda_findings = "EDA not run."
    if eda_summary_path.exists():
        with open(eda_summary_path) as f:
            eda_data = json.load(f)
        eda_findings = "\n".join(f"- {f}" for f in eda_data.get("findings", []))

    # Find best model
    best_model = None
    best_score = -1.0
    for model_name, m in all_metrics.items():
        score = m.get("roc_auc") or m.get("f1_macro") or m.get("r2") or 0.0
        if score and score > best_score:
            best_score = score
            best_model = model_name

    # Generate narrative via Claude Sonnet
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    prompt = REPORT_PROMPT.format(
        metadata=json.dumps(meta, indent=2),
        eda_findings=eda_findings,
        model_metrics=json.dumps(all_metrics, indent=2),
    )
    response = client.messages.create(
        model=REPORT_MODEL,
        max_tokens=3000,
        messages=[{"role": "user", "content": prompt}],
    )
    narrative = response.content[0].text

    # Build model comparison table rows
    model_rows = ""
    for name, m in all_metrics.items():
        highlight = ' style="background:#d4edda;"' if name == best_model else ""
        if meta["problem_type"] != "regression":
            acc = m.get("accuracy", 0)
            acc_str = f"{acc:.1%}" if isinstance(acc, float) else "N/A"
            f1m = m.get("f1_minority", 0)
            f1m_str = f"{f1m:.3f}" if isinstance(f1m, float) else "N/A"
            auc = m.get("roc_auc") or "N/A"
            model_rows += (
                f"<tr{highlight}>"
                f"<td>{name}</td><td>{acc_str}</td>"
                f"<td>{f1m_str}</td><td>{auc}</td>"
                f"</tr>\n"
            )
        else:
            r2   = m.get("r2", "N/A")
            mae  = m.get("mae", "N/A")
            rmse = m.get("rmse", "N/A")
            model_rows += (
                f"<tr{highlight}>"
                f"<td>{name}</td><td>{r2}</td>"
                f"<td>{mae}</td><td>{rmse}</td>"
                f"</tr>\n"
            )

    table_headers = (
        "<tr><th>Model</th><th>Accuracy</th><th>F1 (minority)</th><th>ROC-AUC</th></tr>"
        if meta["problem_type"] != "regression"
        else "<tr><th>Model</th><th>R2</th><th>MAE</th><th>RMSE</th></tr>"
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Autonomous ML Agent — Report</title>
  <style>
    body  {{ font-family: 'Segoe UI', sans-serif; max-width: 980px;
             margin: 40px auto; padding: 0 20px; color: #1a1a2e; }}
    h1    {{ color: #16213e; border-bottom: 3px solid #0f3460; padding-bottom: 8px; }}
    h2    {{ color: #0f3460; margin-top: 36px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
    th,td {{ border: 1px solid #dee2e6; padding: 10px 14px; text-align: left; }}
    thead {{ background: #0f3460; color: white; }}
    tr:nth-child(even) {{ background: #f8f9fa; }}
    .badge {{ display:inline-block; padding:4px 12px; border-radius:20px;
              font-weight:bold; font-size:0.9em; }}
    .good  {{ background:#d4edda; color:#155724; }}
    .narrative {{ background:#f0f4ff; border-left:4px solid #0f3460;
                  padding:20px; border-radius:0 8px 8px 0; white-space:pre-wrap; }}
    .info-grid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:16px; margin:20px 0; }}
    .info-card {{ border:1px solid #dee2e6; border-radius:8px; padding:16px;
                  text-align:center; background:#f8f9fa; }}
    .info-card .value {{ font-size:2em; font-weight:bold; color:#0f3460; }}
    .info-card .label {{ font-size:0.85em; color:#6c757d; margin-top:4px; }}
  </style>
</head>
<body>
<h1>Autonomous ML Pipeline Agent — Evaluation Report</h1>
<p><strong>Job ID:</strong> {job_id} &nbsp;|&nbsp;
   <strong>Target:</strong> {meta['target_column']} &nbsp;|&nbsp;
   <strong>Problem:</strong> {meta['problem_type']} &nbsp;|&nbsp;
   <strong>Best model:</strong>
   <span class="badge good">{best_model}</span></p>

<h2>Dataset Overview</h2>
<div class="info-grid">
  <div class="info-card">
    <div class="value">{meta['shape'][0]:,}</div>
    <div class="label">Rows</div>
  </div>
  <div class="info-card">
    <div class="value">{meta['shape'][1]}</div>
    <div class="label">Columns</div>
  </div>
  <div class="info-card">
    <div class="value">{meta['missing_total']}</div>
    <div class="label">Missing Values</div>
  </div>
</div>

<h2>Model Comparison</h2>
<table>
  <thead>{table_headers}</thead>
  <tbody>{model_rows}</tbody>
</table>
<p><em>Highlighted row = best model selected by agent.</em></p>

<h2>Narrative Analysis (claude-sonnet-4-6)</h2>
<div class="narrative">{narrative}</div>

<hr>
<p style="color:#6c757d;font-size:0.85em;">
  Report generated by the Autonomous ML Pipeline Agent.<br>
  Muhammad Irtaza Khan — Portfolio Project
</p>
</body>
</html>"""

    report_path = job_dir / "report.html"
    report_path.write_text(html, encoding="utf-8")

    return (
        f"Report generated and saved to {report_path}.\n"
        f"Best model: {best_model} (score: {best_score:.4f})\n"
        f"Models evaluated: {list(all_metrics.keys())}"
    )
