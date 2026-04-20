"""
Streamlit frontend for the Autonomous ML Pipeline Agent.

Tabs:
  1. Run Agent  — upload CSV, set target column, launch job
  2. Monitor    — live job status + agent reasoning log
  3. Results    — model metrics comparison
  4. Report     — full HTML report
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8001")

st.set_page_config(
    page_title="Autonomous ML Agent",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Autonomous ML Pipeline Agent")
st.caption(
    "Upload any CSV dataset — the AI agent handles EDA, model selection, "
    "training, evaluation, and report generation automatically."
)

# ── Session state ──────────────────────────────────────────────────────────────
if "job_id" not in st.session_state:
    st.session_state.job_id = None
if "job_complete" not in st.session_state:
    st.session_state.job_complete = False

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["🚀 Run Agent", "📡 Monitor", "📊 Results", "📄 Report"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Run Agent
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Upload Dataset & Launch Pipeline")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded = st.file_uploader(
            "Upload a CSV file", type=["csv"],
            help="Any tabular CSV. The agent will auto-detect the problem type."
        )

    with col2:
        target_col = st.text_input(
            "Target column (optional)",
            placeholder="Leave blank for auto-detect",
            help="The column the agent should predict. Leave empty to auto-detect."
        )

    if uploaded:
        df_preview = pd.read_csv(uploaded)
        st.subheader("Preview (first 5 rows)")
        st.dataframe(df_preview.head(), use_container_width=True)

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Rows", f"{len(df_preview):,}")
        col_b.metric("Columns", len(df_preview.columns))
        col_c.metric("Missing values", int(df_preview.isnull().sum().sum()))

        st.markdown("**Columns:** " + ", ".join(f"`{c}`" for c in df_preview.columns))

    launch = st.button(
        "🚀 Launch Autonomous Pipeline",
        type="primary",
        disabled=uploaded is None,
    )

    if launch and uploaded:
        uploaded.seek(0)
        with st.spinner("Submitting job..."):
            try:
                resp = requests.post(
                    f"{API_URL}/jobs",
                    files={"file": (uploaded.name, uploaded, "text/csv")},
                    data={"target_column": target_col or "auto"},
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
                st.session_state.job_id       = data["job_id"]
                st.session_state.job_complete = False
                st.success(
                    f"Job launched! ID: **{data['job_id']}**  \n"
                    "Switch to the **Monitor** tab to watch progress."
                )
            except Exception as e:
                st.error(f"Failed to launch job: {e}")

    # Show recent jobs
    st.divider()
    st.subheader("Recent Jobs")
    try:
        jobs_resp = requests.get(f"{API_URL}/jobs", timeout=5)
        if jobs_resp.ok:
            jobs = jobs_resp.json().get("jobs", [])
            if jobs:
                jobs_df = pd.DataFrame(jobs)
                st.dataframe(jobs_df, use_container_width=True)

                selected = st.selectbox(
                    "Load a previous job",
                    options=[""] + [j["job_id"] for j in jobs],
                )
                if selected:
                    st.session_state.job_id = selected
                    st.info(f"Loaded job {selected}. Check the other tabs.")
            else:
                st.info("No jobs yet. Upload a dataset above to start.")
    except Exception:
        st.warning("Could not reach API. Make sure Docker services are running.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Monitor
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Job Monitor")

    job_id = st.session_state.job_id
    if not job_id:
        st.info("No active job. Launch one in the Run Agent tab.")
    else:
        st.markdown(f"**Job ID:** `{job_id}`")

        auto_refresh = st.checkbox("Auto-refresh every 5s", value=True)

        try:
            status_resp = requests.get(f"{API_URL}/jobs/{job_id}", timeout=5)
            if status_resp.ok:
                status = status_resp.json()
                s = status.get("status", "unknown")

                if s == "complete":
                    st.success(f"Pipeline complete: {status.get('message', '')}")
                    st.session_state.job_complete = True
                elif s == "failed":
                    st.error(f"Pipeline failed: {status.get('message', '')}")
                elif s == "running":
                    st.info(f"Running: {status.get('message', '')}")
                    if auto_refresh:
                        time.sleep(5)
                        st.rerun()
                else:
                    st.warning(f"Status: {s} — {status.get('message', '')}")
                    if auto_refresh:
                        time.sleep(5)
                        st.rerun()

                # Show raw status
                with st.expander("Raw status JSON"):
                    st.json(status)
            else:
                st.error("Could not fetch job status.")
        except Exception as e:
            st.error(f"API error: {e}")

        # Agent reasoning log
        st.subheader("Agent Reasoning Log")
        log_path = Path(f"jobs/{job_id}/agent_log.json")
        if log_path.exists():
            with open(log_path) as f:
                logs = json.load(f)
            for entry in logs:
                role    = entry.get("role", "")
                content = entry.get("content", "")
                if not content:
                    continue
                if role == "ai":
                    st.markdown(f"**Agent:** {content[:800]}{'...' if len(content) > 800 else ''}")
                elif role == "tool":
                    with st.expander(f"Tool result"):
                        st.text(content[:1000])
        else:
            st.info("Agent log not yet available.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Results
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Model Results")

    job_id = st.session_state.job_id
    if not job_id:
        st.info("No active job.")
    else:
        try:
            metrics_resp = requests.get(f"{API_URL}/jobs/{job_id}/metrics", timeout=5)
            if metrics_resp.ok:
                metrics = metrics_resp.json()

                # Build comparison table
                rows = []
                for model_name, m in metrics.items():
                    row = {"Model": model_name}
                    row.update({k: v for k, v in m.items()
                                if k not in ("model", "problem_type",
                                             "classification_report", "top_features")
                                and v is not None})
                    rows.append(row)

                if rows:
                    df_metrics = pd.DataFrame(rows).set_index("Model")
                    st.dataframe(df_metrics.style.highlight_max(axis=0, color="#d4edda"),
                                 use_container_width=True)

                # Per-model detail
                for model_name, m in metrics.items():
                    with st.expander(f"Detail — {model_name}"):
                        if "classification_report" in m:
                            st.text(m["classification_report"])
                        if "top_features" in m:
                            st.markdown("**Top features:**")
                            for feat, imp in m["top_features"]:
                                st.progress(float(imp), text=f"{feat}: {imp:.4f}")

                # Show plots
                eval_dir = Path(f"jobs/{job_id}/eval")
                eda_dir  = Path(f"jobs/{job_id}/eda")

                if eval_dir.exists():
                    st.subheader("Confusion Matrices & Feature Importance")
                    plots = list(eval_dir.glob("*.png"))
                    if plots:
                        cols = st.columns(min(len(plots), 2))
                        for i, p in enumerate(plots):
                            cols[i % 2].image(str(p), use_container_width=True)

                if eda_dir.exists():
                    st.subheader("EDA Plots")
                    eda_plots = list(eda_dir.glob("*.png"))
                    if eda_plots:
                        cols = st.columns(min(len(eda_plots), 2))
                        for i, p in enumerate(eda_plots):
                            cols[i % 2].image(str(p), use_container_width=True)
            else:
                st.info("No metrics yet. Wait for the pipeline to complete.")
        except Exception as e:
            st.error(f"Error fetching metrics: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Report
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Full Evaluation Report")

    job_id = st.session_state.job_id
    if not job_id:
        st.info("No active job.")
    else:
        report_path = Path(f"jobs/{job_id}/report.html")
        if report_path.exists():
            st.components.v1.html(
                report_path.read_text(encoding="utf-8"),
                height=900,
                scrolling=True,
            )
        else:
            st.info("Report not generated yet. Wait for the pipeline to finish.")
            if st.button("Check for report"):
                st.rerun()
