"""
Tool: run_eda
Performs exploratory data analysis — statistics, correlations, class breakdown.
Saves plots to jobs/{job_id}/eda/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from langchain_core.tools import tool

from app.config import JOBS_DIR, UPLOADS_DIR


@tool
def run_eda(job_id: str) -> str:
    """
    Run exploratory data analysis on the loaded dataset.
    Generates statistics and saves plots (class distribution, correlations,
    missing values, feature distributions). Call this after load_dataset.

    Args:
        job_id: Unique job identifier.

    Returns:
        EDA summary with key statistics and findings.
    """
    metadata_path = JOBS_DIR / job_id / "metadata.json"
    if not metadata_path.exists():
        return "ERROR: Run load_dataset first."

    with open(metadata_path) as f:
        meta = json.load(f)

    df = pd.read_csv(UPLOADS_DIR / job_id / "dataset.csv")
    target_col    = meta["target_column"]
    problem_type  = meta["problem_type"]
    numeric_cols  = [c for c in meta["numeric_columns"] if c != target_col]

    eda_dir = JOBS_DIR / job_id / "eda"
    eda_dir.mkdir(exist_ok=True)

    findings = []

    # ── 1. Descriptive statistics ──────────────────────────────────────────────
    desc = df[numeric_cols].describe().round(4) if numeric_cols else pd.DataFrame()
    if not desc.empty:
        desc.to_csv(eda_dir / "descriptive_stats.csv")
        findings.append(f"Computed descriptive stats for {len(numeric_cols)} numeric features.")

    # ── 2. Class distribution plot (classification only) ──────────────────────
    if problem_type != "regression":
        fig, ax = plt.subplots(figsize=(7, 4))
        df[target_col].value_counts().plot(kind="bar", ax=ax, color="#0f3460")
        ax.set_title(f"Class Distribution — {target_col}")
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        plt.tight_layout()
        fig.savefig(eda_dir / "class_distribution.png", dpi=120)
        plt.close(fig)
        findings.append("Saved class distribution plot.")

    # ── 3. Correlation heatmap ─────────────────────────────────────────────────
    if len(numeric_cols) >= 2:
        corr_cols = numeric_cols[:15]   # cap at 15 to keep plot readable
        corr = df[corr_cols].corr().round(3)
        corr.to_csv(eda_dir / "correlation_matrix.csv")

        fig, ax = plt.subplots(figsize=(max(8, len(corr_cols)), max(6, len(corr_cols) - 2)))
        sns.heatmap(corr, annot=len(corr_cols) <= 10, fmt=".2f",
                    cmap="coolwarm", center=0, ax=ax, square=True)
        ax.set_title("Feature Correlation Heatmap")
        plt.tight_layout()
        fig.savefig(eda_dir / "correlation_heatmap.png", dpi=120)
        plt.close(fig)

        # Find high correlations
        high_corr = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                val = abs(corr.iloc[i, j])
                if val > 0.8:
                    high_corr.append(
                        f"{corr.columns[i]} <-> {corr.columns[j]} ({val:.2f})"
                    )
        if high_corr:
            findings.append(f"High correlations (>0.8): {high_corr}")
        else:
            findings.append("No high correlations (>0.8) found between features.")

    # ── 4. Feature vs target (top 6 numeric features) ────────────────────────
    if numeric_cols and problem_type != "regression":
        top_feats = numeric_cols[:6]
        fig, axes = plt.subplots(2, 3, figsize=(14, 8)) if len(top_feats) >= 3 \
                    else plt.subplots(1, len(top_feats), figsize=(5 * len(top_feats), 4))
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for idx, feat in enumerate(top_feats):
            ax = axes[idx]
            for cls in df[target_col].unique():
                subset = df[df[target_col] == cls][feat].dropna()
                ax.hist(subset, alpha=0.6, label=str(cls), bins=25)
            ax.set_title(feat)
            ax.set_xlabel(feat)
            ax.legend(fontsize=7)

        # Hide unused subplots
        for idx in range(len(top_feats), len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle("Feature Distributions by Class", fontsize=13)
        plt.tight_layout()
        fig.savefig(eda_dir / "feature_distributions.png", dpi=120)
        plt.close(fig)
        findings.append(f"Saved feature distribution plots for {top_feats}.")

    # ── 5. Missing values ─────────────────────────────────────────────────────
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        missing.plot(kind="bar", ax=ax, color="#e74c3c")
        ax.set_title("Missing Values per Column")
        ax.set_ylabel("Count")
        plt.tight_layout()
        fig.savefig(eda_dir / "missing_values.png", dpi=120)
        plt.close(fig)
        findings.append(f"Missing values in {len(missing)} columns: {missing.to_dict()}")
    else:
        findings.append("No missing values in the dataset.")

    # ── 6. Key numeric stats for agent ────────────────────────────────────────
    stats_summary = {}
    if not desc.empty:
        for col in numeric_cols[:8]:
            if col in desc.columns:
                stats_summary[col] = {
                    "mean":  round(float(desc[col]["mean"]), 4),
                    "std":   round(float(desc[col]["std"]), 4),
                    "min":   round(float(desc[col]["min"]), 4),
                    "max":   round(float(desc[col]["max"]), 4),
                }

    # Save EDA summary
    eda_summary = {"findings": findings, "stats": stats_summary}
    with open(eda_dir / "summary.json", "w") as f:
        json.dump(eda_summary, f, indent=2)

    return (
        "EDA complete. Plots saved to jobs directory.\n\n"
        "Findings:\n" + "\n".join(f"- {f}" for f in findings) +
        f"\n\nKey feature stats (sample):\n{json.dumps(stats_summary, indent=2)}"
    )
