from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def configure_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")


def plot_forecast_case(
    history: np.ndarray,
    actual: np.ndarray,
    predictions: dict[str, np.ndarray],
    destination: Path | None = None,
):
    configure_style()
    figure, axis = plt.subplots(figsize=(11, 4.5))
    history_x = np.arange(-len(history) + 1, 1)
    future_x = np.arange(1, len(actual) + 1)
    axis.plot(history_x, history, color="#3f4b59", label="History")
    axis.plot(future_x, actual, color="#111111", linewidth=2, label="Observed")
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    for color, (name, values) in zip(colors, predictions.items()):
        axis.plot(future_x, values, label=name, color=color)
    axis.axvline(0, color="#666666", linestyle="--", linewidth=1)
    axis.set(xlabel="Hours from forecast origin", ylabel="PM2.5", title="24-hour PM2.5 forecast case")
    axis.legend(ncol=2)
    figure.tight_layout()
    if destination:
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=180, bbox_inches="tight")
    return figure


def plot_horizon_metrics(
    metrics: pd.DataFrame,
    metric: str = "mae",
    destination: Path | None = None,
    subset: str = "all",
):
    configure_style()
    mask = pd.to_numeric(metrics["horizon"], errors="coerce").notna()
    if "subset" in metrics:
        mask &= metrics["subset"].eq(subset)
    data = metrics.loc[mask].copy()
    if data.empty:
        raise ValueError(f"No horizon metrics found for subset {subset!r}")
    data["horizon"] = data["horizon"].astype(int)
    figure, axis = plt.subplots(figsize=(10, 4.5))
    sns.lineplot(data=data, x="horizon", y=metric, hue="model", marker="o", ax=axis)
    axis.set(title=f"{metric.upper()} by forecast horizon", xlabel="Forecast horizon (hours)")
    figure.tight_layout()
    if destination:
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=180, bbox_inches="tight")
    return figure


def plot_retrieval_diagnostics(evidence: pd.DataFrame, destination: Path | None = None):
    configure_style()
    if evidence.empty:
        raise ValueError("Retrieval evidence is empty")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    sns.histplot(
        data=evidence,
        x="total_score",
        hue="method",
        element="step",
        stat="density",
        common_norm=False,
        ax=axes[0],
    )
    axes[0].set_title("Retrieved-candidate score distribution")
    axes[1].scatter(
        pd.to_datetime(evidence["query_origin"], utc=True),
        evidence["time_series_score"],
        s=8,
        alpha=0.35,
    )
    axes[1].set(title="Time-series similarity over origins", xlabel="Query origin", ylabel="Similarity")
    figure.tight_layout()
    if destination:
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=180, bbox_inches="tight")
    return figure


def plot_drift_scores(evidence: pd.DataFrame, destination: Path | None = None):
    configure_style()
    if evidence.empty:
        raise ValueError("Drift evidence is empty")
    data = evidence.sort_values("origin_time")
    figure, axis = plt.subplots(figsize=(11, 4.2))
    axis.plot(pd.to_datetime(data["origin_time"], utc=True), data["drift_score"], linewidth=1)
    threshold = float(data["drift_threshold"].iloc[0])
    axis.axhline(threshold, color="#D55E00", linestyle="--", label="Validation threshold")
    axis.set(title="Composite drift score", xlabel="Forecast origin", ylabel="Drift score")
    axis.legend()
    figure.tight_layout()
    if destination:
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=180, bbox_inches="tight")
    return figure
