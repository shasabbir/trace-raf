from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .drift import DriftResult
from .retrieval import RetrievalResult
from .windows import WindowDataset


def xgb_local_contributions(model, matrix: np.ndarray) -> np.ndarray:
    import xgboost as xgb

    return model.get_booster().predict(xgb.DMatrix(matrix), pred_contribs=True)[:, :-1]


def grouped_feature_perturbation(
    actual: np.ndarray,
    matrix: np.ndarray,
    reference_matrix: np.ndarray,
    feature_names: Sequence[str],
    groups: Mapping[str, tuple[str, ...]],
    predict: Callable[[np.ndarray], np.ndarray],
    seed: int,
    run_id: str = "unassigned",
    model: str = "unassigned",
) -> pd.DataFrame:
    """Measure validation-set faithfulness by permutation and median deletion."""
    if matrix.ndim != 2 or reference_matrix.ndim != 2:
        raise ValueError("Feature matrices must be two-dimensional")
    if matrix.shape[1] != reference_matrix.shape[1] or matrix.shape[1] != len(feature_names):
        raise ValueError("Feature matrices and names must have matching columns")
    baseline_prediction = predict(matrix)
    if baseline_prediction.shape != actual.shape:
        raise ValueError("Prediction callback returned an incompatible shape")
    baseline_mse = float(np.mean((actual - baseline_prediction) ** 2))
    reference_median = np.nanmedian(reference_matrix, axis=0)
    rng = np.random.default_rng(seed)
    rows = []
    for group, prefixes in groups.items():
        columns = [
            index for index, name in enumerate(feature_names)
            if any(name.startswith(prefix) for prefix in prefixes)
        ]
        if not columns:
            continue
        order = rng.permutation(len(matrix))
        permuted = matrix.copy()
        permuted[:, columns] = permuted[order][:, columns]
        deleted = matrix.copy()
        deleted[:, columns] = reference_median[columns]
        for intervention, changed in (("permutation", permuted), ("median_deletion", deleted)):
            changed_mse = float(np.mean((actual - predict(changed)) ** 2))
            rows.append(
                {
                    "run_id": run_id,
                    "model": model,
                    "feature_group": group,
                    "intervention": intervention,
                    "feature_count": len(columns),
                    "baseline_validation_mse": baseline_mse,
                    "perturbed_validation_mse": changed_mse,
                    "mse_increase": changed_mse - baseline_mse,
                }
            )
    return pd.DataFrame(rows)


def _recent_event_ids(events: pd.DataFrame, origin: pd.Timestamp, hours: int = 72) -> list[str]:
    if events.empty:
        return []
    published = pd.to_datetime(events["published_at"], utc=True)
    origin = pd.Timestamp(origin)
    origin = origin.tz_localize("UTC") if origin.tzinfo is None else origin.tz_convert("UTC")
    lower = origin - pd.Timedelta(hours=hours)
    selected = events.loc[(published > lower) & (published <= origin)].sort_values("published_at", ascending=False)
    return selected["event_id"].astype(str).head(3).tolist()


def generate_explanations(
    dataset: WindowDataset,
    predictions: np.ndarray,
    retrieval: RetrievalResult,
    drift: DriftResult,
    events: pd.DataFrame,
    feature_contributions: np.ndarray | None = None,
    contribution_names: list[str] | None = None,
    validation_residual_rmse: np.ndarray | None = None,
) -> pd.DataFrame:
    evidence_by_query = {}
    if not retrieval.evidence.empty:
        top = retrieval.evidence.loc[retrieval.evidence["rank"] <= 3]
        evidence_by_query = top.groupby("query_window_id")["candidate_window_id"].apply(list).to_dict()
    rows: list[dict] = []
    for index, metadata in dataset.metadata.iterrows():
        current = float(dataset.x[index, -1])
        forecast_mean = float(np.mean(predictions[index]))
        delta = forecast_mean - current
        direction = "increase" if delta > 1 else "decrease" if delta < -1 else "remain near the current level"
        driver_names: list[str] = []
        driver_records: list[dict[str, float | str]] = []
        if feature_contributions is not None and contribution_names:
            order = np.argsort(np.abs(feature_contributions[index]))[::-1][:3]
            driver_names = [contribution_names[position] for position in order]
            driver_records = [
                {
                    "feature": contribution_names[position],
                    "mean_contribution": float(feature_contributions[index, position]),
                }
                for position in order
            ]
        query_id = metadata["window_id"]
        retrieved_ids = evidence_by_query.get(query_id, [])
        event_ids = _recent_event_ids(events, metadata["origin_time"])
        drift_parts = drift.components[index]
        drift_reasons = [
            name for name, value in zip(drift.component_names, drift_parts) if value >= 0.5
        ]
        drift_component_values = {
            name: float(value) for name, value in zip(drift.component_names, drift_parts)
        }
        retrieval_spread = float(np.nanmean(retrieval.spread[index]))
        validation_rmse = (
            float(np.sqrt(np.nanmean(np.square(validation_residual_rmse))))
            if validation_residual_rmse is not None
            else np.nan
        )
        uncertainty_proxy = (
            float(np.sqrt(retrieval_spread**2 + validation_rmse**2))
            if np.isfinite(retrieval_spread) and np.isfinite(validation_rmse)
            else np.nan
        )
        sentences = [
            f"The mean 24-hour forecast is expected to {direction} by {abs(delta):.1f} PM2.5 units relative to the latest observation."
        ]
        if driver_names:
            sentences.append("The strongest model contribution fields are " + ", ".join(driver_names) + ".")
        if retrieved_ids:
            sentences.append("The forecast is supported by retrieved historical windows " + ", ".join(retrieved_ids) + ".")
        if event_ids:
            sentences.append("Recent source-recorded event evidence includes " + ", ".join(event_ids) + ".")
        if np.isfinite(uncertainty_proxy):
            sentences.append(
                f"The uncertainty proxy is {uncertainty_proxy:.1f}, combining retrieved-trajectory spread "
                "with validation residual RMSE."
            )
        if drift.flag[index]:
            reason = ", ".join(drift_reasons) if drift_reasons else "the composite shift score"
            sentences.append(f"The distribution-shift flag is active because of {reason}; this is an uncertainty warning, not a causal claim.")
        rows.append(
            {
                "window_id": query_id,
                "origin_time": metadata["origin_time"],
                "explanation": " ".join(sentences),
                "retrieved_evidence_ids": json.dumps(retrieved_ids),
                "event_evidence_ids": json.dumps(event_ids),
                "top_feature_effects": json.dumps(driver_records),
                "drift_components": json.dumps(drift_component_values),
                "retrieval_spread": retrieval_spread,
                "validation_residual_rmse": validation_rmse,
                "uncertainty_proxy": uncertainty_proxy,
                "drift_score": float(drift.score[index]),
                "drift_flag": bool(drift.flag[index]),
            }
        )
    return pd.DataFrame(rows)
