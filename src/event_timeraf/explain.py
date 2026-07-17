from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .drift import DriftResult
from .retrieval import RetrievalResult
from .windows import WindowDataset


def xgb_local_contributions(model, matrix: np.ndarray) -> np.ndarray:
    import xgboost as xgb

    return model.get_booster().predict(xgb.DMatrix(matrix), pred_contribs=True)[:, :-1]


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
        if feature_contributions is not None and contribution_names:
            order = np.argsort(np.abs(feature_contributions[index]))[::-1][:3]
            driver_names = [contribution_names[position] for position in order]
        query_id = metadata["window_id"]
        retrieved_ids = evidence_by_query.get(query_id, [])
        event_ids = _recent_event_ids(events, metadata["origin_time"])
        drift_parts = drift.components[index]
        drift_reasons = [
            name for name, value in zip(drift.component_names, drift_parts) if value >= 0.5
        ]
        sentences = [
            f"The mean 24-hour forecast is expected to {direction} by {abs(delta):.1f} PM2.5 units relative to the latest observation."
        ]
        if driver_names:
            sentences.append("The strongest model contribution fields are " + ", ".join(driver_names) + ".")
        if retrieved_ids:
            sentences.append("The forecast is supported by retrieved historical windows " + ", ".join(retrieved_ids) + ".")
        if event_ids:
            sentences.append("Recent source-recorded event evidence includes " + ", ".join(event_ids) + ".")
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
                "drift_score": float(drift.score[index]),
                "drift_flag": bool(drift.flag[index]),
            }
        )
    return pd.DataFrame(rows)

