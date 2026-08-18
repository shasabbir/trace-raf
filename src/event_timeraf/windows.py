from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ProjectConfig
from .features import CALENDAR_FEATURES, model_feature_columns


@dataclass
class WindowDataset:
    x: np.ndarray
    y: np.ndarray
    features: np.ndarray
    future_calendar: np.ndarray
    metadata: pd.DataFrame
    feature_names: tuple[str, ...]
    calendar_names: tuple[str, ...]

    def subset(self, split: str) -> "WindowDataset":
        mask = self.metadata["split"].to_numpy() == split
        return WindowDataset(
            x=self.x[mask],
            y=self.y[mask],
            features=self.features[mask],
            future_calendar=self.future_calendar[mask],
            metadata=self.metadata.loc[mask].reset_index(drop=True),
            feature_names=self.feature_names,
            calendar_names=self.calendar_names,
        )

    def save(self, arrays_path: Path, metadata_path: Path) -> None:
        arrays_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            arrays_path,
            x=self.x,
            y=self.y,
            features=self.features,
            future_calendar=self.future_calendar,
        )
        self.metadata.to_parquet(metadata_path, index=False)
        schema = {
            "feature_names": self.feature_names,
            "calendar_names": self.calendar_names,
            "x_shape": list(self.x.shape),
            "y_shape": list(self.y.shape),
        }
        arrays_path.with_suffix(".json").write_text(json.dumps(schema, indent=2), encoding="utf-8")


def _split_labels(n_rows: int, ratios: tuple[float, float, float]) -> tuple[int, int]:
    train_end = int(n_rows * ratios[0])
    validation_end = int(n_rows * (ratios[0] + ratios[1]))
    return train_end, validation_end


def _split_boundaries(data: pd.DataFrame, cfg: ProjectConfig) -> tuple[int, int]:
    validation_start = cfg.forecast.validation_start_date
    test_start = cfg.forecast.test_start_date
    if validation_start is None or test_start is None:
        return _split_labels(len(data), cfg.forecast.split_ratios)
    timestamps = pd.to_datetime(data["timestamp_utc"], utc=True)
    validation_time = pd.Timestamp(validation_start, tz="UTC")
    test_time = pd.Timestamp(test_start, tz="UTC")
    train_end = int(timestamps.searchsorted(validation_time, side="left"))
    validation_end = int(timestamps.searchsorted(test_time, side="left"))
    if not 0 < train_end < validation_end < len(data):
        raise ValueError("Configured forecast split dates do not produce three non-empty periods")
    return train_end, validation_end


def build_window_dataset(frame: pd.DataFrame, cfg: ProjectConfig) -> WindowDataset:
    data = frame.sort_values("timestamp_utc").reset_index(drop=True).copy()
    timestamps = pd.to_datetime(data["timestamp_utc"], utc=True)
    lookback = cfg.forecast.lookback
    horizon = cfg.forecast.horizon
    feature_names = tuple(model_feature_columns(data))
    calendar_names = tuple(CALENDAR_FEATURES)
    train_end, validation_end = _split_boundaries(data, cfg)

    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    feature_rows: list[np.ndarray] = []
    calendar_rows: list[np.ndarray] = []
    metadata: list[dict] = []
    pm_input = data["pm25"].to_numpy(dtype=float)
    pm_target = data["pm25_observed"].to_numpy(dtype=float)
    feature_values = data.loc[:, feature_names].to_numpy(dtype=float)
    calendar_values = data.loc[:, calendar_names].to_numpy(dtype=float)
    site_id = str(data["site_id"].dropna().iloc[0])

    for origin in range(lookback - 1, len(data) - horizon):
        target_end = origin + horizon
        if origin < train_end and target_end < train_end:
            split = "train"
        elif train_end <= origin < validation_end and target_end < validation_end:
            split = "validation"
        elif origin >= validation_end:
            split = "test"
        else:
            continue

        x = pm_input[origin - lookback + 1 : origin + 1]
        y = pm_target[origin + 1 : target_end + 1]
        features = feature_values[origin]
        future_calendar = calendar_values[origin + 1 : target_end + 1]
        if (
            np.isnan(x).any()
            or np.isnan(y).any()
            or np.isnan(features).any()
            or np.isnan(future_calendar).any()
        ):
            continue
        x_rows.append(x.astype(np.float32))
        y_rows.append(y.astype(np.float32))
        feature_rows.append(features.astype(np.float32))
        calendar_rows.append(future_calendar.astype(np.float32))
        metadata.append(
            {
                "window_id": f"w_{origin:07d}",
                "origin_time": timestamps.iloc[origin],
                "input_start": timestamps.iloc[origin - lookback + 1],
                "target_start": timestamps.iloc[origin + 1],
                "target_end": timestamps.iloc[target_end],
                "origin_row": origin,
                "split": split,
                "site_id": site_id,
            }
        )

    if not metadata:
        raise ValueError("No valid windows were created; inspect missingness and feature coverage")
    dataset = WindowDataset(
        x=np.stack(x_rows),
        y=np.stack(y_rows),
        features=np.stack(feature_rows),
        future_calendar=np.stack(calendar_rows),
        metadata=pd.DataFrame(metadata),
        feature_names=feature_names,
        calendar_names=calendar_names,
    )
    assert_window_integrity(dataset, cfg)
    return dataset


def assert_window_integrity(dataset: WindowDataset, cfg: ProjectConfig) -> None:
    if dataset.x.ndim != 2 or dataset.x.shape[1] != cfg.forecast.lookback:
        raise AssertionError("Input windows do not match configured lookback")
    if dataset.y.ndim != 2 or dataset.y.shape[1] != cfg.forecast.horizon:
        raise AssertionError("Targets do not match configured horizon")
    if len(dataset.metadata) != len(dataset.x):
        raise AssertionError("Metadata and array lengths differ")
    if np.isnan(dataset.y).any():
        raise AssertionError("Targets must never be interpolated or missing")
    metadata = dataset.metadata
    if not (metadata["input_start"] <= metadata["origin_time"]).all():
        raise AssertionError("Input starts after a forecast origin")
    if not (metadata["origin_time"] < metadata["target_start"]).all():
        raise AssertionError("Targets are not strictly after forecast origins")
    order = {"train": 0, "validation": 1, "test": 2}
    split_order = metadata["split"].map(order).to_numpy()
    if np.any(np.diff(split_order) < 0):
        raise AssertionError("Window splits are not chronological")


def window_attrition_table(
    frame: pd.DataFrame,
    dataset: WindowDataset,
    cfg: ProjectConfig,
    run_id: str = "unassigned",
) -> pd.DataFrame:
    """Reconcile hourly coverage with the staged loss of forecast origins."""
    data = frame.sort_values("timestamp_utc").reset_index(drop=True)
    lookback = cfg.forecast.lookback
    horizon = cfg.forecast.horizon
    origins = np.arange(lookback - 1, len(data) - horizon)
    train_end, validation_end = _split_boundaries(data, cfg)
    target_ends = origins + horizon
    split_eligible = (
        ((origins < train_end) & (target_ends < train_end))
        | ((origins >= train_end) & (origins < validation_end) & (target_ends < validation_end))
        | (origins >= validation_end)
    )

    pm_input = data["pm25"].to_numpy(dtype=float)
    pm_target = data["pm25_observed"].to_numpy(dtype=float)
    features = data.loc[:, model_feature_columns(data)].to_numpy(dtype=float)
    calendar = data.loc[:, CALENDAR_FEATURES].to_numpy(dtype=float)
    complete_input = np.array(
        [np.isfinite(pm_input[origin - lookback + 1 : origin + 1]).all() for origin in origins]
    )
    complete_target = np.array(
        [np.isfinite(pm_target[origin + 1 : origin + horizon + 1]).all() for origin in origins]
    )
    complete_features = np.isfinite(features[origins]).all(axis=1)
    complete_calendar = np.array(
        [np.isfinite(calendar[origin + 1 : origin + horizon + 1]).all() for origin in origins]
    )

    stages = [
        ("source_hourly_rows", len(data), "Hourly rows in the aligned modeling table"),
        ("bounded_candidate_origins", len(origins), "Origins with a complete history/future time bound"),
    ]
    retained = split_eligible.copy()
    stages.append(("chronological_split_eligible", int(retained.sum()), "Targets do not cross split boundaries"))
    for stage, mask, reason in (
        ("complete_lookback", complete_input, "No missing PM2.5 value in the 168-hour input"),
        ("complete_observed_target", complete_target, "All 24 target values are directly observed"),
        ("complete_origin_features", complete_features, "All causal origin features are finite"),
        ("complete_future_calendar", complete_calendar, "All known-future calendar values are finite"),
    ):
        retained &= mask
        stages.append((stage, int(retained.sum()), reason))
    stages.append(("retained_windows", len(dataset.x), "WindowDataset rows used by the experiment"))
    if int(retained.sum()) != len(dataset.x):
        raise AssertionError("Attrition accounting does not reconcile with WindowDataset")

    rows = []
    previous = None
    for order, (stage, count, criterion) in enumerate(stages, start=1):
        rows.append(
            {
                "run_id": run_id,
                "stage_order": order,
                "stage": stage,
                "count": int(count),
                "removed_since_previous": 0 if previous is None else int(previous - count),
                "retained_fraction_of_bounded_origins": (
                    np.nan if len(origins) == 0 or stage == "source_hourly_rows" else count / len(origins)
                ),
                "criterion": criterion,
            }
        )
        previous = count
    return pd.DataFrame(rows)


def design_matrix(
    dataset: WindowDataset,
    feature_prefixes: tuple[str, ...],
    horizon_index: int,
    extra: np.ndarray | None = None,
) -> np.ndarray:
    indices = [i for i, name in enumerate(dataset.feature_names) if name.startswith(feature_prefixes)]
    if not indices:
        raise ValueError(f"No features match prefixes {feature_prefixes}")
    parts = [dataset.features[:, indices], dataset.future_calendar[:, horizon_index, :]]
    if extra is not None:
        parts.append(extra)
    return np.column_stack(parts).astype(np.float32)

