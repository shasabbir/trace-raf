from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from .config import ProjectConfig
from .windows import WindowDataset


@dataclass
class DriftResult:
    components: np.ndarray
    component_names: tuple[str, ...]
    score: np.ndarray
    flag: np.ndarray
    threshold: float


class DriftDetector:
    base_component_names = (
        "mean_shift",
        "variance_shift",
        "retrieval_similarity_drop",
        "weather_shift",
    )

    def __init__(self, cfg: ProjectConfig, include_event_component: bool = True):
        self.cfg = cfg
        self.include_event_component = include_event_component
        self.component_names = self.base_component_names + (("event_burst",) if include_event_component else ())
        self.center: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.weather_center: np.ndarray | None = None
        self.weather_scale: np.ndarray | None = None
        self.threshold: float | None = None

    @staticmethod
    def _index(dataset: WindowDataset, name: str) -> int:
        try:
            return dataset.feature_names.index(name)
        except ValueError as error:
            raise ValueError(f"Required drift feature is missing: {name}") from error

    def _raw(self, dataset: WindowDataset, retrieval_similarity: np.ndarray, fit_weather: bool = False) -> np.ndarray:
        mean24 = dataset.features[:, self._index(dataset, "pm25_roll_mean_24h")]
        mean168 = dataset.features[:, self._index(dataset, "pm25_roll_mean_168h")]
        std24 = dataset.features[:, self._index(dataset, "pm25_roll_std_24h")]
        std168 = dataset.features[:, self._index(dataset, "pm25_roll_std_168h")]
        weather_indices = [i for i, name in enumerate(dataset.feature_names) if name in {
            "weather_temperature_c", "weather_relative_humidity", "weather_pressure_hpa", "weather_wind_speed_ms"
        }]
        weather = dataset.features[:, weather_indices]
        if fit_weather:
            self.weather_center = np.nanmedian(weather, axis=0)
            self.weather_scale = np.nanstd(weather, axis=0)
            self.weather_scale = np.maximum(self.weather_scale, 1e-6)
        if self.weather_center is None or self.weather_scale is None:
            raise RuntimeError("Drift detector weather reference has not been fitted")
        weather_shift = np.sqrt(np.mean(((weather - self.weather_center) / self.weather_scale) ** 2, axis=1))
        similarity = np.asarray(retrieval_similarity, dtype=float)
        similarity_fill = np.nanmedian(similarity) if np.isfinite(similarity).any() else 0.0
        similarity = np.nan_to_num(similarity, nan=similarity_fill)
        components = [
            np.abs(mean24 - mean168),
            np.abs(np.log((std24 + 1e-6) / (std168 + 1e-6))),
            1.0 - np.clip(similarity, 0, 1),
            weather_shift,
        ]
        if self.include_event_component:
            event_burst = dataset.features[:, self._index(dataset, "event_burst_ratio")]
            components.append(np.maximum(event_burst, 0))
        return np.column_stack(components)

    def fit_reference(self, train: WindowDataset, retrieval_similarity: np.ndarray) -> "DriftDetector":
        raw = self._raw(train, retrieval_similarity, fit_weather=True)
        self.center = np.nanmedian(raw, axis=0)
        mad = np.nanmedian(np.abs(raw - self.center), axis=0)
        self.scale = np.maximum(1.4826 * mad, 1e-6)
        return self

    def _components(self, dataset: WindowDataset, retrieval_similarity: np.ndarray) -> np.ndarray:
        if self.center is None or self.scale is None:
            raise RuntimeError("Drift detector reference has not been fitted")
        raw = self._raw(dataset, retrieval_similarity)
        standardized = (raw - self.center) / self.scale
        if self.cfg.drift.score_mode == "two_sided":
            standardized = np.abs(standardized)
        return np.clip(standardized, 0, 6) / 6.0

    def calibrate(self, validation: WindowDataset, retrieval_similarity: np.ndarray) -> float:
        components = self._components(validation, retrieval_similarity)
        scores = components.mean(axis=1)
        self.threshold = float(np.quantile(scores, self.cfg.drift.threshold_quantile))
        return self.threshold

    def transform(self, dataset: WindowDataset, retrieval_similarity: np.ndarray) -> DriftResult:
        if self.threshold is None:
            raise RuntimeError("Drift threshold has not been calibrated on validation data")
        components = self._components(dataset, retrieval_similarity).astype(np.float32)
        score = components.mean(axis=1).astype(np.float32)
        return DriftResult(
            components=components,
            component_names=self.component_names,
            score=score,
            flag=score >= self.threshold,
            threshold=self.threshold,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)


class KSTwoSampleDriftDetector:
    """Validation-calibrated KS benchmark over each causal PM2.5 lookback."""

    def __init__(self, cfg: ProjectConfig):
        self.cfg = cfg
        self.reference: np.ndarray | None = None
        self.threshold: float | None = None

    def fit_reference(self, train: WindowDataset) -> "KSTwoSampleDriftDetector":
        # One value per training origin avoids overweighting observations that
        # appear in many overlapping lookback windows.
        values = np.asarray(train.x[:, -1], dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError("Cannot fit KS detector on an empty training reference")
        rng = np.random.default_rng(self.cfg.seed)
        size = min(values.size, self.cfg.drift.ks_reference_size)
        self.reference = rng.choice(values, size=size, replace=False)
        return self

    def _scores(self, dataset: WindowDataset) -> np.ndarray:
        if self.reference is None:
            raise RuntimeError("KS detector reference has not been fitted")
        return np.asarray(
            [ks_2samp(row, self.reference, method="asymp").statistic for row in dataset.x],
            dtype=np.float32,
        )

    def calibrate(self, validation: WindowDataset) -> float:
        self.threshold = float(
            np.quantile(self._scores(validation), self.cfg.drift.threshold_quantile)
        )
        return self.threshold

    def transform(self, dataset: WindowDataset) -> DriftResult:
        if self.threshold is None:
            raise RuntimeError("KS threshold has not been calibrated on validation data")
        score = self._scores(dataset)
        return DriftResult(
            components=score[:, None],
            component_names=("ks_statistic",),
            score=score,
            flag=score >= self.threshold,
            threshold=self.threshold,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)


def drift_evidence_frame(
    dataset: WindowDataset,
    result: DriftResult,
    run_id: str,
) -> pd.DataFrame:
    frame = dataset.metadata[["window_id", "origin_time", "split"]].copy()
    frame.insert(0, "run_id", run_id)
    for index, name in enumerate(result.component_names):
        frame[f"drift_{name}"] = result.components[:, index]
    frame["drift_score"] = result.score
    frame["drift_flag"] = result.flag
    frame["drift_threshold"] = result.threshold
    return frame
