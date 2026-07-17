from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PathsConfig:
    raw: Path
    processed: Path
    knowledge_base: Path
    outputs: Path

    def create(self) -> None:
        for path in (self.raw, self.processed, self.knowledge_base, self.outputs):
            path.mkdir(parents=True, exist_ok=True)
        for name in ("audit", "tables", "figures", "predictions", "evidence", "models", "logs"):
            (self.outputs / name).mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class DataConfig:
    start_year: int
    end_year: int
    epa_state_code: str
    epa_county_code: str
    epa_parameter_code: int = 88101
    minimum_pm25_coverage: float = 0.70
    minimum_weather_coverage: float = 0.95
    maximum_fill_gap_hours: int = 3
    minimum_event_days: int = 30
    minimum_event_overlap_days: int = 180
    minimum_event_categories: int = 2
    minimum_event_records_per_category: int = 20
    noaa_weather_max_distance_km: float = 80.0


@dataclass(frozen=True)
class ForecastConfig:
    lookback: int = 168
    horizon: int = 24
    split_ratios: tuple[float, float, float] = (0.70, 0.15, 0.15)


@dataclass(frozen=True)
class RetrievalConfig:
    k: int = 8
    k_values: tuple[int, ...] = (1, 4, 8, 16)
    kb_stride_hours: int = 24
    block_size: int = 256
    epsilon: float = 1e-6
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "time_series": 0.5,
            "weather": 0.2,
            "calendar": 0.1,
            "event": 0.2,
        }
    )


@dataclass(frozen=True)
class DriftConfig:
    recent_window_hours: int = 24
    reference_window_hours: int = 168
    threshold_quantile: float = 0.90


@dataclass(frozen=True)
class ModelConfig:
    params: dict[str, Any]


@dataclass(frozen=True)
class EvaluationConfig:
    bootstrap_resamples: int = 500
    bootstrap_block_hours: int = 24


@dataclass(frozen=True)
class TSFMConfig:
    checkpoint: str = "amazon/chronos-bolt-small"
    batch_size: int = 64
    fusion_weights: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class ProjectConfig:
    root: Path
    name: str
    seed: int
    timezone: str
    paths: PathsConfig
    data: DataConfig
    forecast: ForecastConfig
    retrieval: RetrievalConfig
    drift: DriftConfig
    model: ModelConfig
    evaluation: EvaluationConfig
    tsfm: TSFMConfig

    def validate(self) -> None:
        if self.data.start_year > self.data.end_year:
            raise ValueError("data.start_year must not exceed data.end_year")
        if self.forecast.lookback < self.forecast.horizon:
            raise ValueError("forecast.lookback must be at least forecast.horizon")
        if abs(sum(self.forecast.split_ratios) - 1.0) > 1e-9:
            raise ValueError("forecast.split_ratios must sum to 1")
        if any(r <= 0 for r in self.forecast.split_ratios):
            raise ValueError("forecast.split_ratios must all be positive")
        if self.retrieval.k <= 0 or self.retrieval.k > max(self.retrieval.k_values):
            raise ValueError("retrieval.k must be positive and represented by k_values")
        if abs(sum(self.retrieval.weights.values()) - 1.0) > 1e-9:
            raise ValueError("retrieval weights must sum to 1")
        if not 0 < self.drift.threshold_quantile < 1:
            raise ValueError("drift.threshold_quantile must be between 0 and 1")


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_config(path: str | Path, project_root: str | Path | None = None) -> ProjectConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    root = Path(project_root).resolve() if project_root else config_path.parent.parent
    project = raw["project"]
    paths = raw["paths"]
    data = dict(raw["data"])
    data["epa_state_code"] = str(data["epa_state_code"]).zfill(2)
    data["epa_county_code"] = str(data["epa_county_code"]).zfill(3)
    forecast = dict(raw["forecast"])
    forecast["split_ratios"] = tuple(float(v) for v in forecast["split_ratios"])
    retrieval = dict(raw["retrieval"])
    retrieval["k_values"] = tuple(int(v) for v in retrieval["k_values"])
    tsfm = dict(raw["tsfm"])
    tsfm["fusion_weights"] = tuple(float(v) for v in tsfm["fusion_weights"])

    cfg = ProjectConfig(
        root=root,
        name=project["name"],
        seed=int(project["seed"]),
        timezone=project["timezone"],
        paths=PathsConfig(**{key: _resolve(root, value) for key, value in paths.items()}),
        data=DataConfig(**data),
        forecast=ForecastConfig(**forecast),
        retrieval=RetrievalConfig(**retrieval),
        drift=DriftConfig(**raw["drift"]),
        model=ModelConfig(params=dict(raw["model"])),
        evaluation=EvaluationConfig(**raw["evaluation"]),
        tsfm=TSFMConfig(**tsfm),
    )
    cfg.validate()
    cfg.paths.create()
    return cfg
