from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
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
    study_end_date: str | None = None
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
    kb_stride_values: tuple[int, ...] = (192, 24, 6)
    event_weight_values: tuple[float, ...] = (0.0, 0.2, 0.5, 0.8, 1.0)
    event_context_feature: str = "event_count_72h"
    event_stratified_fallback: str = "all_eligible"
    normalize_event_scores: bool = True
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
    score_mode: str = "two_sided"
    ks_reference_size: int = 4096


@dataclass(frozen=True)
class ModelConfig:
    params: dict[str, Any]


@dataclass(frozen=True)
class EvaluationConfig:
    bootstrap_resamples: int = 5000
    bootstrap_block_hours: int = 168
    dm_hac_lags: int = 168
    minimum_subset_origins: int = 50
    aqi_thresholds: tuple[float, ...] = (35.4, 55.4)
    probabilistic_quantiles: tuple[float, ...] = (
        0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90,
    )


@dataclass(frozen=True)
class BaselineConfig:
    neural_batch_size: int = 256
    neural_max_epochs: int = 30
    neural_patience: int = 5
    neural_learning_rate: float = 1e-3
    neural_weight_decay: float = 1e-4
    dlinear_moving_average: int = 25
    lstm_hidden_size: int = 64
    lstm_layers: int = 2
    lstm_dropout: float = 0.1
    patch_length: int = 24
    patch_stride: int = 12
    patch_layers: int = 3
    patch_d_model: int = 64
    patch_heads: int = 4
    patch_ffn_dim: int = 128


@dataclass(frozen=True)
class TSFMConfig:
    checkpoint: str = "amazon/chronos-bolt-small"
    batch_size: int = 64
    fusion_weights: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class SelectiveResidualConfig:
    oof_boundaries: tuple[float, ...] = (0.50, 0.75, 1.0)
    gate_fit_fraction: float = 0.67
    gate_strength_values: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    gate_max_depth: int = 2
    gate_max_iter: int = 100
    gate_learning_rate: float = 0.05
    gate_l2_regularization: float = 1.0
    residual_clip_quantile: float = 0.01


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
    baseline: BaselineConfig
    tsfm: TSFMConfig
    selective_residual: SelectiveResidualConfig

    def validate(self) -> None:
        if self.data.start_year > self.data.end_year:
            raise ValueError("data.start_year must not exceed data.end_year")
        if self.data.study_end_date is not None:
            try:
                study_end = date.fromisoformat(self.data.study_end_date)
            except ValueError as error:
                raise ValueError("data.study_end_date must use YYYY-MM-DD format") from error
            if study_end.year != self.data.end_year:
                raise ValueError("data.study_end_date must fall within data.end_year")
        if self.forecast.lookback < self.forecast.horizon:
            raise ValueError("forecast.lookback must be at least forecast.horizon")
        if abs(sum(self.forecast.split_ratios) - 1.0) > 1e-9:
            raise ValueError("forecast.split_ratios must sum to 1")
        if any(r <= 0 for r in self.forecast.split_ratios):
            raise ValueError("forecast.split_ratios must all be positive")
        if self.retrieval.k <= 0 or self.retrieval.k > max(self.retrieval.k_values):
            raise ValueError("retrieval.k must be positive and represented by k_values")
        if self.retrieval.kb_stride_hours <= 0:
            raise ValueError("retrieval.kb_stride_hours must be positive")
        if self.retrieval.kb_stride_hours not in self.retrieval.kb_stride_values:
            raise ValueError("retrieval.kb_stride_hours must be represented by kb_stride_values")
        if any(value <= 0 for value in self.retrieval.kb_stride_values):
            raise ValueError("retrieval.kb_stride_values must all be positive")
        if any(not 0 <= value <= 1 for value in self.retrieval.event_weight_values):
            raise ValueError("retrieval.event_weight_values must be within [0, 1]")
        if self.retrieval.event_stratified_fallback not in {"all_eligible", "no_result"}:
            raise ValueError("Unsupported retrieval.event_stratified_fallback")
        if abs(sum(self.retrieval.weights.values()) - 1.0) > 1e-9:
            raise ValueError("retrieval weights must sum to 1")
        if not 0 < self.drift.threshold_quantile < 1:
            raise ValueError("drift.threshold_quantile must be between 0 and 1")
        if self.drift.score_mode not in {"upper_tail", "two_sided"}:
            raise ValueError("drift.score_mode must be upper_tail or two_sided")
        if self.drift.ks_reference_size <= 0:
            raise ValueError("drift.ks_reference_size must be positive")
        if self.evaluation.minimum_subset_origins <= 0:
            raise ValueError("evaluation.minimum_subset_origins must be positive")
        if self.evaluation.bootstrap_resamples < 2000:
            raise ValueError("evaluation.bootstrap_resamples must be at least 2000 for publication runs")
        if self.evaluation.bootstrap_block_hours < 168:
            raise ValueError("evaluation.bootstrap_block_hours must be at least 168")
        if self.evaluation.dm_hac_lags <= 0:
            raise ValueError("evaluation.dm_hac_lags must be positive")
        if any(value <= 0 for value in self.evaluation.aqi_thresholds):
            raise ValueError("evaluation.aqi_thresholds must all be positive")
        quantiles = self.evaluation.probabilistic_quantiles
        if any(not 0 < value < 1 for value in quantiles) or tuple(sorted(set(quantiles))) != quantiles:
            raise ValueError("evaluation.probabilistic_quantiles must be unique, sorted, and within (0, 1)")
        if 0.1 not in quantiles or 0.9 not in quantiles:
            raise ValueError("evaluation.probabilistic_quantiles must include 0.1 and 0.9")
        if min(quantiles) < 0.1 or max(quantiles) > 0.9:
            raise ValueError(
                "evaluation.probabilistic_quantiles must stay within the Chronos-Bolt "
                "native [0.1, 0.9] grid"
            )
        if self.baseline.neural_batch_size <= 0 or self.baseline.neural_max_epochs <= 0:
            raise ValueError("Neural baseline batch size and epochs must be positive")
        if self.baseline.neural_patience <= 0:
            raise ValueError("Neural baseline patience must be positive")
        if self.baseline.neural_learning_rate <= 0 or self.baseline.neural_weight_decay < 0:
            raise ValueError("Neural baseline optimizer settings are invalid")
        if (
            self.baseline.dlinear_moving_average <= 0
            or self.baseline.dlinear_moving_average % 2 == 0
        ):
            raise ValueError("baseline.dlinear_moving_average must be a positive odd integer")
        if self.baseline.lstm_hidden_size <= 0 or self.baseline.lstm_layers <= 0:
            raise ValueError("baseline LSTM dimensions must be positive")
        if not 0 <= self.baseline.lstm_dropout < 1:
            raise ValueError("baseline.lstm_dropout must be in [0, 1)")
        patch_values = (
            self.baseline.patch_length,
            self.baseline.patch_stride,
            self.baseline.patch_layers,
            self.baseline.patch_d_model,
            self.baseline.patch_heads,
            self.baseline.patch_ffn_dim,
        )
        if any(value <= 0 for value in patch_values):
            raise ValueError("PatchTST dimensions must be positive")
        if self.baseline.patch_d_model % self.baseline.patch_heads:
            raise ValueError("baseline.patch_d_model must be divisible by baseline.patch_heads")
        if self.baseline.patch_length > self.forecast.lookback:
            raise ValueError("baseline.patch_length must not exceed the forecast lookback")
        boundaries = self.selective_residual.oof_boundaries
        if (
            len(boundaries) < 2
            or tuple(sorted(set(boundaries))) != boundaries
            or not 0 < boundaries[0] < 1
            or boundaries[-1] != 1.0
        ):
            raise ValueError(
                "selective_residual.oof_boundaries must be unique, increasing, and end at 1.0"
            )
        if not 0 < self.selective_residual.gate_fit_fraction < 1:
            raise ValueError("selective_residual.gate_fit_fraction must be within (0, 1)")
        strengths = self.selective_residual.gate_strength_values
        if (
            tuple(sorted(set(strengths))) != strengths
            or not strengths
            or strengths[0] != 0.0
            or any(not 0 <= value <= 1 for value in strengths)
        ):
            raise ValueError(
                "selective_residual.gate_strength_values must be unique, sorted, include 0, "
                "and stay within [0, 1]"
            )
        if self.selective_residual.gate_max_depth <= 0 or self.selective_residual.gate_max_iter <= 0:
            raise ValueError("selective residual gate dimensions must be positive")
        if (
            self.selective_residual.gate_learning_rate <= 0
            or self.selective_residual.gate_l2_regularization < 0
        ):
            raise ValueError("selective residual gate optimizer settings are invalid")
        if not 0 <= self.selective_residual.residual_clip_quantile < 0.5:
            raise ValueError("selective_residual.residual_clip_quantile must be within [0, 0.5)")


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
    retrieval["kb_stride_values"] = tuple(
        int(v) for v in retrieval.get("kb_stride_values", [retrieval["kb_stride_hours"]])
    )
    retrieval["event_weight_values"] = tuple(
        float(v) for v in retrieval.get("event_weight_values", [retrieval["weights"]["event"]])
    )
    tsfm = dict(raw["tsfm"])
    tsfm["fusion_weights"] = tuple(float(v) for v in tsfm["fusion_weights"])
    evaluation = dict(raw["evaluation"])
    evaluation["aqi_thresholds"] = tuple(
        float(v) for v in evaluation.get("aqi_thresholds", (35.4, 55.4))
    )
    evaluation["probabilistic_quantiles"] = tuple(
        float(v) for v in evaluation.get("probabilistic_quantiles", EvaluationConfig().probabilistic_quantiles)
    )
    selective_residual = dict(raw.get("selective_residual", {}))
    selective_residual["oof_boundaries"] = tuple(
        float(v) for v in selective_residual.get("oof_boundaries", SelectiveResidualConfig().oof_boundaries)
    )
    selective_residual["gate_strength_values"] = tuple(
        float(v)
        for v in selective_residual.get(
            "gate_strength_values", SelectiveResidualConfig().gate_strength_values
        )
    )

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
        evaluation=EvaluationConfig(**evaluation),
        baseline=BaselineConfig(**raw.get("baseline", {})),
        tsfm=TSFMConfig(**tsfm),
        selective_residual=SelectiveResidualConfig(**selective_residual),
    )
    cfg.validate()
    cfg.paths.create()
    return cfg
