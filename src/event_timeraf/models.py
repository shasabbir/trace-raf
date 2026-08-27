from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import ProjectConfig
from .windows import WindowDataset


def hour_month_climatology_forecast(
    train: WindowDataset,
    queries: WindowDataset,
    timezone: str,
) -> np.ndarray:
    """Forecast from training-only target-hour and target-month means."""
    horizon = train.y.shape[1]
    train_origins = pd.to_datetime(train.metadata["origin_time"], utc=True)
    query_origins = pd.to_datetime(queries.metadata["origin_time"], utc=True)
    records: list[pd.DataFrame] = []
    for step in range(1, horizon + 1):
        target_time = (train_origins + pd.Timedelta(hours=step)).dt.tz_convert(timezone)
        records.append(
            pd.DataFrame(
                {
                    "month": target_time.dt.month.to_numpy(),
                    "hour": target_time.dt.hour.to_numpy(),
                    "value": train.y[:, step - 1],
                }
            )
        )
    reference = pd.concat(records, ignore_index=True)
    group_means = reference.groupby(["month", "hour"])["value"].mean()
    hour_means = reference.groupby("hour")["value"].mean()
    global_mean = float(reference["value"].mean())
    prediction = np.empty((len(queries.x), horizon), dtype=np.float32)
    for step in range(1, horizon + 1):
        target_time = (query_origins + pd.Timedelta(hours=step)).dt.tz_convert(timezone)
        values = []
        for month, hour in zip(target_time.dt.month, target_time.dt.hour):
            values.append(group_means.get((month, hour), hour_means.get(hour, global_mean)))
        prediction[:, step - 1] = np.asarray(values, dtype=np.float32)
    return prediction


def persistence_forecast(x: np.ndarray, horizon: int) -> np.ndarray:
    return np.repeat(x[:, -1, None], horizon, axis=1).astype(np.float32)


def daily_seasonal_forecast(x: np.ndarray, horizon: int) -> np.ndarray:
    if x.shape[1] < 24 or horizon > 24:
        raise ValueError("Daily seasonal baseline requires lookback >= 24 and horizon <= 24")
    return x[:, -24:][:, :horizon].astype(np.float32)


def weekly_seasonal_forecast(x: np.ndarray, horizon: int) -> np.ndarray:
    if x.shape[1] < 168 or horizon > 24:
        raise ValueError("Weekly seasonal baseline requires lookback >= 168 and horizon <= 24")
    return x[:, :horizon].astype(np.float32)


def origin_feature_matrix(
    dataset: WindowDataset,
    prefixes: tuple[str, ...],
    extra: np.ndarray | None = None,
    extra_names: list[str] | tuple[str, ...] | None = None,
) -> tuple[np.ndarray, list[str]]:
    indices = [i for i, name in enumerate(dataset.feature_names) if name.startswith(prefixes)]
    if not indices:
        raise ValueError(f"No feature names match {prefixes}")
    matrix = dataset.features[:, indices]
    names = [dataset.feature_names[i] for i in indices]
    if extra is not None:
        if len(extra) != len(matrix):
            raise ValueError("Extra feature rows do not match dataset")
        if extra_names is not None and len(extra_names) != extra.shape[1]:
            raise ValueError("Extra feature names do not match extra feature columns")
        matrix = np.column_stack([matrix, extra])
        names.extend(extra_names or [f"extra_{i:03d}" for i in range(extra.shape[1])])
    return matrix.astype(np.float32), names


@dataclass
class DirectXGBForecaster:
    cfg: ProjectConfig
    include_future_calendar: bool = True

    def __post_init__(self) -> None:
        self.models: list = []
        self.feature_names: list[str] = []

    def _matrix(self, origin_features: np.ndarray, future_calendar: np.ndarray, horizon: int) -> np.ndarray:
        if self.include_future_calendar:
            return np.column_stack([origin_features, future_calendar[:, horizon, :]]).astype(np.float32)
        return origin_features.astype(np.float32)

    def fit(
        self,
        origin_features: np.ndarray,
        future_calendar: np.ndarray,
        targets: np.ndarray,
        feature_names: list[str] | None = None,
        calendar_names: tuple[str, ...] | None = None,
    ) -> "DirectXGBForecaster":
        from xgboost import XGBRegressor

        self.models = []
        params = dict(self.cfg.model.params)
        params.update(
            {
                "objective": "reg:squarederror",
                "tree_method": "hist",
                "random_state": self.cfg.seed,
            }
        )
        for horizon in range(targets.shape[1]):
            matrix = self._matrix(origin_features, future_calendar, horizon)
            model = XGBRegressor(**params)
            model.fit(matrix, targets[:, horizon], verbose=False)
            self.models.append(model)
        self.feature_names = list(feature_names or [f"feature_{i}" for i in range(origin_features.shape[1])])
        if self.include_future_calendar:
            self.feature_names.extend(calendar_names or [f"calendar_{i}" for i in range(future_calendar.shape[2])])
        return self

    def predict(self, origin_features: np.ndarray, future_calendar: np.ndarray) -> np.ndarray:
        if not self.models:
            raise RuntimeError("Model has not been fitted")
        predictions = [
            model.predict(self._matrix(origin_features, future_calendar, horizon))
            for horizon, model in enumerate(self.models)
        ]
        return np.column_stack(predictions).astype(np.float32)

    def feature_importance(self, horizon: int = 0) -> np.ndarray:
        return np.asarray(self.models[horizon].feature_importances_, dtype=float)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)


@dataclass
class DirectRidgeForecaster:
    """Regularized linear control using the same direct multi-horizon inputs."""

    cfg: ProjectConfig
    include_future_calendar: bool = True
    alpha: float = 10.0

    def __post_init__(self) -> None:
        self.models: list = []
        self.feature_names: list[str] = []

    def _matrix(self, origin_features: np.ndarray, future_calendar: np.ndarray, horizon: int) -> np.ndarray:
        if self.include_future_calendar:
            return np.column_stack([origin_features, future_calendar[:, horizon, :]]).astype(np.float32)
        return origin_features.astype(np.float32)

    def fit(
        self,
        origin_features: np.ndarray,
        future_calendar: np.ndarray,
        targets: np.ndarray,
        feature_names: list[str] | None = None,
        calendar_names: tuple[str, ...] | None = None,
    ) -> "DirectRidgeForecaster":
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self.models = []
        for horizon in range(targets.shape[1]):
            model = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha))
            model.fit(self._matrix(origin_features, future_calendar, horizon), targets[:, horizon])
            self.models.append(model)
        self.feature_names = list(feature_names or [f"feature_{i}" for i in range(origin_features.shape[1])])
        if self.include_future_calendar:
            self.feature_names.extend(calendar_names or [f"calendar_{i}" for i in range(future_calendar.shape[2])])
        return self

    def predict(self, origin_features: np.ndarray, future_calendar: np.ndarray) -> np.ndarray:
        if not self.models:
            raise RuntimeError("Model has not been fitted")
        return np.column_stack(
            [
                model.predict(self._matrix(origin_features, future_calendar, horizon))
                for horizon, model in enumerate(self.models)
            ]
        ).astype(np.float32)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)


@dataclass
class DirectLightGBMForecaster:
    """LightGBM control with the same direct multi-horizon design as XGBoost."""

    cfg: ProjectConfig
    include_future_calendar: bool = True

    def __post_init__(self) -> None:
        self.models: list = []
        self.feature_names: list[str] = []

    def _matrix(self, origin_features: np.ndarray, future_calendar: np.ndarray, horizon: int) -> np.ndarray:
        if self.include_future_calendar:
            return np.column_stack([origin_features, future_calendar[:, horizon, :]]).astype(np.float32)
        return origin_features.astype(np.float32)

    def fit(
        self,
        origin_features: np.ndarray,
        future_calendar: np.ndarray,
        targets: np.ndarray,
        feature_names: list[str] | None = None,
        calendar_names: tuple[str, ...] | None = None,
    ) -> "DirectLightGBMForecaster":
        from lightgbm import LGBMRegressor

        params = dict(self.cfg.model.params)
        params.update(
            {
                "objective": "regression",
                "random_state": self.cfg.seed,
                "verbosity": -1,
                "subsample_freq": 1,
                "deterministic": True,
                "force_col_wise": True,
            }
        )
        self.models = []
        for horizon in range(targets.shape[1]):
            model = LGBMRegressor(**params)
            model.fit(self._matrix(origin_features, future_calendar, horizon), targets[:, horizon])
            self.models.append(model)
        self.feature_names = list(feature_names or [f"feature_{i}" for i in range(origin_features.shape[1])])
        if self.include_future_calendar:
            self.feature_names.extend(calendar_names or [f"calendar_{i}" for i in range(future_calendar.shape[2])])
        return self

    def predict(self, origin_features: np.ndarray, future_calendar: np.ndarray) -> np.ndarray:
        if not self.models:
            raise RuntimeError("Model has not been fitted")
        return np.column_stack(
            [
                model.predict(self._matrix(origin_features, future_calendar, horizon))
                for horizon, model in enumerate(self.models)
            ]
        ).astype(np.float32)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)


@dataclass
class ConvexForecastEnsemble:
    """OOF-training-selected convex blend of two independently fitted forecasters."""

    first_name: str
    second_name: str

    def __post_init__(self) -> None:
        self.first_weight: float | None = None

    @staticmethod
    def _validate_shapes(actual: np.ndarray, first: np.ndarray, second: np.ndarray) -> None:
        if actual.shape != first.shape or actual.shape != second.shape:
            raise ValueError("Actual and component forecast arrays must have identical shapes")
        if actual.ndim != 2:
            raise ValueError("Forecast arrays must have shape [origins, horizon]")
        if not all(np.isfinite(values).all() for values in (actual, first, second)):
            raise ValueError("Convex ensemble inputs must be finite")

    def fit(
        self,
        actual: np.ndarray,
        first: np.ndarray,
        second: np.ndarray,
    ) -> "ConvexForecastEnsemble":
        self._validate_shapes(actual, first, second)
        direction = first - second
        denominator = float(np.sum(direction * direction))
        if denominator <= 1e-12:
            self.first_weight = 0.5
        else:
            numerator = float(np.sum((actual - second) * direction))
            self.first_weight = float(np.clip(numerator / denominator, 0.0, 1.0))
        return self

    def predict(self, first: np.ndarray, second: np.ndarray) -> np.ndarray:
        if self.first_weight is None:
            raise RuntimeError("Ensemble has not been fitted")
        if first.shape != second.shape or first.ndim != 2:
            raise ValueError("Component forecast arrays must have identical two-dimensional shapes")
        return (
            self.first_weight * first + (1.0 - self.first_weight) * second
        ).astype(np.float32)

    def weights(self) -> dict[str, float]:
        if self.first_weight is None:
            raise RuntimeError("Ensemble has not been fitted")
        return {
            self.first_name: self.first_weight,
            self.second_name: 1.0 - self.first_weight,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)


def residual_gate_features(
    base_prediction: np.ndarray,
    correction: np.ndarray,
    correction_spread: np.ndarray,
    mean_similarity: np.ndarray,
    max_similarity: np.ndarray,
    candidate_count: np.ndarray,
    eligible_candidate_count: np.ndarray,
    selected_event_fraction: np.ndarray,
    event_conditioning_applied: np.ndarray,
    component_disagreement: np.ndarray,
    drift_score: np.ndarray,
    drift_flag: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Build compact origin-level reliability features without target information."""
    arrays = (base_prediction, correction, correction_spread, component_disagreement)
    if any(values.ndim != 2 for values in arrays):
        raise ValueError("Forecast and correction inputs must be two-dimensional")
    if any(values.shape != base_prediction.shape for values in arrays[1:]):
        raise ValueError("Forecast and correction inputs must have identical shapes")
    n_rows = len(base_prediction)
    vectors = (
        mean_similarity,
        max_similarity,
        candidate_count,
        eligible_candidate_count,
        selected_event_fraction,
        event_conditioning_applied,
        drift_score,
        drift_flag,
    )
    if any(len(values) != n_rows for values in vectors):
        raise ValueError("Reliability vectors must match the number of forecast origins")
    matrix = np.column_stack(
        [
            base_prediction.mean(axis=1),
            base_prediction.std(axis=1),
            base_prediction.max(axis=1),
            correction.mean(axis=1),
            correction.std(axis=1),
            np.max(np.abs(correction), axis=1),
            correction_spread.mean(axis=1),
            correction_spread.max(axis=1),
            mean_similarity,
            max_similarity,
            np.log1p(candidate_count),
            np.log1p(eligible_candidate_count),
            selected_event_fraction,
            event_conditioning_applied.astype(float),
            component_disagreement.mean(axis=1),
            component_disagreement.max(axis=1),
            drift_score,
            drift_flag.astype(float),
        ]
    ).astype(np.float32)
    names = [
        "base_mean",
        "base_std",
        "base_max",
        "correction_mean",
        "correction_std",
        "correction_max_abs",
        "correction_spread_mean",
        "correction_spread_max",
        "retrieval_mean_similarity",
        "retrieval_max_similarity",
        "log_candidate_count",
        "log_eligible_candidate_count",
        "selected_event_fraction",
        "event_conditioning_applied",
        "base_component_disagreement_mean",
        "base_component_disagreement_max",
        "drift_score",
        "drift_flag",
    ]
    if not np.isfinite(matrix).all():
        raise ValueError("Residual gate features must be finite")
    return matrix, names


def apply_residual_correction(
    base_prediction: np.ndarray,
    correction: np.ndarray,
    strength: float,
) -> np.ndarray:
    """Apply a globally scaled residual analogue correction."""
    if base_prediction.shape != correction.shape or base_prediction.ndim != 2:
        raise ValueError("Base and correction arrays must be identically shaped matrices")
    if not np.isfinite(base_prediction).all() or not np.isfinite(correction).all():
        raise ValueError("Base and correction arrays must be finite")
    strength = float(strength)
    if not np.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError("Residual strength must be within [0, 1]")
    return (base_prediction + strength * correction).astype(np.float32)


def choose_residual_strength(
    actual: np.ndarray,
    base_prediction: np.ndarray,
    correction: np.ndarray,
    strengths: tuple[float, ...],
) -> tuple[float, dict[float, float]]:
    """Select one global residual strength using validation MSE only."""
    if actual.shape != base_prediction.shape or actual.shape != correction.shape:
        raise ValueError("Actual, base, and correction arrays must have identical shapes")
    if not strengths:
        raise ValueError("At least one residual strength is required")
    if not np.isfinite(actual).all():
        raise ValueError("Actual values must be finite")
    scores = {
        float(strength): float(
            np.mean(
                (
                    actual
                    - apply_residual_correction(
                        base_prediction, correction, float(strength)
                    )
                )
                ** 2
            )
        )
        for strength in strengths
    }
    return min(scores, key=scores.get), scores


def blend_base_with_analogue(
    base_prediction: np.ndarray,
    analogue_prediction: np.ndarray,
    analogue_weight: float,
) -> np.ndarray:
    """Convexly blend a supervised base forecast with a retrieved raw future."""
    if base_prediction.shape != analogue_prediction.shape or base_prediction.ndim != 2:
        raise ValueError("Base and analogue arrays must be identically shaped matrices")
    if not np.isfinite(base_prediction).all() or not np.isfinite(analogue_prediction).all():
        raise ValueError("Base and analogue arrays must be finite")
    analogue_weight = float(analogue_weight)
    if not np.isfinite(analogue_weight) or not 0.0 <= analogue_weight <= 1.0:
        raise ValueError("Analogue weight must be within [0, 1]")
    return (
        (1.0 - analogue_weight) * base_prediction
        + analogue_weight * analogue_prediction
    ).astype(np.float32)


def choose_analogue_weight(
    actual: np.ndarray,
    base_prediction: np.ndarray,
    analogue_prediction: np.ndarray,
    weights: tuple[float, ...],
) -> tuple[float, dict[float, float]]:
    """Select a raw-future fusion weight using validation MSE only."""
    if actual.shape != base_prediction.shape or actual.shape != analogue_prediction.shape:
        raise ValueError("Actual, base, and analogue arrays must have identical shapes")
    if not weights:
        raise ValueError("At least one analogue weight is required")
    if not np.isfinite(actual).all():
        raise ValueError("Actual values must be finite")
    scores = {
        float(weight): float(
            np.mean(
                (
                    actual
                    - blend_base_with_analogue(
                        base_prediction, analogue_prediction, float(weight)
                    )
                )
                ** 2
            )
        )
        for weight in weights
    }
    return min(scores, key=scores.get), scores


@dataclass
class SelectiveResidualGate:
    """TRACE-RAF gate trained on validation-only oracle correction utility."""

    cfg: ProjectConfig

    def __post_init__(self) -> None:
        self.model = None
        self.selected_strength: float | None = None
        self.selection_scores: dict[float, float] = {}
        self.feature_names: list[str] = []

    @staticmethod
    def _oracle_gate(
        actual: np.ndarray,
        base_prediction: np.ndarray,
        correction: np.ndarray,
    ) -> np.ndarray:
        residual = actual - base_prediction
        denominator = np.sum(correction * correction, axis=1)
        numerator = np.sum(residual * correction, axis=1)
        return np.clip(
            np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-12),
            0.0,
            1.0,
        ).astype(np.float32)

    def _new_model(self):
        from sklearn.ensemble import HistGradientBoostingRegressor

        settings = self.cfg.selective_residual
        return HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=settings.gate_learning_rate,
            max_iter=settings.gate_max_iter,
            max_depth=settings.gate_max_depth,
            l2_regularization=settings.gate_l2_regularization,
            random_state=self.cfg.seed,
        )

    def fit(
        self,
        actual: np.ndarray,
        base_prediction: np.ndarray,
        correction: np.ndarray,
        reliability_features: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> "SelectiveResidualGate":
        if actual.shape != base_prediction.shape or actual.shape != correction.shape:
            raise ValueError("Actual, base, and correction arrays must have identical shapes")
        if len(actual) != len(reliability_features) or reliability_features.ndim != 2:
            raise ValueError("Reliability features must align with forecast origins")
        if len(actual) < 20:
            raise ValueError("At least 20 validation origins are required for gate calibration")
        if not all(np.isfinite(values).all() for values in (
            actual, base_prediction, correction, reliability_features
        )):
            raise ValueError("Gate calibration inputs must be finite")
        split = int(len(actual) * self.cfg.selective_residual.gate_fit_fraction)
        split = min(max(split, 10), len(actual) - 10)
        target = self._oracle_gate(actual, base_prediction, correction)
        selection_model = self._new_model().fit(reliability_features[:split], target[:split])
        gate = np.clip(selection_model.predict(reliability_features[split:]), 0.0, 1.0)
        self.selection_scores = {}
        for strength in self.cfg.selective_residual.gate_strength_values:
            prediction = base_prediction[split:] + strength * gate[:, None] * correction[split:]
            self.selection_scores[float(strength)] = float(
                np.mean((actual[split:] - prediction) ** 2)
            )
        self.selected_strength = min(self.selection_scores, key=self.selection_scores.get)
        self.model = self._new_model().fit(reliability_features, target)
        self.feature_names = list(
            feature_names or [f"gate_feature_{index:02d}" for index in range(reliability_features.shape[1])]
        )
        if len(self.feature_names) != reliability_features.shape[1]:
            raise ValueError("Gate feature names do not match the reliability matrix")
        return self

    def gate_values(self, reliability_features: np.ndarray) -> np.ndarray:
        if self.model is None or self.selected_strength is None:
            raise RuntimeError("Selective residual gate has not been fitted")
        values = np.clip(self.model.predict(reliability_features), 0.0, 1.0)
        return (self.selected_strength * values).astype(np.float32)

    def predict(
        self,
        base_prediction: np.ndarray,
        correction: np.ndarray,
        reliability_features: np.ndarray,
    ) -> np.ndarray:
        if base_prediction.shape != correction.shape:
            raise ValueError("Base and correction arrays must have identical shapes")
        if len(base_prediction) != len(reliability_features):
            raise ValueError("Reliability features must align with forecast origins")
        gate = self.gate_values(reliability_features)
        return (base_prediction + gate[:, None] * correction).astype(np.float32)

    def save(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError("Selective residual gate has not been fitted")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)


class NeuralWindowForecaster:
    """Deterministic train/validation wrapper for univariate neural baselines."""

    def __init__(self, cfg: ProjectConfig, architecture: str):
        if architecture not in {"dlinear", "lstm", "patchtst"}:
            raise ValueError("architecture must be dlinear, lstm, or patchtst")
        self.cfg = cfg
        self.architecture = architecture
        self.model = None
        self.history: list[dict[str, float | int]] = []
        self.best_epoch: int | None = None

    def _build_model(self):
        import torch
        from torch import nn

        if self.architecture == "dlinear":
            lookback = self.cfg.forecast.lookback
            horizon = self.cfg.forecast.horizon
            kernel = self.cfg.baseline.dlinear_moving_average
            if kernel <= 0 or kernel % 2 == 0:
                raise ValueError("DLinear moving-average kernel must be a positive odd integer")

            class DLinearModule(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.kernel = kernel
                    self.seasonal = nn.Linear(lookback, horizon)
                    self.trend = nn.Linear(lookback, horizon)

                def forward(self, values):
                    padding = (self.kernel - 1) // 2
                    padded = torch.cat(
                        [
                            values[:, :1].repeat(1, padding),
                            values,
                            values[:, -1:].repeat(1, padding),
                        ],
                        dim=1,
                    )
                    trend = torch.nn.functional.avg_pool1d(
                        padded.unsqueeze(1), kernel_size=self.kernel, stride=1
                    ).squeeze(1)
                    return self.seasonal(values - trend) + self.trend(trend)

            return DLinearModule()

        if self.architecture == "lstm":
            horizon = self.cfg.forecast.horizon
            hidden_size = self.cfg.baseline.lstm_hidden_size
            layers = self.cfg.baseline.lstm_layers
            dropout = self.cfg.baseline.lstm_dropout if layers > 1 else 0.0

            class LSTMModule(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.encoder = nn.LSTM(
                        input_size=1,
                        hidden_size=hidden_size,
                        num_layers=layers,
                        dropout=dropout,
                        batch_first=True,
                    )
                    self.output = nn.Linear(hidden_size, horizon)

                def forward(self, values):
                    encoded, _ = self.encoder(values.unsqueeze(-1))
                    return self.output(encoded[:, -1])

            return LSTMModule()

        from transformers import PatchTSTConfig, PatchTSTForPrediction

        configuration = PatchTSTConfig(
            num_input_channels=1,
            context_length=self.cfg.forecast.lookback,
            prediction_length=self.cfg.forecast.horizon,
            patch_length=self.cfg.baseline.patch_length,
            patch_stride=self.cfg.baseline.patch_stride,
            num_hidden_layers=self.cfg.baseline.patch_layers,
            d_model=self.cfg.baseline.patch_d_model,
            num_attention_heads=self.cfg.baseline.patch_heads,
            ffn_dim=self.cfg.baseline.patch_ffn_dim,
            loss="mse",
            scaling=None,
        )
        patch_model = PatchTSTForPrediction(configuration)

        class PatchTSTAdapter(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, values):
                output = self.model(past_values=values.unsqueeze(-1)).prediction_outputs
                if output.ndim == 3 and output.shape[-1] == 1:
                    output = output[..., 0]
                if output.ndim != 2:
                    raise RuntimeError(f"Unexpected PatchTST prediction shape: {tuple(output.shape)}")
                return output

        return PatchTSTAdapter(patch_model)

    @staticmethod
    def _normalize(x: np.ndarray, y: np.ndarray | None = None):
        location = x.mean(axis=1, keepdims=True)
        scale = x.std(axis=1, keepdims=True)
        scale = np.maximum(scale, 1e-3)
        x_normalized = (x - location) / scale
        if y is None:
            return x_normalized.astype(np.float32), location.astype(np.float32), scale.astype(np.float32)
        y_normalized = (y - location) / scale
        return x_normalized.astype(np.float32), y_normalized.astype(np.float32)

    def fit(self, train: WindowDataset, validation: WindowDataset) -> "NeuralWindowForecaster":
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.cfg.seed)
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model().to(device)
        train_x, train_y = self._normalize(train.x, train.y)
        validation_x, validation_y = self._normalize(validation.x, validation.y)
        generator = torch.Generator().manual_seed(self.cfg.seed)
        training_loader = DataLoader(
            TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
            batch_size=self.cfg.baseline.neural_batch_size,
            shuffle=True,
            generator=generator,
        )
        validation_loader = DataLoader(
            TensorDataset(torch.from_numpy(validation_x), torch.from_numpy(validation_y)),
            batch_size=self.cfg.baseline.neural_batch_size,
            shuffle=False,
        )
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.baseline.neural_learning_rate,
            weight_decay=self.cfg.baseline.neural_weight_decay,
        )
        criterion = torch.nn.MSELoss()
        best_loss = np.inf
        best_state = None
        stale_epochs = 0
        self.history = []
        for epoch in range(1, self.cfg.baseline.neural_max_epochs + 1):
            self.model.train()
            train_loss_sum = 0.0
            train_points = 0
            for features, targets in training_loader:
                features, targets = features.to(device), targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(self.model(features), targets)
                loss.backward()
                optimizer.step()
                train_loss_sum += float(loss.detach()) * len(features)
                train_points += len(features)
            self.model.eval()
            validation_loss_sum = 0.0
            validation_points = 0
            with torch.no_grad():
                for features, targets in validation_loader:
                    features, targets = features.to(device), targets.to(device)
                    loss = criterion(self.model(features), targets)
                    validation_loss_sum += float(loss) * len(features)
                    validation_points += len(features)
            train_loss = train_loss_sum / train_points
            validation_loss = validation_loss_sum / validation_points
            self.history.append(
                {"epoch": epoch, "train_normalized_mse": train_loss, "validation_normalized_mse": validation_loss}
            )
            if validation_loss < best_loss - 1e-6:
                best_loss = validation_loss
                best_state = {name: value.detach().cpu().clone() for name, value in self.model.state_dict().items()}
                self.best_epoch = epoch
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.cfg.baseline.neural_patience:
                    break
        if best_state is None:
            raise RuntimeError(f"{self.architecture} training did not produce a valid checkpoint")
        self.model.load_state_dict(best_state)
        self.model.to(device)
        return self

    def predict(self, dataset: WindowDataset) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fitted")
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        device = next(self.model.parameters()).device
        normalized, location, scale = self._normalize(dataset.x)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(normalized)),
            batch_size=self.cfg.baseline.neural_batch_size,
            shuffle=False,
        )
        values = []
        self.model.eval()
        with torch.no_grad():
            for (features,) in loader:
                values.append(self.model(features.to(device)).cpu().numpy())
        prediction = np.concatenate(values).astype(np.float32)
        return (prediction * scale + location).astype(np.float32)

    def history_frame(self, run_id: str = "unassigned") -> pd.DataFrame:
        frame = pd.DataFrame(self.history)
        frame.insert(0, "architecture", self.architecture)
        frame.insert(0, "run_id", run_id)
        frame["selected_epoch"] = frame["epoch"].eq(self.best_epoch)
        return frame

    def save(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError("Model has not been fitted")
        import torch

        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "architecture": self.architecture,
                "state_dict": {name: value.detach().cpu() for name, value in self.model.state_dict().items()},
                "best_epoch": self.best_epoch,
                "history": self.history,
            },
            path,
        )


def chronos_forecast(
    x: np.ndarray,
    horizon: int,
    checkpoint: str,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run a frozen Chronos/Chronos-Bolt checkpoint on fixed lookback windows."""
    import torch
    from chronos import BaseChronosPipeline

    use_cuda = torch.cuda.is_available()
    supports_bfloat16 = bool(
        use_cuda
        and hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    )
    torch.manual_seed(0)
    if use_cuda:
        torch.cuda.manual_seed_all(0)
    pipeline = BaseChronosPipeline.from_pretrained(
        checkpoint,
        device_map="cuda" if use_cuda else "cpu",
        torch_dtype=torch.bfloat16 if supports_bfloat16 else torch.float32,
    )
    means: list[np.ndarray] = []
    lower: list[np.ndarray] = []
    upper: list[np.ndarray] = []
    for start in range(0, len(x), batch_size):
        context = torch.as_tensor(x[start : start + batch_size], dtype=torch.float32)
        quantiles, mean = pipeline.predict_quantiles(
            context,
            prediction_length=horizon,
            quantile_levels=[0.1, 0.5, 0.9],
        )
        quantile_array = quantiles.detach().cpu().numpy() if hasattr(quantiles, "detach") else np.asarray(quantiles)
        mean_array = mean.detach().cpu().numpy() if hasattr(mean, "detach") else np.asarray(mean)
        means.append(mean_array)
        lower.append(quantile_array[:, :, 0])
        upper.append(quantile_array[:, :, 2])
    return (
        np.concatenate(means).astype(np.float32),
        np.concatenate(lower).astype(np.float32),
        np.concatenate(upper).astype(np.float32),
    )


def chronos_quantile_forecast(
    x: np.ndarray,
    horizon: int,
    checkpoint: str,
    quantile_levels: tuple[float, ...],
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Run Chronos once and retain a dense quantile grid for probabilistic evaluation."""
    import torch
    from chronos import BaseChronosPipeline

    if not quantile_levels or any(not 0 < value < 1 for value in quantile_levels):
        raise ValueError("quantile_levels must be non-empty and strictly within (0, 1)")
    if tuple(sorted(quantile_levels)) != tuple(quantile_levels):
        raise ValueError("quantile_levels must be sorted")
    if min(quantile_levels) < 0.1 or max(quantile_levels) > 0.9:
        raise ValueError("Chronos-Bolt quantile levels must stay within its native [0.1, 0.9] grid")
    use_cuda = torch.cuda.is_available()
    supports_bfloat16 = bool(
        use_cuda and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
    )
    torch.manual_seed(0)
    if use_cuda:
        torch.cuda.manual_seed_all(0)
    pipeline = BaseChronosPipeline.from_pretrained(
        checkpoint,
        device_map="cuda" if use_cuda else "cpu",
        torch_dtype=torch.bfloat16 if supports_bfloat16 else torch.float32,
    )
    means: list[np.ndarray] = []
    quantiles: list[np.ndarray] = []
    for start in range(0, len(x), batch_size):
        context = torch.as_tensor(x[start : start + batch_size], dtype=torch.float32)
        quantile_values, mean = pipeline.predict_quantiles(
            context,
            prediction_length=horizon,
            quantile_levels=list(quantile_levels),
        )
        quantile_array = (
            quantile_values.detach().cpu().numpy()
            if hasattr(quantile_values, "detach")
            else np.asarray(quantile_values)
        )
        mean_array = mean.detach().cpu().numpy() if hasattr(mean, "detach") else np.asarray(mean)
        means.append(mean_array)
        quantiles.append(quantile_array)
    return np.concatenate(means).astype(np.float32), np.concatenate(quantiles).astype(np.float32)


def fuse_forecasts(tsfm: np.ndarray, retrieved: np.ndarray, tsfm_weight: float) -> np.ndarray:
    if tsfm.shape != retrieved.shape:
        raise ValueError("Forecast arrays must have identical shapes")
    return (tsfm_weight * tsfm + (1.0 - tsfm_weight) * retrieved).astype(np.float32)


def choose_fusion_weight(
    actual: np.ndarray,
    tsfm: np.ndarray,
    retrieved: np.ndarray,
    weights: tuple[float, ...],
) -> tuple[float, dict[float, float]]:
    scores = {
        float(weight): float(np.mean((actual - fuse_forecasts(tsfm, retrieved, float(weight))) ** 2))
        for weight in weights
    }
    selected = min(scores, key=scores.get)
    return selected, scores
