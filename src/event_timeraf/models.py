from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from .config import ProjectConfig
from .windows import WindowDataset


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
