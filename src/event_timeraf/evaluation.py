from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def _smape(actual: np.ndarray, predicted: np.ndarray, epsilon: float = 1e-6) -> float:
    denominator = np.abs(actual) + np.abs(predicted) + epsilon
    return float(200 * np.mean(np.abs(actual - predicted) / denominator))


def _mape(actual: np.ndarray, predicted: np.ndarray, epsilon: float = 1e-3) -> float:
    return float(100 * np.mean(np.abs(actual - predicted) / np.maximum(np.abs(actual), epsilon)))


def metric_values(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual_flat = np.asarray(actual, dtype=float).ravel()
    predicted_flat = np.asarray(predicted, dtype=float).ravel()
    valid = np.isfinite(actual_flat) & np.isfinite(predicted_flat)
    if not valid.any():
        return {name: np.nan for name in ("mse", "mae", "rmse", "mape", "smape", "r2")}
    actual_flat = actual_flat[valid]
    predicted_flat = predicted_flat[valid]
    residual = actual_flat - predicted_flat
    mse = float(np.mean(residual**2))
    denominator = float(np.sum((actual_flat - actual_flat.mean()) ** 2))
    r2 = 1.0 - float(np.sum(residual**2)) / denominator if denominator > 0 else np.nan
    return {
        "mse": mse,
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(mse)),
        "mape": _mape(actual_flat, predicted_flat),
        "smape": _smape(actual_flat, predicted_flat),
        "r2": r2,
    }


def metrics_table(
    actual: np.ndarray,
    predicted: np.ndarray,
    model: str,
    subset: str = "all",
) -> pd.DataFrame:
    rows = [{"model": model, "subset": subset, "horizon": "overall", **metric_values(actual, predicted)}]
    for horizon in range(actual.shape[1]):
        rows.append(
            {
                "model": model,
                "subset": subset,
                "horizon": horizon + 1,
                **metric_values(actual[:, horizon], predicted[:, horizon]),
            }
        )
    horizon_frame = pd.DataFrame(rows[1:])
    macro = {metric: float(horizon_frame[metric].mean()) for metric in ("mse", "mae", "rmse", "mape", "smape", "r2")}
    rows.append({"model": model, "subset": subset, "horizon": "macro", **macro})
    return pd.DataFrame(rows)


def predictions_long(
    actual: np.ndarray,
    predicted: np.ndarray,
    metadata: pd.DataFrame,
    model: str,
    drift_flag: np.ndarray | None = None,
    event_flag: np.ndarray | None = None,
) -> pd.DataFrame:
    horizon = actual.shape[1]
    origins = pd.to_datetime(metadata["origin_time"], utc=True).repeat(horizon).reset_index(drop=True)
    steps = np.tile(np.arange(1, horizon + 1), len(metadata))
    target_time = origins + pd.to_timedelta(steps, unit="h")
    return pd.DataFrame(
        {
            "model": model,
            "window_id": metadata["window_id"].repeat(horizon).to_numpy(),
            "origin_time": origins,
            "horizon": steps,
            "target_time": target_time,
            "actual": actual.reshape(-1),
            "prediction": predicted.reshape(-1),
            "split": metadata["split"].repeat(horizon).to_numpy(),
            "drift_flag": np.repeat(drift_flag, horizon) if drift_flag is not None else False,
            "event_flag": np.repeat(event_flag, horizon) if event_flag is not None else False,
        }
    )


def paired_block_bootstrap_difference(
    actual: np.ndarray,
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    block_length: int,
    resamples: int,
    seed: int,
) -> dict[str, float]:
    if not (actual.shape == prediction_a.shape == prediction_b.shape):
        raise ValueError("Actual and prediction arrays must have identical shapes")
    rng = np.random.default_rng(seed)
    n = len(actual)
    differences = np.empty(resamples, dtype=float)
    blocks_needed = int(np.ceil(n / block_length))
    for sample in range(resamples):
        starts = rng.integers(0, n, size=blocks_needed)
        indices = np.concatenate([(start + np.arange(block_length)) % n for start in starts])[:n]
        differences[sample] = metric(actual[indices], prediction_a[indices]) - metric(actual[indices], prediction_b[indices])
    return {
        "difference": float(metric(actual, prediction_a) - metric(actual, prediction_b)),
        "ci_low": float(np.quantile(differences, 0.025)),
        "ci_high": float(np.quantile(differences, 0.975)),
    }
