from __future__ import annotations

from collections.abc import Callable
from math import erfc, sqrt

import numpy as np
import pandas as pd


EVENT_FLAG_COLUMNS = (
    "recent_event_flag",
    "active_event_flag",
    "target_event_flag",
)


def build_event_period_flags(
    metadata: pd.DataFrame,
    events: pd.DataFrame,
    recent_hours: int = 72,
) -> pd.DataFrame:
    """Build causal context flags and a post-hoc target-overlap label."""
    output = pd.DataFrame(False, index=np.arange(len(metadata)), columns=EVENT_FLAG_COLUMNS)
    if events.empty:
        return output

    event_start = pd.to_datetime(events["event_time"], errors="coerce", utc=True)
    event_end = pd.to_datetime(events.get("event_end", event_start), errors="coerce", utc=True).fillna(event_start)
    published = pd.to_datetime(events["published_at"], errors="coerce", utc=True)
    valid = event_start.notna() & event_end.notna() & published.notna()
    event_start = event_start[valid].to_numpy()
    event_end = event_end[valid].to_numpy()
    published = published[valid].to_numpy()
    if len(event_start) == 0:
        return output

    origins = pd.to_datetime(metadata["origin_time"], utc=True)
    target_starts = pd.to_datetime(metadata["target_start"], utc=True)
    target_ends = pd.to_datetime(metadata["target_end"], utc=True)
    for index, (origin, target_start, target_end) in enumerate(zip(origins, target_starts, target_ends)):
        recent_start = origin - pd.Timedelta(hours=recent_hours)
        output.loc[index, "recent_event_flag"] = bool(((published > recent_start) & (published <= origin)).any())
        output.loc[index, "active_event_flag"] = bool(
            ((event_start <= origin) & (event_end >= origin) & (published <= origin)).any()
        )
        # This is a post-hoc evaluation label. It must never enter model inputs.
        output.loc[index, "target_event_flag"] = bool(
            ((event_start <= target_end) & (event_end >= target_start)).any()
        )
    return output


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
    run_id: str = "unassigned",
    event_availability_mode: str = "not_applicable",
) -> pd.DataFrame:
    valid_points = np.isfinite(actual) & np.isfinite(predicted)
    metadata = {
        "run_id": run_id,
        "model": model,
        "subset": subset,
        "event_availability_mode": event_availability_mode,
        "n_origins": int(len(actual)),
        "n_points": int(valid_points.sum()),
    }
    rows = [{**metadata, "horizon": "overall", **metric_values(actual, predicted)}]
    for horizon in range(actual.shape[1]):
        rows.append(
            {
                **metadata,
                "horizon": horizon + 1,
                **metric_values(actual[:, horizon], predicted[:, horizon]),
            }
        )
    horizon_frame = pd.DataFrame(rows[1:])
    macro = {metric: float(horizon_frame[metric].mean()) for metric in ("mse", "mae", "rmse", "mape", "smape", "r2")}
    rows.append({**metadata, "horizon": "macro", **macro})
    return pd.DataFrame(rows)


def predictions_long(
    actual: np.ndarray,
    predicted: np.ndarray,
    metadata: pd.DataFrame,
    model: str,
    seed: int,
    run_id: str,
    drift_flag: np.ndarray | None = None,
    drift_score: np.ndarray | None = None,
    event_flags: pd.DataFrame | None = None,
    event_availability_mode: str = "not_applicable",
) -> pd.DataFrame:
    horizon = actual.shape[1]
    origins = pd.to_datetime(metadata["origin_time"], utc=True).repeat(horizon).reset_index(drop=True)
    steps = np.tile(np.arange(1, horizon + 1), len(metadata))
    target_time = origins + pd.to_timedelta(steps, unit="h")
    if event_flags is None:
        event_flags = pd.DataFrame(False, index=np.arange(len(metadata)), columns=EVENT_FLAG_COLUMNS)
    missing_flags = set(EVENT_FLAG_COLUMNS) - set(event_flags.columns)
    if missing_flags:
        raise ValueError(f"Event flags are missing columns: {sorted(missing_flags)}")
    if len(event_flags) != len(metadata):
        raise ValueError("Event flags and metadata must have the same number of rows")
    result = pd.DataFrame(
        {
            "run_id": run_id,
            "seed": int(seed),
            "model": model,
            "window_id": metadata["window_id"].repeat(horizon).to_numpy(),
            "origin_time": origins,
            "horizon": steps,
            "target_time": target_time,
            "actual": actual.reshape(-1),
            "prediction": predicted.reshape(-1),
            "split": metadata["split"].repeat(horizon).to_numpy(),
            "drift_flag": np.repeat(drift_flag, horizon) if drift_flag is not None else False,
            "drift_score": np.repeat(drift_score, horizon) if drift_score is not None else np.nan,
            "event_availability_mode": event_availability_mode,
        }
    )
    for column in EVENT_FLAG_COLUMNS:
        result[column] = np.repeat(event_flags[column].to_numpy(dtype=bool), horizon)
    # Backward-compatible name used by result tables: event means target overlap.
    result["event_flag"] = result["target_event_flag"]
    return result


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


def _origin_loss_difference(
    actual: np.ndarray,
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
    loss: str,
) -> np.ndarray:
    if not (actual.shape == prediction_a.shape == prediction_b.shape):
        raise ValueError("Actual and prediction arrays must have identical shapes")
    if actual.ndim != 2:
        raise ValueError("Forecast arrays must have shape [origins, horizon]")
    valid = np.isfinite(actual) & np.isfinite(prediction_a) & np.isfinite(prediction_b)
    if not valid.all():
        raise ValueError("Forecast-comparison inputs must contain only finite paired values")
    if loss == "mse":
        loss_a = (actual - prediction_a) ** 2
        loss_b = (actual - prediction_b) ** 2
    elif loss == "mae":
        loss_a = np.abs(actual - prediction_a)
        loss_b = np.abs(actual - prediction_b)
    else:
        raise ValueError("loss must be 'mse' or 'mae'")
    return np.mean(loss_a - loss_b, axis=1, dtype=np.float64)


def paired_block_bootstrap_loss_difference(
    actual: np.ndarray,
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
    loss: str,
    block_length: int,
    resamples: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    """Efficient paired moving-block interval for an origin-level loss difference."""
    differential = _origin_loss_difference(actual, prediction_a, prediction_b, loss)
    n = len(differential)
    if not 1 <= block_length <= n:
        raise ValueError("block_length must be between one and the number of origins")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")

    blocks_needed = int(np.ceil(n / block_length))
    remainder = n - (blocks_needed - 1) * block_length
    circular = np.concatenate([differential, differential[: block_length - 1]])
    cumulative = np.concatenate([[0.0], np.cumsum(circular, dtype=np.float64)])
    full_sums = cumulative[block_length : block_length + n] - cumulative[:n]
    partial_sums = cumulative[remainder : remainder + n] - cumulative[:n]

    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(resamples, blocks_needed))
    sampled_sums = full_sums[starts[:, :-1]].sum(axis=1) if blocks_needed > 1 else 0.0
    sampled_sums = sampled_sums + partial_sums[starts[:, -1]]
    sampled_differences = sampled_sums / n
    observed = float(differential.mean())
    alpha = 1.0 - confidence
    centered = sampled_differences - sampled_differences.mean()
    p_value = (1.0 + np.count_nonzero(np.abs(centered) >= abs(observed))) / (resamples + 1.0)
    return {
        "difference": observed,
        "ci_low": float(np.quantile(sampled_differences, alpha / 2.0)),
        "ci_high": float(np.quantile(sampled_differences, 1.0 - alpha / 2.0)),
        "bootstrap_p_value": float(min(1.0, p_value)),
        "block_length": int(block_length),
        "resamples": int(resamples),
    }


def diebold_mariano_hac(
    actual: np.ndarray,
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
    loss: str,
    hac_lags: int,
) -> dict[str, float | int]:
    """Two-sided Diebold-Mariano test with a Bartlett/Newey-West HAC variance."""
    differential = _origin_loss_difference(actual, prediction_a, prediction_b, loss)
    n = len(differential)
    lags = min(int(hac_lags), n - 1)
    if lags < 0:
        raise ValueError("At least one forecast origin is required")
    centered = differential - differential.mean()
    long_run_variance = float(np.dot(centered, centered) / n)
    for lag in range(1, lags + 1):
        covariance = float(np.dot(centered[lag:], centered[:-lag]) / n)
        long_run_variance += 2.0 * (1.0 - lag / (lags + 1.0)) * covariance
    variance_of_mean = max(long_run_variance / n, 0.0)
    if variance_of_mean == 0.0:
        statistic = 0.0 if differential.mean() == 0.0 else np.sign(differential.mean()) * np.inf
    else:
        statistic = float(differential.mean() / np.sqrt(variance_of_mean))
    p_value = float(erfc(abs(statistic) / sqrt(2.0)))
    return {
        "dm_statistic": statistic,
        "dm_p_value": p_value,
        "dm_hac_lags": int(lags),
    }


def holm_adjust_pvalues(p_values: np.ndarray | list[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("p_values must be a finite one-dimensional array within [0, 1]")
    order = np.argsort(values)
    adjusted_sorted = np.maximum.accumulate((len(values) - np.arange(len(values))) * values[order])
    adjusted = np.empty_like(values)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


def horizon_skill_table(
    actual: np.ndarray,
    predicted: np.ndarray,
    climatology: np.ndarray,
    model: str,
    run_id: str = "unassigned",
) -> pd.DataFrame:
    if not (actual.shape == predicted.shape == climatology.shape):
        raise ValueError("Actual, model, and climatology arrays must have identical shapes")
    rows = []
    for horizon in range(actual.shape[1]):
        model_mse = metric_values(actual[:, horizon], predicted[:, horizon])["mse"]
        reference_mse = metric_values(actual[:, horizon], climatology[:, horizon])["mse"]
        rows.append(
            {
                "run_id": run_id,
                "model": model,
                "horizon": horizon + 1,
                "model_mse": model_mse,
                "climatology_mse": reference_mse,
                "skill_vs_climatology": 1.0 - model_mse / reference_mse if reference_mse > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def exceedance_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    thresholds: tuple[float, ...] | list[float],
    model: str,
    run_id: str = "unassigned",
) -> pd.DataFrame:
    """Classify each origin by its observed and forecast 24-hour mean PM2.5."""
    if actual.shape != predicted.shape or actual.ndim != 2:
        raise ValueError("Actual and prediction arrays must share shape [origins, horizon]")
    actual_mean = np.mean(actual, axis=1)
    predicted_mean = np.mean(predicted, axis=1)
    rows = []
    for threshold in thresholds:
        observed = actual_mean > threshold
        forecast = predicted_mean > threshold
        tp = int(np.sum(observed & forecast))
        fp = int(np.sum(~observed & forecast))
        fn = int(np.sum(observed & ~forecast))
        tn = int(np.sum(~observed & ~forecast))
        precision = tp / (tp + fp) if tp + fp else np.nan
        recall = tp / (tp + fn) if tp + fn else np.nan
        specificity = tn / (tn + fp) if tn + fp else np.nan
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else np.nan
        csi = tp / (tp + fp + fn) if tp + fp + fn else np.nan
        if observed.any() and (~observed).any():
            from sklearn.metrics import average_precision_score, roc_auc_score

            auroc = float(roc_auc_score(observed, predicted_mean))
            average_precision = float(average_precision_score(observed, predicted_mean))
        else:
            auroc = np.nan
            average_precision = np.nan
        rows.append(
            {
                "run_id": run_id,
                "model": model,
                "threshold_ug_m3": float(threshold),
                "aggregation": f"{actual.shape[1]}h_forecast_mean",
                "n_origins": len(actual),
                "observed_exceedances": int(observed.sum()),
                "forecast_exceedances": int(forecast.sum()),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "critical_success_index": csi,
                "specificity": specificity,
                "balanced_accuracy": (
                    (recall + specificity) / 2.0
                    if np.isfinite(recall) and np.isfinite(specificity)
                    else np.nan
                ),
                "auroc": auroc,
                "average_precision": average_precision,
            }
        )
    return pd.DataFrame(rows)


def interval_metrics(
    actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float,
    model: str,
    run_id: str = "unassigned",
) -> pd.DataFrame:
    if not (actual.shape == lower.shape == upper.shape):
        raise ValueError("Actual and interval arrays must have identical shapes")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    if np.any(lower > upper):
        raise ValueError("Every lower interval bound must not exceed its upper bound")
    below = actual < lower
    above = actual > upper
    width = upper - lower
    interval_score = width + (2.0 / alpha) * (lower - actual) * below + (2.0 / alpha) * (actual - upper) * above

    def pinball(prediction: np.ndarray, quantile: float) -> float:
        residual = actual - prediction
        return float(np.mean(np.maximum(quantile * residual, (quantile - 1.0) * residual)))

    return pd.DataFrame(
        [
            {
                "run_id": run_id,
                "model": model,
                "nominal_coverage": 1.0 - alpha,
                "empirical_coverage": float(np.mean(~below & ~above)),
                "mean_width": float(np.mean(width)),
                "winkler_interval_score": float(np.mean(interval_score)),
                "lower_pinball_loss": pinball(lower, alpha / 2.0),
                "upper_pinball_loss": pinball(upper, 1.0 - alpha / 2.0),
                "n_points": int(actual.size),
            }
        ]
    )


def log_scale_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    model: str,
    run_id: str = "unassigned",
) -> pd.DataFrame:
    if actual.shape != predicted.shape:
        raise ValueError("Actual and prediction arrays must have identical shapes")
    actual_log = np.log1p(np.clip(actual, 0.0, None))
    predicted_log = np.log1p(np.clip(predicted, 0.0, None))
    values = metric_values(actual_log, predicted_log)
    return pd.DataFrame(
        [
            {
                "run_id": run_id,
                "model": model,
                "transform": "log1p_nonnegative",
                "n_points": int(actual.size),
                **values,
            }
        ]
    )


def quantile_forecast_metrics(
    actual: np.ndarray,
    quantile_forecasts: np.ndarray,
    quantile_levels: tuple[float, ...] | list[float],
    model: str,
    run_id: str = "unassigned",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return quantile calibration and a transparent quantile-grid CRPS approximation."""
    levels = np.asarray(quantile_levels, dtype=float)
    forecasts = np.asarray(quantile_forecasts, dtype=float)
    if actual.ndim != 2 or forecasts.shape != (*actual.shape, len(levels)):
        raise ValueError("Quantile forecasts must have shape [origins, horizon, quantiles]")
    if np.any(np.diff(levels) <= 0) or levels[0] <= 0 or levels[-1] >= 1:
        raise ValueError("Quantile levels must be strictly increasing within (0, 1)")
    if not np.isfinite(forecasts).all() or not np.isfinite(actual).all():
        raise ValueError("Quantile metric inputs must be finite")
    monotone = np.maximum.accumulate(forecasts, axis=-1)
    calibration_rows = []
    pinball_values = []
    for index, level in enumerate(levels):
        prediction = monotone[..., index]
        residual = actual - prediction
        pinball = float(np.mean(np.maximum(level * residual, (level - 1.0) * residual)))
        pinball_values.append(pinball)
        calibration_rows.append(
            {
                "run_id": run_id,
                "model": model,
                "quantile": float(level),
                "empirical_cdf": float(np.mean(actual <= prediction)),
                "calibration_error": float(np.mean(actual <= prediction) - level),
                "pinball_loss": pinball,
                "n_points": int(actual.size),
            }
        )
    losses = np.asarray(pinball_values)
    interior = np.sum((losses[:-1] + losses[1:]) * np.diff(levels) / 2.0)
    tail_approximation = levels[0] * losses[0] + (1.0 - levels[-1]) * losses[-1]
    crps_approximation = float(2.0 * (interior + tail_approximation))
    summary = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "model": model,
                "quantile_grid": ",".join(f"{value:.2f}" for value in levels),
                "crps_quantile_approximation": crps_approximation,
                "mean_absolute_calibration_error": float(
                    np.mean(np.abs([row["calibration_error"] for row in calibration_rows]))
                ),
                "monotonicity_correction_fraction": float(np.mean(monotone != forecasts)),
                "n_points": int(actual.size),
            }
        ]
    )
    return pd.DataFrame(calibration_rows), summary
