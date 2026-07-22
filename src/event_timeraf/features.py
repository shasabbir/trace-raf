from __future__ import annotations

import numpy as np
import pandas as pd
import holidays

from .config import ProjectConfig


CALENDAR_FEATURES = (
    "cal_hour_sin",
    "cal_hour_cos",
    "cal_dow_sin",
    "cal_dow_cos",
    "cal_month_sin",
    "cal_month_cos",
    "cal_is_weekend",
    "cal_is_us_federal_holiday",
    "cal_is_california_holiday",
)

WEATHER_FEATURES = (
    "weather_temperature_c",
    "weather_relative_humidity",
    "weather_pressure_hpa",
    "weather_wind_speed_ms",
    "weather_wind_direction_deg",
    "weather_precipitation_mm",
)

EVENT_CATEGORIES = (
    "wildfire",
    "smoke",
    "excessive_heat",
    "high_wind",
    "heavy_rain",
    "flood",
    "dust",
    "traffic",
    "industrial_activity",
    "policy",
    "other_weather",
)


def add_calendar_features(frame: pd.DataFrame, cfg: ProjectConfig) -> pd.DataFrame:
    result = frame.copy()
    local = pd.DatetimeIndex(result["timestamp_utc"]).tz_convert(cfg.timezone)
    years = range(cfg.data.start_year, cfg.data.end_year + 1)
    federal = holidays.US(years=years)
    california = holidays.US(years=years, subdiv="CA")
    dates = pd.Series(local.date, index=result.index)
    result["cal_hour_sin"] = np.sin(2 * np.pi * local.hour / 24)
    result["cal_hour_cos"] = np.cos(2 * np.pi * local.hour / 24)
    result["cal_dow_sin"] = np.sin(2 * np.pi * local.dayofweek / 7)
    result["cal_dow_cos"] = np.cos(2 * np.pi * local.dayofweek / 7)
    result["cal_month_sin"] = np.sin(2 * np.pi * (local.month - 1) / 12)
    result["cal_month_cos"] = np.cos(2 * np.pi * (local.month - 1) / 12)
    result["cal_is_weekend"] = (local.dayofweek >= 5).astype(np.int8)
    result["cal_is_us_federal_holiday"] = dates.isin(federal).astype(np.int8)
    result["cal_is_california_holiday"] = dates.isin(california).astype(np.int8)
    return result


def add_pm25_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    values = result["pm25"]
    for lag in (1, 3, 6, 12, 24, 48, 168):
        result[f"pm25_lag_{lag}h"] = values.shift(lag)
    for window in (3, 6, 12, 24, 72, 168):
        rolling = values.rolling(window, min_periods=window)
        result[f"pm25_roll_mean_{window}h"] = rolling.mean()
    for window in (24, 168):
        rolling = values.rolling(window, min_periods=window)
        result[f"pm25_roll_std_{window}h"] = rolling.std(ddof=0)
        result[f"pm25_roll_min_{window}h"] = rolling.min()
        result[f"pm25_roll_max_{window}h"] = rolling.max()
    result["pm25_diff_1h"] = values.diff(1)
    result["pm25_diff_24h"] = values.diff(24)
    result["pm25_current"] = values
    result["pm25_current_filled"] = result["pm25_filled"].astype(np.int8)
    return result


def add_weather_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    rename = {
        "temperature_c": "weather_temperature_c",
        "relative_humidity": "weather_relative_humidity",
        "pressure_hpa": "weather_pressure_hpa",
        "wind_speed_ms": "weather_wind_speed_ms",
        "wind_direction_deg": "weather_wind_direction_deg",
        "precipitation_mm": "weather_precipitation_mm",
        "precipitation_missing": "weather_precipitation_missing_flag",
    }
    result = result.rename(columns=rename)
    result["weather_rain_flag"] = (result["weather_precipitation_mm"] > 0).astype(np.int8)
    result["weather_low_wind_flag"] = (result["weather_wind_speed_ms"] < 2.0).astype(np.int8)
    result["weather_high_humidity_flag"] = (result["weather_relative_humidity"] > 80).astype(np.int8)
    result["weather_stagnation_proxy"] = (
        result["weather_relative_humidity"] / (result["weather_wind_speed_ms"].clip(lower=0.2))
    )
    for column in ("weather_temperature_c", "weather_relative_humidity", "weather_wind_speed_ms"):
        result[f"{column}_roll_mean_24h"] = result[column].rolling(24, min_periods=24).mean()
    return result


def hourly_event_features(index: pd.DatetimeIndex, events: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame(index=index)
    if events.empty:
        output["event_count_1h"] = 0.0
        output["event_active_count_1h"] = 0.0
        for category in EVENT_CATEGORIES:
            output[f"event_{category}_1h"] = 0.0
            output[f"event_{category}_active_1h"] = 0.0
    else:
        usable = events.copy()
        usable["published_hour"] = pd.to_datetime(usable["published_at"], utc=True).dt.floor("h")
        usable["event_start_hour"] = pd.to_datetime(usable["event_time"], utc=True).dt.floor("h")
        usable["event_end_hour"] = pd.to_datetime(
            usable.get("event_end", usable["event_time"]), errors="coerce", utc=True
        ).dt.ceil("h")
        usable["event_end_hour"] = usable["event_end_hour"].fillna(usable["event_start_hour"])
        total = usable.groupby("published_hour").size()
        output["event_count_1h"] = total.reindex(index, fill_value=0).astype(float)
        output["event_active_count_1h"] = 0.0
        for category in EVENT_CATEGORIES:
            counts = usable.loc[usable["category"] == category].groupby("published_hour").size()
            output[f"event_{category}_1h"] = counts.reindex(index, fill_value=0).astype(float)
            output[f"event_{category}_active_1h"] = 0.0

        for event in usable.itertuples(index=False):
            active_start = max(event.event_start_hour, event.published_hour)
            active_end = max(event.event_end_hour, active_start)
            active = (index >= active_start) & (index <= active_end)
            output.loc[active, "event_active_count_1h"] += 1.0
            category_column = f"event_{event.category}_active_1h"
            if category_column in output:
                output.loc[active, category_column] += 1.0

    for window in (24, 72, 168):
        output[f"event_count_{window}h"] = output["event_count_1h"].rolling(window, min_periods=1).sum()
    for category in EVENT_CATEGORIES:
        base = output[f"event_{category}_1h"]
        output[f"event_{category}_72h"] = base.rolling(72, min_periods=1).sum()
    baseline = output["event_count_1h"].rolling(168, min_periods=24).mean().replace(0, np.nan)
    output["event_burst_ratio"] = (output["event_count_24h"] / (24 * baseline)).replace([np.inf, -np.inf], np.nan).fillna(0)
    return output


def build_modeling_table(
    pm25: pd.DataFrame,
    weather: pd.DataFrame,
    events: pd.DataFrame,
    cfg: ProjectConfig,
) -> pd.DataFrame:
    air = pm25.copy()
    air["timestamp_utc"] = pd.to_datetime(air["timestamp_utc"], utc=True)
    met = weather.copy()
    met["timestamp_utc"] = pd.to_datetime(met["timestamp_utc"], utc=True)
    met = met.drop(columns=["timestamp_local"], errors="ignore")
    result = air.merge(met, on="timestamp_utc", how="left", validate="one_to_one")
    result = result.sort_values("timestamp_utc").reset_index(drop=True)
    result = add_calendar_features(result, cfg)
    result = add_pm25_features(result)
    result = add_weather_features(result)
    event_frame = hourly_event_features(pd.DatetimeIndex(result["timestamp_utc"]), events)
    event_frame = event_frame.reset_index(names="timestamp_utc")
    result = result.merge(event_frame, on="timestamp_utc", how="left", validate="one_to_one")
    return result


def model_feature_columns(frame: pd.DataFrame) -> list[str]:
    prefixes = ("pm25_", "weather_", "cal_", "event_")
    excluded = {"pm25_observed", "pm25", "pm25_filled"}
    return [
        column
        for column in frame.columns
        if column.startswith(prefixes) and column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
    ]
