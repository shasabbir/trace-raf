from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import sys
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

from .config import ProjectConfig


EPA_AIRDATA_URL = "https://aqs.epa.gov/aqsweb/airdata"
NOAA_ISD_HISTORY_URL = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
NOAA_GLOBAL_HOURLY_URL = "https://www.ncei.noaa.gov/data/global-hourly/access"
NOAA_STORM_URL = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles"


def download_file(url: str, destination: Path, force: bool = False) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0 and not force:
        return destination
    temporary = destination.with_suffix(destination.suffix + ".part")
    if force and temporary.exists():
        temporary.unlink()
    existing = temporary.stat().st_size if temporary.exists() else 0
    headers = {"User-Agent": "Event-TimeRAF/0.1 (academic data pipeline)"}
    if existing:
        headers["Range"] = f"bytes={existing}-"
    with requests.get(url, stream=True, timeout=(30, 300), headers=headers) as response:
        response.raise_for_status()
        append = existing > 0 and response.status_code == 206
        with temporary.open("ab" if append else "wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    temporary.replace(destination)
    return destination


def _normal_code(series: pd.Series, width: int) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(width)


def download_epa_pm25(cfg: ProjectConfig, force: bool = False) -> pd.DataFrame:
    """Download national EPA hourly files and retain Los Angeles County rows."""
    cache_dir = cfg.paths.raw / "epa_airdata"
    frames: list[pd.DataFrame] = []
    columns = [
        "State Code",
        "County Code",
        "Site Num",
        "Parameter Code",
        "POC",
        "Latitude",
        "Longitude",
        "Date Local",
        "Time Local",
        "Date GMT",
        "Time GMT",
        "Sample Measurement",
        "Units of Measure",
        "Sample Duration",
        "Method Type",
        "Method Code",
        "Method Name",
        "Qualifier",
        "Event Type",
    ]

    for year in range(cfg.data.start_year, cfg.data.end_year + 1):
        filtered_path = cache_dir / f"la_{cfg.data.epa_parameter_code}_{year}.parquet"
        if filtered_path.exists() and not force:
            frames.append(pd.read_parquet(filtered_path))
            continue

        archive = cache_dir / f"hourly_{cfg.data.epa_parameter_code}_{year}.zip"
        url = f"{EPA_AIRDATA_URL}/hourly_{cfg.data.epa_parameter_code}_{year}.zip"
        if archive.exists() and not zipfile.is_zipfile(archive):
            archive.unlink()
        download_file(url, archive, force=force)
        if not zipfile.is_zipfile(archive):
            raise ValueError(f"Downloaded EPA archive is incomplete or invalid: {archive}")
        selected: list[pd.DataFrame] = []
        with zipfile.ZipFile(archive) as zipped:
            members = [name for name in zipped.namelist() if name.lower().endswith(".csv")]
            if not members:
                raise ValueError(f"No CSV file found in {archive}")
            with zipped.open(members[0]) as source:
                for chunk in pd.read_csv(
                    source,
                    usecols=lambda name: name in columns,
                    dtype={"State Code": str, "County Code": str, "Site Num": str},
                    chunksize=250_000,
                    low_memory=False,
                ):
                    state = _normal_code(chunk["State Code"], 2)
                    county = _normal_code(chunk["County Code"], 3)
                    keep = (state == cfg.data.epa_state_code) & (county == cfg.data.epa_county_code)
                    if keep.any():
                        selected.append(chunk.loc[keep].copy())
        if not selected:
            raise ValueError(f"EPA file for {year} contains no configured county rows")
        filtered = pd.concat(selected, ignore_index=True)
        filtered.to_parquet(filtered_path, index=False)
        frames.append(filtered)

    return pd.concat(frames, ignore_index=True)


def prepare_epa_pm25(raw: pd.DataFrame, cfg: ProjectConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = raw.copy()
    if "Units of Measure" not in frame:
        frame["Units of Measure"] = pd.NA
    if "Sample Duration" not in frame:
        frame["Sample Duration"] = "not provided by archive"
    frame["site_id"] = (
        _normal_code(frame["State Code"], 2)
        + "-"
        + _normal_code(frame["County Code"], 3)
        + "-"
        + _normal_code(frame["Site Num"], 4)
    )
    frame["timestamp_utc"] = pd.to_datetime(
        frame["Date GMT"].astype(str) + " " + frame["Time GMT"].astype(str),
        errors="coerce",
        utc=True,
    )
    frame["value"] = pd.to_numeric(frame["Sample Measurement"], errors="coerce")
    frame.loc[frame["value"] < 0, "value"] = np.nan
    frame = frame.dropna(subset=["timestamp_utc", "value", "site_id"])

    expected_start = pd.Timestamp(f"{cfg.data.start_year}-01-01", tz="UTC")
    expected_end = pd.Timestamp(f"{cfg.data.end_year + 1}-01-01", tz="UTC")
    expected_hours = int((expected_end - expected_start) / pd.Timedelta(hours=1))
    coverage = (
        frame.groupby("site_id")
        .agg(
            observed_hours=("timestamp_utc", "nunique"),
            source_records=("timestamp_utc", "size"),
            latitude=("Latitude", "median"),
            longitude=("Longitude", "median"),
            first_time=("timestamp_utc", "min"),
            last_time=("timestamp_utc", "max"),
            units=("Units of Measure", lambda values: " | ".join(sorted(set(values.dropna().astype(str))))),
            sample_durations=("Sample Duration", lambda values: " | ".join(sorted(set(values.dropna().astype(str))))),
        )
        .reset_index()
    )
    coverage["coverage"] = coverage["observed_hours"] / expected_hours
    coverage = coverage.sort_values(["observed_hours", "site_id"], ascending=[False, True])
    if coverage.empty:
        raise ValueError("No valid EPA PM2.5 measurements remain after cleaning")
    site_id = str(coverage.iloc[0]["site_id"])

    site = frame.loc[frame["site_id"] == site_id].copy()
    hourly = (
        site.groupby("timestamp_utc", as_index=False)
        .agg(
            pm25_observed=("value", "median"),
            monitor_count=("POC", "nunique"),
            latitude=("Latitude", "median"),
            longitude=("Longitude", "median"),
        )
        .set_index("timestamp_utc")
        .sort_index()
    )
    index = pd.date_range(expected_start, expected_end, freq="h", inclusive="left")
    hourly = hourly.reindex(index)
    hourly.index.name = "timestamp_utc"
    hourly["pm25"] = hourly["pm25_observed"].ffill(limit=cfg.data.maximum_fill_gap_hours)
    hourly["pm25_filled"] = hourly["pm25_observed"].isna() & hourly["pm25"].notna()
    hourly["site_id"] = site_id
    hourly["latitude"] = hourly["latitude"].fillna(float(coverage.iloc[0]["latitude"]))
    hourly["longitude"] = hourly["longitude"].fillna(float(coverage.iloc[0]["longitude"]))
    hourly["timestamp_local"] = hourly.index.tz_convert(cfg.timezone)
    return hourly.reset_index(), coverage


def haversine_km(lat1: float, lon1: float, lat2: Iterable[float], lon2: Iterable[float]) -> np.ndarray:
    lat2_array = np.asarray(lat2, dtype=float)
    lon2_array = np.asarray(lon2, dtype=float)
    phi1 = math.radians(lat1)
    phi2 = np.radians(lat2_array)
    dphi = phi2 - phi1
    dlambda = np.radians(lon2_array - lon1)
    value = np.sin(dphi / 2) ** 2 + math.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 6371.0088 * 2 * np.arctan2(np.sqrt(value), np.sqrt(1 - value))


def noaa_station_candidates(
    cfg: ProjectConfig,
    latitude: float,
    longitude: float,
    force: bool = False,
) -> pd.DataFrame:
    inventory_path = cfg.paths.raw / "noaa_isd" / "isd-history.csv"
    download_file(NOAA_ISD_HISTORY_URL, inventory_path, force=force)
    inventory = pd.read_csv(inventory_path, dtype={"USAF": str, "WBAN": str, "BEGIN": str, "END": str})
    inventory["USAF"] = _normal_code(inventory["USAF"], 6)
    inventory["WBAN"] = _normal_code(inventory["WBAN"], 5)
    inventory["BEGIN_DATE"] = pd.to_datetime(inventory["BEGIN"], format="%Y%m%d", errors="coerce")
    inventory["END_DATE"] = pd.to_datetime(inventory["END"], format="%Y%m%d", errors="coerce")
    start = pd.Timestamp(f"{cfg.data.start_year}-01-01")
    end = pd.Timestamp(f"{cfg.data.end_year}-12-31")
    eligible = inventory.loc[
        (inventory["CTRY"] == "US")
        & (inventory["BEGIN_DATE"] <= start)
        & (inventory["END_DATE"] >= end)
        & inventory["LAT"].notna()
        & inventory["LON"].notna()
        & inventory["ICAO"].notna()
        & (inventory["ICAO"].astype(str).str.strip() != "")
    ].copy()
    eligible["distance_km"] = haversine_km(latitude, longitude, eligible["LAT"], eligible["LON"])
    eligible = eligible.loc[eligible["distance_km"] <= cfg.data.noaa_weather_max_distance_km]
    if eligible.empty:
        raise ValueError("No NOAA ISD station passes date and distance requirements")
    # Airport/field stations are preferred because they usually provide the
    # continuous ASOS/AWOS record required by an hourly forecasting study.
    station_name = eligible["STATION NAME"].astype(str).str.upper()
    eligible["continuous_station_priority"] = station_name.str.contains(
        r"AIRPORT|ARPT|\bFLD\b|\bFIELD\b", regex=True
    ).astype(int)
    return eligible.sort_values(
        ["continuous_station_priority", "distance_km", "USAF", "WBAN"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)


def select_noaa_station(cfg: ProjectConfig, latitude: float, longitude: float, force: bool = False) -> pd.Series:
    return noaa_station_candidates(cfg, latitude, longitude, force=force).iloc[0]


def _scaled_noaa(series: pd.Series, missing: int, scale: float = 10.0) -> pd.Series:
    first = series.astype("string").str.split(",").str[0]
    values = pd.to_numeric(first, errors="coerce")
    values = values.mask(values.abs() >= missing)
    return values / scale


def _parse_wind(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    parts = series.astype("string").str.split(",")
    direction = pd.to_numeric(parts.str[0], errors="coerce").mask(lambda s: s >= 999)
    speed = pd.to_numeric(parts.str[3], errors="coerce").mask(lambda s: s >= 9999) / 10.0
    return direction, speed


def _parse_precip(series: pd.Series) -> pd.Series:
    parts = series.astype("string").str.split(",")
    depth = pd.to_numeric(parts.str[1], errors="coerce").mask(lambda s: s >= 9999)
    return depth / 10.0


def download_noaa_weather(cfg: ProjectConfig, station: pd.Series, force: bool = False) -> pd.DataFrame:
    station_id = f"{station['USAF']}{station['WBAN']}"
    cache_dir = cfg.paths.raw / "noaa_isd" / station_id
    frames: list[pd.DataFrame] = []
    for year in range(cfg.data.start_year, cfg.data.end_year + 1):
        path = cache_dir / f"{station_id}_{year}.csv"
        url = f"{NOAA_GLOBAL_HOURLY_URL}/{year}/{station_id}.csv"
        download_file(url, path, force=force)
        frames.append(pd.read_csv(path, low_memory=False))
    return pd.concat(frames, ignore_index=True)


def prepare_noaa_weather(raw: pd.DataFrame, cfg: ProjectConfig) -> pd.DataFrame:
    frame = pd.DataFrame()
    frame["timestamp_utc"] = pd.to_datetime(raw["DATE"], errors="coerce", utc=True)
    frame["temperature_c"] = _scaled_noaa(raw["TMP"], missing=9999)
    frame["dewpoint_c"] = _scaled_noaa(raw["DEW"], missing=9999) if "DEW" in raw else np.nan
    sea_level_pressure = _scaled_noaa(raw["SLP"], missing=99999) if "SLP" in raw else pd.Series(np.nan, index=raw.index)
    station_pressure = _scaled_noaa(raw["STP"], missing=99999) if "STP" in raw else pd.Series(np.nan, index=raw.index)
    frame["pressure_hpa"] = sea_level_pressure.fillna(station_pressure)
    if "WND" in raw:
        frame["wind_direction_deg"], frame["wind_speed_ms"] = _parse_wind(raw["WND"])
    else:
        frame["wind_direction_deg"] = np.nan
        frame["wind_speed_ms"] = np.nan
    frame["precipitation_mm"] = _parse_precip(raw["AA1"]) if "AA1" in raw else 0.0
    a = 17.625
    b = 243.04
    numerator = np.exp((a * frame["dewpoint_c"]) / (b + frame["dewpoint_c"]))
    denominator = np.exp((a * frame["temperature_c"]) / (b + frame["temperature_c"]))
    frame["relative_humidity"] = (100 * numerator / denominator).clip(0, 100)
    frame = frame.dropna(subset=["timestamp_utc"]).set_index("timestamp_utc").sort_index()
    hourly = frame.resample("h").agg(
        {
            "temperature_c": "mean",
            "dewpoint_c": "mean",
            "relative_humidity": "mean",
            "pressure_hpa": "mean",
            "wind_direction_deg": "mean",
            "wind_speed_ms": "mean",
            "precipitation_mm": "sum",
        }
    )
    for column in hourly.columns:
        if column != "precipitation_mm":
            hourly[column] = hourly[column].ffill(limit=cfg.data.maximum_fill_gap_hours)
    hourly["precipitation_mm"] = hourly["precipitation_mm"].fillna(0.0)
    hourly["timestamp_local"] = hourly.index.tz_convert(cfg.timezone)
    return hourly.reset_index()


def select_and_download_noaa_weather(
    cfg: ProjectConfig,
    latitude: float,
    longitude: float,
    force: bool = False,
    candidate_limit: int = 10,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Choose the nearest station that passes measured hourly coverage."""
    candidates = noaa_station_candidates(cfg, latitude, longitude, force=force)
    expected_index = pd.date_range(
        pd.Timestamp(f"{cfg.data.start_year}-01-01", tz="UTC"),
        pd.Timestamp(f"{cfg.data.end_year + 1}-01-01", tz="UTC"),
        freq="h",
        inclusive="left",
    )
    core_columns = ["temperature_c", "relative_humidity", "wind_speed_ms", "pressure_hpa"]
    best: tuple[float, pd.Series, pd.DataFrame, pd.DataFrame] | None = None
    failures: list[str] = []
    for _, station in candidates.head(candidate_limit).iterrows():
        station_id = f"{station['USAF']}{station['WBAN']}"
        try:
            raw = download_noaa_weather(cfg, station, force=force)
            weather = prepare_noaa_weather(raw, cfg)
            aligned = weather.set_index("timestamp_utc").reindex(expected_index)
            coverage = float(aligned[core_columns].notna().all(axis=1).mean())
            station = station.copy()
            station["measured_core_coverage"] = coverage
            if best is None or coverage > best[0]:
                best = (coverage, station, raw, weather)
            if coverage >= cfg.data.minimum_weather_coverage:
                return station, raw, weather
        except Exception as error:
            failures.append(f"{station_id}: {error}")
    if best is None:
        raise ValueError("No NOAA station could be loaded. " + "; ".join(failures))
    return best[1], best[2], best[3]


def _storm_archive_name(year: int) -> str:
    response = requests.get(f"{NOAA_STORM_URL}/", timeout=(30, 120))
    response.raise_for_status()
    pattern = re.compile(rf'(StormEvents_details-ftp_v1\.0_d{year}_c\d{{8}}\.csv\.gz)')
    names = sorted(set(pattern.findall(response.text)))
    if not names:
        raise ValueError(f"No NOAA Storm Events details archive found for {year}")
    return names[-1]


def _event_category(event_type: str) -> str:
    value = event_type.upper()
    mapping = [
        (("WILDFIRE",), "wildfire"),
        (("DENSE SMOKE",), "smoke"),
        (("EXCESSIVE HEAT", "HEAT"), "excessive_heat"),
        (("HIGH WIND", "STRONG WIND"), "high_wind"),
        (("DUST",), "dust"),
        (("FLASH FLOOD", "FLOOD"), "flood"),
        (("HEAVY RAIN", "THUNDERSTORM"), "heavy_rain"),
    ]
    for needles, category in mapping:
        if any(needle in value for needle in needles):
            return category
    return "other_weather"


def download_storm_events(cfg: ProjectConfig, force: bool = False) -> pd.DataFrame:
    cache_dir = cfg.paths.raw / "noaa_storm_events"
    frames: list[pd.DataFrame] = []
    for year in range(cfg.data.start_year, cfg.data.end_year + 1):
        filtered_path = cache_dir / f"los_angeles_events_{year}.parquet"
        if filtered_path.exists() and not force:
            filtered = pd.read_parquet(filtered_path)
            if "SOURCE_ARCHIVE_YEAR" not in filtered:
                filtered["SOURCE_ARCHIVE_YEAR"] = year
            frames.append(filtered)
            continue
        name = _storm_archive_name(year)
        archive = download_file(f"{NOAA_STORM_URL}/{name}", cache_dir / name, force=force)
        raw = pd.read_csv(archive, compression="gzip", low_memory=False)
        fips = pd.to_numeric(raw.get("CZ_FIPS"), errors="coerce")
        cz_type = raw.get("CZ_TYPE", pd.Series("", index=raw.index)).astype(str).str.upper()
        cz_name = raw.get("CZ_NAME", pd.Series("", index=raw.index)).astype(str).str.upper()
        state = raw.get("STATE", pd.Series("", index=raw.index)).astype(str).str.upper()
        keep = (state == "CALIFORNIA") & (((cz_type == "C") & (fips == 37)) | cz_name.str.contains("LOS ANGELES"))
        filtered = raw.loc[keep].copy()
        filtered["SOURCE_ARCHIVE_YEAR"] = year
        filtered.to_parquet(filtered_path, index=False)
        frames.append(filtered)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def prepare_storm_events(raw: pd.DataFrame, cfg: ProjectConfig) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(
            columns=["event_id", "event_time", "event_end", "published_at", "category", "source", "title", "summary"]
        )
    start = pd.to_datetime(raw["BEGIN_DATE_TIME"], errors="coerce", format="mixed", dayfirst=True)
    end = pd.to_datetime(raw["END_DATE_TIME"], errors="coerce", format="mixed", dayfirst=True)
    start = start.dt.tz_localize(cfg.timezone, ambiguous="NaT", nonexistent="shift_forward")
    end = end.dt.tz_localize(cfg.timezone, ambiguous="NaT", nonexistent="shift_forward")
    source_year = pd.to_numeric(
        raw.get("SOURCE_ARCHIVE_YEAR", raw.get("YEAR", pd.Series(pd.NA, index=raw.index))),
        errors="coerce",
    ).astype("Int64")
    coverage_start = pd.to_datetime(source_year.astype(str) + "-01-01", errors="coerce", utc=True)
    coverage_end = pd.to_datetime((source_year + 1).astype(str) + "-01-01", errors="coerce", utc=True)
    events = pd.DataFrame(
        {
            "event_id": "storm_" + raw["EVENT_ID"].astype("Int64").astype(str),
            "event_time": start.dt.tz_convert("UTC"),
            "event_end": end.dt.tz_convert("UTC"),
            # Storm Events has no machine-readable publication timestamp. The
            # retrospective MVP treats event start as the availability time and
            # records this assumption for sensitivity analysis.
            "published_at": start.dt.tz_convert("UTC"),
            "availability_assumption": "event_start",
            "source_coverage_start": coverage_start,
            "source_coverage_end": coverage_end,
            "coverage_basis": "NOAA annual source archive",
            "category": raw["EVENT_TYPE"].fillna("Unknown").map(_event_category),
            "source": "NOAA Storm Events",
            "source_url": "https://www.ncei.noaa.gov/stormevents/",
            "title": raw["EVENT_TYPE"].fillna("NOAA event") + " - " + raw["CZ_NAME"].fillna("Los Angeles"),
            "summary": raw.get("EVENT_NARRATIVE", pd.Series("", index=raw.index)).fillna(""),
        }
    )
    return events.dropna(subset=["event_time", "published_at"]).drop_duplicates("event_id").sort_values("event_time")


def load_optional_hms_events(path: str | Path, cfg: ProjectConfig) -> pd.DataFrame:
    """Load a source-preserving HMS cache already converted to CSV/Parquet.

    Required columns are event_time, published_at, category, and event_id. This
    explicit schema avoids silently guessing publication times from geometries.
    """
    source = Path(path)
    frame = pd.read_parquet(source) if source.suffix.lower() == ".parquet" else pd.read_csv(source)
    required = {"event_id", "event_time", "published_at", "category"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"HMS cache is missing required columns: {sorted(missing)}")
    for column in ("event_time", "published_at"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce", utc=True)
    if {"source_coverage_start", "source_coverage_end"}.issubset(frame.columns):
        frame["source_coverage_start"] = pd.to_datetime(frame["source_coverage_start"], errors="coerce", utc=True)
        frame["source_coverage_end"] = pd.to_datetime(frame["source_coverage_end"], errors="coerce", utc=True)
    else:
        coverage_start = frame["event_time"].min()
        coverage_end = frame["event_time"].max()
        frame["source_coverage_start"] = coverage_start
        frame["source_coverage_end"] = coverage_end
        frame["coverage_basis"] = "first-to-last event in supplied HMS cache"
    frame["source"] = frame.get("source", "NOAA HMS")
    frame["source_url"] = frame.get("source_url", "https://ospo.noaa.gov/products/land/hms.html")
    return frame.dropna(subset=["event_time", "published_at"]).copy()


def _event_coverage_days(events: pd.DataFrame, study_start: pd.Timestamp, study_end: pd.Timestamp) -> int:
    if events.empty or not {"source_coverage_start", "source_coverage_end"}.issubset(events.columns):
        return 0
    starts = pd.to_datetime(events["source_coverage_start"], errors="coerce", utc=True)
    ends = pd.to_datetime(events["source_coverage_end"], errors="coerce", utc=True)
    coverage_ranges = pd.DataFrame({"start": starts, "end": ends}).dropna().drop_duplicates()
    covered: set[pd.Timestamp] = set()
    for start, end in coverage_ranges.itertuples(index=False, name=None):
        clipped_start = max(start, study_start).floor("D")
        clipped_end = min(end, study_end).ceil("D")
        if clipped_start < clipped_end:
            covered.update(pd.date_range(clipped_start, clipped_end, freq="D", inclusive="left"))
    return len(covered)


def build_data_audit(
    pm25: pd.DataFrame,
    weather: pd.DataFrame,
    events: pd.DataFrame,
    cfg: ProjectConfig,
    station_coverage: pd.DataFrame,
    weather_station: pd.Series,
) -> dict:
    observed_coverage = float(pm25["pm25_observed"].notna().mean())
    weather_columns = ["temperature_c", "relative_humidity", "wind_speed_ms", "pressure_hpa"]
    pm_timeline = pd.DataFrame({"timestamp_utc": pd.to_datetime(pm25["timestamp_utc"], utc=True)})
    aligned_weather = pm_timeline.merge(
        weather[["timestamp_utc", *weather_columns]], on="timestamp_utc", how="left", validate="one_to_one"
    )
    weather_coverage = float(aligned_weather[weather_columns].notna().all(axis=1).mean())
    event_times = pd.to_datetime(events["event_time"], errors="coerce", utc=True) if not events.empty else pd.Series(dtype="datetime64[ns, UTC]")
    event_days = int(event_times.dt.floor("D").nunique()) if not events.empty else 0
    categories = events["category"].value_counts().to_dict() if not events.empty else {}
    study_start = pd.Timestamp(f"{cfg.data.start_year}-01-01", tz="UTC")
    study_end = pd.Timestamp(f"{cfg.data.end_year + 1}-01-01", tz="UTC")
    event_overlap_days = _event_coverage_days(events, study_start, study_end)
    qualifying_categories = sum(
        count >= cfg.data.minimum_event_records_per_category for count in categories.values()
    )
    selected_site_id = str(pm25["site_id"].iloc[0])
    selected_site = station_coverage.loc[station_coverage["site_id"] == selected_site_id].iloc[0]
    units = [unit.strip() for unit in str(selected_site.get("units", "")).split("|") if unit.strip()]
    units_known_and_consistent = len(units) == 1
    availability_assumptions = (
        sorted(events.get("availability_assumption", pd.Series("reported", index=events.index)).fillna("reported").unique())
        if not events.empty
        else []
    )
    strict_event_availability = bool(availability_assumptions) and availability_assumptions == ["reported"]
    gates = {
        "pm25_coverage": observed_coverage >= cfg.data.minimum_pm25_coverage,
        "pm25_units_known_and_consistent": units_known_and_consistent,
        "weather_coverage": weather_coverage >= cfg.data.minimum_weather_coverage,
        "event_overlap": event_overlap_days >= cfg.data.minimum_event_overlap_days,
        "event_days": event_days >= cfg.data.minimum_event_days,
        "event_categories": qualifying_categories >= cfg.data.minimum_event_categories,
    }
    audit = {
        "study_period": [cfg.data.start_year, cfg.data.end_year],
        "epa_parameter_code": cfg.data.epa_parameter_code,
        "audit_generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "source_urls": {
            "pm25": f"{EPA_AIRDATA_URL}/download_files.html",
            "weather": "https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database",
            "events": "https://www.ncei.noaa.gov/stormevents/ftp.jsp",
        },
        "source_terms_note": "Source licenses/terms were not machine-parsed; consult each recorded source page.",
        "selected_epa_site": selected_site_id,
        "selected_epa_site_observed_hours": int(selected_site["observed_hours"]),
        "selected_epa_site_source_records": int(selected_site["source_records"]),
        "pm25_units": units,
        "pm25_sample_durations": str(selected_site.get("sample_durations", "")),
        "epa_site_count": int(len(station_coverage)),
        "selected_weather_station": f"{weather_station['USAF']}{weather_station['WBAN']}",
        "weather_station_name": str(weather_station["STATION NAME"]),
        "weather_station_distance_km": float(weather_station["distance_km"]),
        "pm25_observed_coverage": observed_coverage,
        "weather_complete_coverage": weather_coverage,
        "event_days": event_days,
        "event_overlap_days": event_overlap_days,
        "event_sources": sorted(events["source"].dropna().astype(str).unique()) if not events.empty else [],
        "event_availability_assumptions": availability_assumptions,
        "strict_event_availability": strict_event_availability,
        "qualifying_event_categories": qualifying_categories,
        "event_categories": categories,
        "gates": gates,
        "processed_duplicate_counts": {
            "pm25_timestamp": int(pd.to_datetime(pm25["timestamp_utc"], utc=True).duplicated().sum()),
            "weather_timestamp": int(pd.to_datetime(weather["timestamp_utc"], utc=True).duplicated().sum()),
            "event_id": int(events["event_id"].duplicated().sum()) if not events.empty else 0,
        },
        "core_ready": bool(
            gates["pm25_coverage"]
            and gates["pm25_units_known_and_consistent"]
            and gates["weather_coverage"]
        ),
        "event_ready": bool(gates["event_overlap"] and gates["event_days"] and gates["event_categories"]),
        "site_candidates": station_coverage.head(10).to_dict(orient="records"),
        "event_availability_note": (
            "NOAA Storm Events uses event start as a retrospective availability assumption. "
            "Event-aware results are sensitivity results unless strict_event_availability is true."
        ),
    }
    path = cfg.paths.outputs / "audit" / "data_audit.json"
    path.write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
    return audit


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_run_manifest(cfg: ProjectConfig, artifacts: Iterable[Path] = ()) -> dict:
    try:
        import importlib.metadata as metadata

        packages = {name: metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "xgboost", "pyarrow")}
    except Exception:
        packages = {}
    manifest = {
        "project": cfg.name,
        "seed": cfg.seed,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "artifacts": {str(path): sha256_file(path) for path in artifacts if path.exists() and path.is_file()},
    }
    path = cfg.paths.outputs / "logs" / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
