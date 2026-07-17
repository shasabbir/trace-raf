# Event-TimeRAF

Kaggle-oriented research implementation for event-aware, retrieval-augmented
24-hour PM2.5 forecasting in Los Angeles County.

## Data

The pipeline uses source-preserving public records:

- US EPA AirData/AQS hourly PM2.5 parameter 88101;
- NOAA NCEI Global Hourly/ISD weather observations;
- NOAA Storm Events records;
- optional NOAA HMS fire/smoke records supplied as a cached table.

NOAA Storm Events does not provide a machine-readable publication timestamp.
The implementation records event start as a retrospective availability
assumption, so those event-aware runs are sensitivity experiments rather than
strict operational forecasts. A supplied event cache is strict only when it
contains genuine `published_at` values.

Raw downloads are cached under `data/raw/`. Generated datasets and model outputs
are ignored by Git. No experimental result is bundled with this repository.

## Kaggle

1. Upload this repository or add it as a Kaggle Dataset.
2. Open `notebooks/01_event_timeraf_kaggle_pipeline.ipynb`.
3. Set `PROJECT_ROOT` to the writable working copy.
4. Run the notebook from top to bottom with internet enabled for the first data
   download, or attach a previously prepared `data/raw` cache.
5. Set `RUN_TSF_MODEL = True` only when internet/model weights and sufficient
   runtime are available.

Set `REQUIRE_STRICT_EVENT_AVAILABILITY = True` to reject retrospective event
timestamps and require a cache with genuine publication times.

The notebook stops at a data-readiness gate when the official records do not
satisfy the configured coverage requirements. It never substitutes synthetic
research data.

## Local checks

```powershell
python -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
pytest -q
```

Install `requirements-optional.txt` only for the frozen Chronos publication gate.

See `structured_plan.md` for the experiment contract and leakage rules.
