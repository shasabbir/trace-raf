# Event-TimeRAF

Kaggle-oriented research implementation for event-aware, retrieval-augmented
24-hour PM2.5 forecasting in Los Angeles County.

## Data

The pipeline uses source-preserving public records:

- US EPA AirData/AQS hourly PM2.5 parameter 88101;
- NOAA NCEI Global Hourly/ISD weather observations delivered through the
  official NOAA Open Data Dissemination (NODD) public buckets;
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
3. The setup cell locates an attached Kaggle Dataset and copies `configs/` and
   `src/` to `/kaggle/working/event_timeraf`. Set `PROJECT_ROOT_OVERRIDE` only
   when automatic discovery is insufficient.
4. Do not use a third-party Storm Events compilation for the final study. If
   Kaggle cannot reach NCEI HTTPS, run
   `notebooks/00_prepare_official_noaa_storm_cache.ipynb` locally or in Google
   Colab. It falls back to NOAA anonymous FTP, then creates a ZIP containing
   unchanged annual archives and `source_manifest.json`. Upload that ZIP as a
   private Kaggle Dataset and add it as notebook input.
5. The main notebook automatically finds and verifies that private cache. Set
   `STORM_EVENTS_CACHE` only when automatic discovery is insufficient.
6. Run the notebook from top to bottom with internet enabled for the first EPA
   and weather download, or attach a previously prepared `data/raw` cache.
7. The packaged notebook defaults to final-publication mode:
   `RUN_TSF_MODEL = True`, `FINAL_EXPERIMENT = True`, and
   `RETRIEVAL_EVIDENCE_REVIEWED = True`. It installs `chronos-forecasting` if
   missing, then requires the frozen Chronos gate to complete.

The attached private Dataset is a delivery cache, not a replacement source. The
loader accepts only NOAA annual filenames listed in the generated official-URL
manifest and verifies every SHA-256 hash before reading data. If Kaggle expands
the inner `.csv.gz` archives to `.csv`, the loader verifies those files with the
manifest's decompressed SHA-256 values. Records remain attributed to NOAA Storm
Events.

For an event-free engineering run only, set `REQUIRE_EVENTS = False`. This now
skips Storm Events acquisition completely. Such a run cannot support the
event-aware experiments or final paper claims.

Set `REQUIRE_STRICT_EVENT_AVAILABILITY = True` to reject retrospective event
timestamps and require a cache with genuine publication times.

For engineering runs only, manually set `RUN_TSF_MODEL = False`,
`FINAL_EXPERIMENT = False`, and `RETRIEVAL_EVIDENCE_REVIEWED = False`. A final
run requires `RETRIEVAL_EVIDENCE_REVIEWED = True` and completion of the
frozen-TSFM gate. Every result table carries a run ID and event-availability
mode, and the results notebook rejects artifacts that do not match the saved
manifest or do not satisfy final-run gates.

The publication-candidate profile uses a 24-hour knowledge-base stride and
reports 192-, 24-, and 6-hour sensitivity results. Candidate records may overlap
one another; leakage is prevented per query by requiring every candidate's
complete target to finish before the query's 168-hour lookback begins.

The primary event-aware path uses normalized event similarity and restricts an
event-context query to historical event-context candidates when at least `k`
causally eligible candidates exist. Every fallback and selected-candidate event
flag is stored in retrieval evidence.

After the final cell completes, download the printed
`event_timeraf_publication_candidate_<RUN_ID>.zip`. Then attach that ZIP to a
new Kaggle notebook and run `02_results_and_figures.ipynb` to audit the frozen
tables without retraining.

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
