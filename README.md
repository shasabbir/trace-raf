# TRACE-RAF / Event-TimeRAF

Kaggle-oriented research implementation for event-aware, retrieval-augmented
24-hour PM2.5 forecasting in Los Angeles County from 1 January 2019 through
31 December 2025. Model development ends on 24 August 2025; observations from
25 August onward form the explicit final holdout and are excluded from model
and hyperparameter selection. January 2025 is reported separately as a
development-stress analysis because it belongs to validation; it must not be
described as final-holdout evidence.

The proposed model is **TRACE-RAF**: Trust-gated Residual Analog Correction for
Event-aware Retrieval-Augmented Forecasting. It retains Event-TimeRAF's audited
event retrieval but adds a convex XGBoost-LightGBM base, expanding-window
out-of-fold residual memory, and a bounded validation-trained gate. The name is
a research model identifier; the repository does not claim legal registration.

## Data

The pipeline uses source-preserving public records:

- US EPA AirData/AQS hourly PM2.5 parameter 88101;
- NOAA NCEI GHCNh weather observations from the official annual station files;
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
   NOAA retired ISD/Global Hourly in August 2025. Because this study continues
   through December, the weather loader uses its official GHCNh replacement
   consistently for every year rather than changing sources inside the final
   holdout. Internet must be enabled for the first GHCNh download, or a complete
   `data/raw/noaa_ghcnh` cache must be attached.
7. The packaged notebook defaults to final-publication mode:
   `RUN_TSF_MODEL = True`, `FINAL_EXPERIMENT = True`, and
   `RETRIEVAL_EVIDENCE_REVIEWED = True`. It installs `chronos-forecasting` if
   missing, then requires the frozen Chronos gate, journal baselines, and all
   reviewer-response sensitivity arms to complete.

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

The publication-candidate profile uses a 24-hour primary knowledge-base stride
and reports 1-, 6-, and 24-hour sensitivity results. TRACE-RAF independently
selects its residual-memory stride from the same set using validation data;
only that selected TRACE configuration is evaluated on the final holdout.
Candidate records may overlap one another; leakage is prevented per query by
requiring every candidate's complete target to finish before the query's
168-hour lookback begins.

The same run evaluates DLinear, LSTM, PatchTST, LightGBM, ridge, climatology,
the pre-specified Event-TimeRAF variants, and TRACE-RAF (`M13`). Its no-event
ablation is `A03`, while `C06` is the no-retrieval convex context ensemble. A
separate three-monitor arm compares
the context XGBoost and event-conditioned retrieval models while holding the
primary weather covariates fixed; its station-to-monitor distances are recorded
so it is interpreted as target-construction sensitivity, not spatial validation.

The primary event-aware path uses normalized event similarity and restricts an
event-context query to historical event-context candidates when at least `k`
causally eligible candidates exist. Every fallback and selected-candidate event
flag is stored in retrieval evidence.

After the final cell completes, download the printed
`event_timeraf_publication_candidate_<RUN_ID>.zip`. Then attach that ZIP to a
new Kaggle notebook and run `02_results_and_figures.ipynb` to audit the frozen
tables, verify every archive hash, and independently recompute the saved metrics
without retraining. Run `03_paper_claim_verification.ipynb` on the same ZIP to
display the exact claim-source tables and reject mixed or incomplete runs.

TRACE-RAF is accepted as complete only when its residual candidates have
out-of-fold predictions whose training targets end before the candidate input,
its residual-memory stride and correction strength are selected on validation,
the gate includes a zero-correction option, and Notebook 02 reconstructs the
saved prediction as `base + gate * residual_correction`.

Physical AQI-threshold metrics remain unchanged. A separate operational table
selects model-specific alert cutoffs on validation data and applies the frozen
cutoffs to the final holdout; it never changes the regression predictions.

The notebook stops at a data-readiness gate when the official records do not
satisfy the configured coverage requirements. It never substitutes synthetic
research data.

## Local checks

```powershell
python -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
pytest -q
```

Install `requirements-optional.txt` for Chronos, PatchTST, and LightGBM outside
the self-installing Kaggle notebook. `requirements-publication-lock.txt` records
the exact successful publication-run package versions.

See `structured_plan.md` for the experiment contract and leakage rules.
