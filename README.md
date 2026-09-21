# TRACE-RAF Submission

This repository contains the notebooks, implementation package, configuration,
dependencies, and frozen result artifacts for TRACE-RAF: event-aware,
retrieval-augmented PM2.5 forecasting.

## Contents

- `notebooks/00_prepare_official_noaa_storm_cache.ipynb`: prepares the official NOAA Storm Events cache.
- `notebooks/01_event_timeraf_kaggle_pipeline.ipynb`: complete experiment pipeline.
- `notebooks/02_results_and_figures.ipynb`: reads the frozen artifacts and reproduces result tables and figures.
- `notebooks/03_paper_claim_verification.ipynb`: checks result completeness and claim-supporting artifacts.
- `src/event_timeraf/`: reusable implementation code.
- `configs/default.yaml`: experiment configuration.
- `outputs/`: frozen audit, evidence, figures, predictions, logs, and result tables.

The archived result run is `20260827T043457543402Z`. The notebooks and outputs
are retained as submitted artifacts; the repository cleanup does not execute
the notebooks or regenerate the results.

## Environment

Install the base dependencies from `requirements.txt`. Use
`requirements-optional.txt` or `requirements-publication-lock.txt` for the
additional forecasting-model dependencies.

The raw official data cache is intentionally not committed. It must be supplied
as an external input when rerunning the data-preparation or training notebook.
