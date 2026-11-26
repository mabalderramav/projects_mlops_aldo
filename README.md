# MLOps Customer Repurchase Project

An end-to-end machine-learning project for predicting whether a customer will repurchase within 60 days. It turns order, product, and user data into a reproducible training dataset, trains an XGBoost classifier, validates and tunes it, and records the resulting artifacts locally.

## What is included

- Data preparation from CSV source data in `data/raw/`
- Feature engineering and preprocessing in `src/features/` and `src/preprocessing/`
- A stratified train/test split written as Parquet files
- XGBoost model training, cross-validation, Optuna hyperparameter tuning, and holdout evaluation
- DVC tracking for the data-preparation stage and optional S3-compatible remote storage
- A Docker Compose definition for local RustFS object storage

## Prerequisites

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)
- Docker and Docker Compose only when using the local object-storage service

## Quick start

From the repository root, create the environment and install the locked dependencies:

```bash
uv sync
```

Prepare the dataset, train the baseline model, and evaluate it:

```bash
uv run python prepare_data.py
uv run python train.py
uv run python evaluate.py
```

The commands use the configuration files in `configs/` by default. The prepared dataset is written to `data/processed/`; model and metric artifacts are written to `artifacts/`.

## Workflow

| Step | Command | Default configuration | Output |
| --- | --- | --- | --- |
| Prepare data | `uv run python prepare_data.py` | `configs/data.yaml` | `data/processed/train.parquet`, `test.parquet`, and `metadata.json` |
| Validate | `uv run python validate.py` | `configs/validation.yaml` | `artifacts/validation_metrics.json` |
| Tune | `uv run python tune.py` | `configs/tuning.yaml` | `artifacts/best_hyperparameters.json` and `tuning_results.json` |
| Train baseline | `uv run python train.py` | `configs/training.yaml` | `artifacts/xgboost.joblib` |
| Evaluate baseline | `uv run python evaluate.py` | `configs/evaluation.yaml` | `artifacts/metrics.json` |

To train and evaluate with tuned parameters, point the settings at the tuned configurations:

```bash
TRAINING_CONFIG_PATH=configs/training_tuned.yaml uv run python train.py
EVALUATION_CONFIG_PATH=configs/evaluation_tuned.yaml uv run python evaluate.py
```

The scripts read settings from environment variables and `.env` if present. Useful overrides include `APP_ENV`, `LOG_LEVEL`, `DATA_CONFIG_PATH`, `TRAINING_CONFIG_PATH`, `VALIDATION_CONFIG_PATH`, `TUNING_CONFIG_PATH`, and `EVALUATION_CONFIG_PATH`.

## Data and modeling

`configs/data.yaml` defines the four raw input sources, filters completed orders, defines the `repurchased_60d` target, and controls the stratified split. The default model is XGBoost. Its baseline parameters are stored in `configs/training.yaml`; the Optuna search space and cross-validation settings are in `configs/tuning.yaml`.

Raw source data may contain sensitive business or customer information. Do not commit credentials, access keys, or replacement production data without confirming the repository's data-handling requirements.

## Reproducibility with DVC

The `prepare_data` stage is declared in `dvc.yaml`. Reproduce it with:

```bash
uv run dvc repro
```

Useful DVC commands are also available through Task:

```bash
task dvc:status
task dvc:pull
task dvc:push
```

Run `task dvc:pull` or `task dvc:push` only after a DVC remote has been configured and you have the required access.

## Local object storage

`docker-compose.yml` starts RustFS, an S3-compatible object store bound to localhost ports `9000` and `9001`. Set `RUSTFS_ACCESS_KEY` and `RUSTFS_SECRET_KEY` in your environment or `.env`, then run:

```bash
task infra:up
task infra:status
task infra:down
```

## Repository layout

```text
configs/       Pipeline and model configuration
data/raw/      Source CSV files
data/processed/ Prepared train/test datasets
src/data/      Data loading, splitting, and pipeline code
src/features/  Feature and target construction
src/preprocessing/ Data cleaning and transformations
src/training/  Model training
src/validation/ Cross-validation
src/tuning/    Optuna tuning
src/evaluation/ Holdout evaluation
artifacts/     Models, metrics, and tuning results
```

## Development

Run the available checks with the development dependencies installed:

```bash
uv run pytest
uv run black --check .
uv run isort --check-only .
uv run mypy src
```

See [AGENTS.md](AGENTS.md) for contribution and automation guidance.
