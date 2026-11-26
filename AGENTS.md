# Repository guidance

## Purpose

This repository implements a customer-repurchase classification workflow. The target is `repurchased_60d`; the default model is XGBoost. Keep changes reproducible, configuration-driven, and scoped to the appropriate pipeline layer.

## Environment and commands

Use `uv` for Python commands. Do not invoke a system Python or create an unmanaged virtual environment.

```bash
uv sync
uv run python prepare_data.py
uv run python validate.py
uv run python tune.py
uv run python train.py
uv run python evaluate.py
uv run dvc repro
```

Quality checks:

```bash
uv run pytest
uv run black --check .
uv run isort --check-only .
uv run mypy src
```

## Project map

- `configs/`: YAML configuration; prefer adding settings here over hard-coding values.
- `src/data/`: input loading, contracts, splitting, and preparation orchestration.
- `src/features/` and `src/preprocessing/`: feature/target construction and cleaning.
- `src/training/`, `src/validation/`, `src/tuning/`, `src/evaluation/`: individual ML workflow stages.
- Root scripts (`prepare_data.py`, `train.py`, etc.): thin command-line entry points.
- `data/raw/`: source inputs; `data/processed/`: generated datasets.
- `artifacts/`: generated models, metrics, and tuning outputs.

## Change conventions

- Preserve the separation between configuration, pipeline orchestration, and model/data logic.
- Use type annotations for new or modified Python functions; the project runs strict `mypy` checks on `src`.
- Format code with Black and keep imports compatible with isort's Black profile (100-character line length).
- Keep random seeds and evaluation metrics explicit in configuration.
- Update `dvc.yaml` when a change affects the inputs, code, or outputs of the data-preparation stage; run `uv run dvc repro` to refresh `dvc.lock` when appropriate.
- Do not add secrets to Git. Use `.env` or environment variables for local credentials. RustFS requires `RUSTFS_ACCESS_KEY` and `RUSTFS_SECRET_KEY`.
- Treat raw data as potentially sensitive. Avoid replacing or publishing it unless the task explicitly requires it.

## Verification expectations

For code changes, run the smallest relevant pipeline command first, then applicable formatting, import, typing, and test checks. For data-pipeline changes, confirm that `data/processed/metadata.json` and the train/test outputs can be generated. Report checks that could not be run and why.
