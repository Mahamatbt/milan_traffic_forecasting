# Milan traffic forecasting pipeline.
#
# Local runs use the project virtualenv; Kaggle runs invoke the same modules
# with CONFIG=config/kaggle.yaml. Every target is a thin wrapper around a
# module in src/ so the pipeline is identical in both environments.

PYTHON ?= .venv/Scripts/python.exe
CONFIG ?= config/default.yaml
ARGS   ?=

.DEFAULT_GOAL := help
.PHONY: help env test lint download ingest matrix eda train evaluate report all clean clean-interim

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

env:  ## Record hardware and library versions to results/environment.json
	$(PYTHON) -m src.env_report --config $(CONFIG)

test:  ## Run the test suite
	$(PYTHON) -m pytest tests/ -q

lint:  ## Lint and format-check src/ and tests/
	$(PYTHON) -m ruff check src/ tests/
	$(PYTHON) -m ruff format --check src/ tests/

download:  ## Fetch the 62 daily CDR files and the Milano Grid GeoJSON
	$(PYTHON) -m src.download --config $(CONFIG) $(ARGS)

ingest:  ## Convert raw text to per-day parquet with memory measurement
	$(PYTHON) -m src.ingest --config $(CONFIG) $(ARGS)

matrix:  ## Assemble the 8928 x 10000 traffic matrix and per-square totals
	$(PYTHON) -m src.build_matrix --config $(CONFIG) $(ARGS)

eda:  ## Produce exploratory figures, statistics and the selected series
	$(PYTHON) -m src.eda --config $(CONFIG) $(ARGS)

train:  ## Train and tune the three models
	$(PYTHON) -m src.train --config $(CONFIG) $(ARGS)

evaluate:  ## Evaluate on the test week and write metrics, plots and diagnostics
	$(PYTHON) -m src.evaluate --config $(CONFIG) $(ARGS)

report:  ## Export figures and tables into report/
	$(PYTHON) -m src.export_report --config $(CONFIG) $(ARGS)

all: env download ingest matrix eda train evaluate report  ## Full pipeline from raw download

clean-interim:  ## Remove per-day parquet, keeping raw downloads and results
	rm -rf data/interim/*

clean:  ## Remove all generated artefacts except raw downloads
	rm -rf data/interim/* data/processed/traffic_matrix.npy \
	       data/processed/timestamps.parquet data/processed/square_ids.npy \
	       data/processed/totals.parquet \
	       results/figures/* results/predictions/* results/tables/*
