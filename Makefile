.PHONY: install data test lint run all

install:
	pip install -e ".[dev]"

data:
	bash scripts/download_data.sh

test:
	pytest -q

lint:
	ruff check src tests

run:
	python -m scamtax.run --data data/final_dataset_output.csv --config configs/default.yaml --out reports

all: install data test run
