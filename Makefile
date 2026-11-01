.PHONY: install data run test lint format figures synth clean

install:
	pip install -e ".[dev,lstm,data]"

data:
	python scripts/download_data.py

run:
	sleeprisk run

test:
	pytest

lint:
	ruff check src tests scripts
	ruff format --check src tests scripts

format:
	ruff format src tests scripts
	ruff check --fix src tests scripts

figures:
	python scripts/make_figures.py

synth:
	sleeprisk synth --out data/synthetic/dataset

clean:
	rm -rf results/runs data/synthetic .pytest_cache .ruff_cache
	find . -name __pycache__ -prune -exec rm -rf {} +
