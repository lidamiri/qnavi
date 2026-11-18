PYTHON ?= python3
VENV ?= .venv
VENV_PYTHON := $(VENV)/bin/python
PACKAGE := qnavi
RUNS ?= runs
SEED ?= 7
EPISODES ?= 5000
CONFIG ?= configs/default.json
PROJECT := $(notdir $(CURDIR))
VERSION := $(shell $(PYTHON) -c "import re,pathlib;print(re.search(r'version\s*=\s*\"([^\"]+)\"',pathlib.Path('pyproject.toml').read_text()).group(1))" 2>/dev/null || echo 0.0.0)
DIST_DIR := dist
ARCHIVE := $(DIST_DIR)/$(PROJECT)-$(VERSION).zip

.DEFAULT_GOAL := help

.PHONY: help venv install dev test test-fast lint format train train-grid train-cliff train-taxi \
        evaluate render compare sweep plot demo clean distclean archive check

help:
	@echo "qnavi build targets"
	@echo ""
	@echo "  make venv         create the virtual environment in $(VENV)"
	@echo "  make install      install the package and runtime dependencies"
	@echo "  make dev          install the package with development dependencies"
	@echo "  make test         run the full test suite"
	@echo "  make test-fast    run the test suite without slow training tests"
	@echo "  make lint         run static checks"
	@echo "  make format       reformat the source tree"
	@echo "  make train        train an agent from CONFIG=$(CONFIG)"
	@echo "  make train-grid   train on the grid world preset"
	@echo "  make train-cliff  train on the cliff walking preset"
	@echo "  make train-taxi   train on the taxi preset"
	@echo "  make evaluate     evaluate the latest checkpoint in $(RUNS)"
	@echo "  make render       render the greedy policy of the latest checkpoint"
	@echo "  make compare      train every algorithm on the grid world and compare"
	@echo "  make sweep        sweep learning rates for Q-learning"
	@echo "  make plot         plot learning curves from the latest run"
	@echo "  make demo         end to end smoke run that finishes in seconds"
	@echo "  make check        run lint and tests"
	@echo "  make archive      build $(ARCHIVE)"
	@echo "  make clean        remove caches and generated run output"
	@echo "  make distclean    clean plus remove the virtual environment"

venv:
	$(PYTHON) -m venv $(VENV)
	$(VENV_PYTHON) -m pip install --upgrade pip

install: venv
	$(VENV_PYTHON) -m pip install -e .

dev: venv
	$(VENV_PYTHON) -m pip install -e ".[dev]"

test:
	$(VENV_PYTHON) -m pytest -q

test-fast:
	$(VENV_PYTHON) -m pytest -q -m "not slow"

lint:
	$(VENV_PYTHON) -m ruff check src tests
	$(VENV_PYTHON) -m ruff format --check src tests

format:
	$(VENV_PYTHON) -m ruff format src tests
	$(VENV_PYTHON) -m ruff check --fix src tests

check: lint test

train:
	$(VENV_PYTHON) -m $(PACKAGE) train --config $(CONFIG) --output $(RUNS) --seed $(SEED)

train-grid:
	$(MAKE) train CONFIG=configs/gridworld.json

train-cliff:
	$(MAKE) train CONFIG=configs/cliff.json

train-taxi:
	$(MAKE) train CONFIG=configs/taxi.json

evaluate:
	$(VENV_PYTHON) -m $(PACKAGE) evaluate --run-dir $(RUNS) --episodes 200 --seed $(SEED)

render:
	$(VENV_PYTHON) -m $(PACKAGE) render --run-dir $(RUNS)

compare:
	$(VENV_PYTHON) -m $(PACKAGE) compare --config configs/gridworld.json --episodes $(EPISODES) --seed $(SEED) --output $(RUNS)/compare

sweep:
	$(VENV_PYTHON) -m $(PACKAGE) sweep --config configs/gridworld.json --episodes $(EPISODES) --seed $(SEED) --output $(RUNS)/sweep

plot:
	$(VENV_PYTHON) -m $(PACKAGE) plot --run-dir $(RUNS)

demo:
	$(VENV_PYTHON) -m $(PACKAGE) train --config configs/demo.json --output $(RUNS)/demo --seed $(SEED)
	$(VENV_PYTHON) -m $(PACKAGE) evaluate --run-dir $(RUNS)/demo --episodes 50 --seed $(SEED)
	$(VENV_PYTHON) -m $(PACKAGE) render --run-dir $(RUNS)/demo

archive:
	@mkdir -p $(DIST_DIR)
	@rm -f $(ARCHIVE)
	@git archive --format=zip --prefix=$(PROJECT)/ -o $(ARCHIVE) HEAD
	@echo "wrote $(ARCHIVE)"

clean:
	find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache -o -name "*.egg-info" \) -prune -exec rm -rf {} +
	rm -rf $(RUNS) htmlcov .coverage

distclean: clean
	rm -rf $(VENV) $(DIST_DIR) build
