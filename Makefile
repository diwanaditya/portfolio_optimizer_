PYTHON ?= python3
VENV ?= .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest

.PHONY: install compile test test-quick lint typecheck security benchmark verify clean

install:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

compile:
	$(PY) -m compileall -q portfolio_optimizer start_live_trading.py

test:
	$(PYTEST) -q

test-quick:
	$(PYTEST) -q -x

lint:
	$(PY) -m compileall -q portfolio_optimizer start_live_trading.py

# The repository currently has no separate type-checker configuration; keep
# this target honest and deterministic until mypy/pyright is introduced.
typecheck:
	$(PY) -m compileall -q portfolio_optimizer start_live_trading.py

security:
	$(PY) -m pip_audit -r requirements.txt

benchmark:
	$(PY) examples/benchmark_report.py

verify: compile test lint typecheck security

clean:
	rm -rf $(VENV) .pytest_cache **/__pycache__ **/*.pyc
