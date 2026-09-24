# ffs needs 3.10+ (pyproject: requires-python = ">=3.10"), but macOS still ships
# 3.9 as `python3`, where featrixsphere fails to import at all. Pick the newest
# suitable interpreter on PATH; override with `make test PYTHON=python3.12`.
PYTHON ?= $(shell for p in python3.13 python3.12 python3.11 python3.10 python3; do \
	command -v $$p >/dev/null 2>&1 \
	  && $$p -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null \
	  && { echo $$p; break; }; \
	done)
PYTHON := $(if $(PYTHON),$(PYTHON),python3)
VERSION := $(shell $(PYTHON) -c "import ffs; print(ffs.__version__)")

.PHONY: help clean build publish-pypi test lint

help:
	@echo "Targets:"
	@echo "  clean          Remove build artifacts"
	@echo "  build          Build sdist and wheel"
	@echo "  publish-pypi   Build and upload to PyPI (featrix-shell)"
	@echo "  test           Run tests"
	@echo "  lint           Run ruff linter"
	@echo "  install        Install in editable mode"

clean:
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

build: clean
	$(PYTHON) -m build

publish-pypi: build
	@echo "Publishing featrix-shell $(VERSION) to PyPI..."
	$(PYTHON) -m twine upload dist/*

test:
	$(PYTHON) -m pytest tests/ -v

lint:
	$(PYTHON) -m ruff check ffs/

install:
	$(PYTHON) -m pip install -e .
