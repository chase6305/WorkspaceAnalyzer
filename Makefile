.PHONY: install test test-robot test-workflow check lint format build clean

PYTHON ?= python
BUILD_FLAGS ?=

install:
	python -m pip install -e '.[all,dev]'

test:
	PYTHONPATH=src $(PYTHON) -m pytest -q -ra --junitxml=outputs/test-results.xml

test-robot:
	PYTHONPATH=src $(PYTHON) -c "import argparse; from workspace_analyzer.presets import default_robot_urdf, require_robot_urdf; require_robot_urdf(default_robot_urdf(), argparse.ArgumentParser())"
	PYTHONPATH=src $(PYTHON) -m pytest -q -ra tests/test_marvin_integration.py --junitxml=outputs/marvin-tests.xml

test-workflow:
	PYTHONPATH=src $(PYTHON) examples/reachability_workflow.py --urdf tests/fixtures/cartesian_stage.urdf --robustness-test --boundary-test --stability-test --stability-initial-guesses 2 --candidate-diversity --select-ik-solutions joint_limit_margin --output-dir outputs/test-workflow

check: lint test test-workflow build

lint:
	ruff check src tests examples
	ruff format --check src tests examples

format:
	ruff check --fix src tests examples
	ruff format src tests examples

build:
	$(PYTHON) -m build $(BUILD_FLAGS)

clean:
	python -c "import shutil; [shutil.rmtree(path, ignore_errors=True) for path in ('build', 'dist', '.pytest_cache', '.ruff_cache')]"
