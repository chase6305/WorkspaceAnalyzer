.PHONY: install test lint format build clean

install:
	python -m pip install -e '.[all,dev]'

test:
	pytest -q

lint:
	ruff check src tests examples
	ruff format --check src tests examples

format:
	ruff check --fix src tests examples
	ruff format src tests examples

build:
	python -m build

clean:
	python -c "import shutil; [shutil.rmtree(path, ignore_errors=True) for path in ('build', 'dist', '.pytest_cache', '.ruff_cache')]"
