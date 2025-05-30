.PHONY: test lint doc

test:
	python -m pytest

lint:
	black --check pynapple
	isort --check pynapple --profile black
	flake8 pynapple --max-complexity 10
	black --check tests
	isort --check tests --profile black

doc:
	cd doc && make html

docs: doc

all: lint test doc
