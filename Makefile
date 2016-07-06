# Makefile for common tasks

.PHONE: help
help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  init       to initialize pip requirements"
	@echo "  test       to run unit tests"
	@echo "  lint       to lint the source files"
	@echo "  docs       to generate Sphinx html docs"

.PHONY: init
init:
	pip install -r requirements.txt

.PHONY: test
test:
	py.test tests

.PHONY: lint
lint:
	flake8 thrifty/ tests/
	pylint -rn thrifty/*.py tests/*.py

.PHONY: docs
docs:
	cd docs && $(MAKE) html
