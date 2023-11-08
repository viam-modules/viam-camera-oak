# Makefile

.DEFAULT_GOAL := install

install:
	pip install -r requirements-dev.txt

lint:
	black src

lint-check:
	black src --diff
	black src --check

test:
	pytest tests
