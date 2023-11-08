# Makefile

.DEFAULT_GOAL := install

install:
	pip install -r requirements-dev.txt

lint:
	black src

test:
	pytest tests
