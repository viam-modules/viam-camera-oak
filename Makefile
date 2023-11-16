# Makefile
.PHONY: integration-tests

.DEFAULT_GOAL := setup

setup:
	pip install -r requirements-dev.txt

lint: lint-fix

lint-fix:
	black src

lint-check:
	black src --diff --check

test: unit-tests

unit-tests:
	pytest tests

integration-tests: integration-tests/tests/*
	cd integration-tests && \
	go test -c -o oak-d-integration-tests ./tests/ && \
	mv oak-d-integration-tests ../
	./oak-d-integration-tests -module ./run.sh

build:
	tar -czf module.tar.gz run.sh requirements.txt src
