# Makefile

.DEFAULT_GOAL := install

install:
	pip install -r requirements-dev.txt

lint:
	black src

lint-check:
	black src --diff
	black src --check

test: test-unit

test-unit:
	pytest unit-tests

test-integration: integration-tests/tests/*
	cd integration-tests && \
	go test -c -o oak-d-integration-tests ./tests/ && \
	mv oak-d-integration-tests ../
	./oak-d-integration-tests -module ./run.sh

build:
	tar -czf module.tar.gz run.sh requirements.txt src
