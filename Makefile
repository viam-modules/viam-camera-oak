# Makefile

.DEFAULT_GOAL := test

lint:
	black src

test:
	pytest tests
