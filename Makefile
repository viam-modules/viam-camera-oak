# Makefile
IMAGE_NAME = appimage-builder-oak
CONTAINER_NAME = appimage-builder-oak
AARCH64_APPIMAGE_NAME = viam-camera-oak-latest-aarch64.AppImage

.PHONY: integration-tests

# Developing
default:
	@echo No make target specified.

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
	go test -c -o oak-integration-tests ./tests/ && \
	mv oak-integration-tests ../
	./oak-integration-tests -module ./run.sh
