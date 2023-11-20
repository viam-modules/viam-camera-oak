# Makefile
IMAGE_NAME = ubuntu:arm64
CONTAINER_NAME = appimage-builder-container-arm64
OUTPUT_FILE = viam-camera-oak-d-0.0.1-aarch64.AppImage

.PHONY: integration-tests

.DEFAULT_GOAL := install

install:
	pip install -r requirements-dev.txt

lint:
	black src

lint-check:
	black src --diff
	black src --check

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

appimage:
	docker build -t $(IMAGE_NAME) . && \
	docker run --name $(CONTAINER_NAME) $(IMAGE_NAME) && \
	docker cp $(CONTAINER_NAME):/app/$(OUTPUT_FILE) ./$(OUTPUT_FILE) && \
	chmod +x ./${OUTPUT_FILE}
