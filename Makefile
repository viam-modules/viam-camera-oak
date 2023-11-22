# Makefile
IMAGE_NAME = python:3.10-bookworm
CONTAINER_NAME = appimage-builder-container-aarch64
APPIMAGE_NAME = viam-camera-oak-d--aarch64.AppImage

.PHONY: integration-tests

.DEFAULT_GOAL := setup

# Developing
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

# Packaging
build: build-non-appimage

build-non-appimage: clean
	tar -czf module.tar.gz run.sh requirements.txt src

build-appimage: clean
	docker build -t $(IMAGE_NAME) . && \
	docker run --name $(CONTAINER_NAME) $(IMAGE_NAME) && \
	docker cp $(CONTAINER_NAME):/app/$(APPIMAGE_NAME) ./$(APPIMAGE_NAME) && \
	chmod +x ./${APPIMAGE_NAME} && \
	tar -czf module.tar.gz run.sh $(APPIMAGE_NAME)

clean:
	rm -f *.AppImage && \
	rm -f module.tar.gz && \
	docker container stop $(CONTAINER_NAME) || true && \
	docker container rm $(CONTAINER_NAME) || true
