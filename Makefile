# Makefile
APPIMAGE_BUILDER_IMAGE = appimage-builder-image
APPIMAGE_BUILDER_CONTAINER = appimage-builder-container
AARCH64_APPIMAGE_NAME = viam-camera-oak-d--aarch64.AppImage

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
	./oak-d-integration-tests -module ./packaging/run.sh

# Packaging
build: build-source

build-source: clean
	tar -czf module.tar.gz ./packaging/run.sh requirements.txt src

# make sure you're in a dev venv on a Macbook before running to ensure build success
build-pyinstaller: clean
	./packaging/pyinstaller.sh

build-appimage: clean
	docker build -f ./packaging/Dockerfile.appimage-builder -t $(APPIMAGE_BUILDER_IMAGE) . && \
	docker run --name $(APPIMAGE_BUILDER_CONTAINER) $(APPIMAGE_BUILDER_IMAGE) && \
	docker cp $(APPIMAGE_BUILDER_CONTAINER):/app/$(AARCH64_APPIMAGE_NAME) ./$(AARCH64_APPIMAGE_NAME) && \
	chmod +x ./${AARCH64_APPIMAGE_NAME} && \
	tar -czf module.tar.gz ./packaging/run.sh $(AARCH64_APPIMAGE_NAME)

clean:
	rm -f *.AppImage && \
	rm -f module.tar.gz && \
	rm -r build && \
	rm -r dist && \
	docker container stop $(APPIMAGE_BUILDER_CONTAINER) || true && \
	docker container rm $(APPIMAGE_BUILDER_CONTAINER) || true
