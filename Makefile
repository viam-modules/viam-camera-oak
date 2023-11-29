# Makefile
IMAGE_NAME = appimage-builder-oak-d
CONTAINER_NAME = appimage-builder-oak-d
AARCH64_APPIMAGE_NAME = viam-camera-oak-d--aarch64.AppImage

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
	go test -c -o oak-d-integration-tests ./tests/ && \
	mv oak-d-integration-tests ../
	./oak-d-integration-tests -module ./run.sh

# Packaging
build: build-non-appimage

build-non-appimage: clean
	tar -czf module.tar.gz run.sh requirements.txt src

build-appimage-aarch64: clean
	docker build -t $(IMAGE_NAME) .
	docker run --name $(CONTAINER_NAME) $(IMAGE_NAME)
	docker cp $(CONTAINER_NAME):/app/$(AARCH64_APPIMAGE_NAME) ./$(AARCH64_APPIMAGE_NAME)
	chmod +x ./${AARCH64_APPIMAGE_NAME}
	tar -czf module.tar.gz run.sh $(AARCH64_APPIMAGE_NAME)

clean:
	rm -f $(AARCH64_APPIMAGE_NAME)
	rm -f module.tar.gz
	docker container stop $(CONTAINER_NAME) || true
	docker container rm $(CONTAINER_NAME) || true
