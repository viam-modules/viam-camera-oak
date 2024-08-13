# Makefile
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
	./oak-integration-tests -module ./local_run.sh -blob ./integration-tests/models/yolov6n_coco_416x416.blob
