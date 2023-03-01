SHELL:=/bin/bash
#Set-ExecutionPolicy Unrestricted -Scope Process on windows
venv:
	python -m venv venv
	source venv/bin/activate && python -m pip install --upgrade pip && pip install -e '.[dev,test]'
	source venv/bin/activate && pre-commit install && pre-commit run

clean:
	-rm -rf venv
	-rm -rf build
	-rm -rf dist

.PHONY: build
build:
	python -m build --wheel
