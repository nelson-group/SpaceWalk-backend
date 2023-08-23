SHELL:=/bin/bash
#Set-ExecutionPolicy Unrestricted -Scope Process on windows
venv: illustris_python
	python -m venv venv
	source venv/bin/activate && python -m pip install --upgrade pip && pip install -e '.[dev,test]'
	source venv/bin/activate && pre-commit install && pre-commit run
	source venv/bin/activate && pip install -e illustris_python

clean:
	-rm -rf venv
	-rm -rf build
	-rm -rf dist

.PHONY: build
build:
	python -m build --wheel


illustris_python:
	git clone https://github.com/illustristng/illustris_python
