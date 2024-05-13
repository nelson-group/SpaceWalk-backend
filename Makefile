SHELL:=/bin/bash
#Set-ExecutionPolicy Unrestricted -Scope Process on windows
venv: il
	python -m venv venv
	source venv/bin/activate && python -m pip install --upgrade pip && pip install -e '.[dev,test]'
	source venv/bin/activate && pip install -e ./il

.PHONY: clean
clean:
	-rm -rf venv
	-rm -rf build
	-rm -rf dist

.PHONY: build
build:
	python -m build --wheel


il:
	git clone https://github.com/illustristng/illustris_python il
