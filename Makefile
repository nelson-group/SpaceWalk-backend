venv:
	python -m venv venv
	source venv/bin/activate && python -m pip install --upgrade pip && pip install -e '.[dev,test]'
	source venv/bin/activate && pre-commit install && pre-commit run

clean:
	-rm -rf venv


