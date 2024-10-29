lint:
	@echo "Running pylint"
	python -m black --line-length 120 src/*.py

instructor:
	@echo "Running instructor"
	python -m src.instructor_tut
