#!/bin/bash
echo "---- code check ---------"
echo "pyspelling"
pyspelling
echo "ruff"
ruff check --fix $(git ls-files '*.py')
echo "pycodestyle"
pycodestyle $(git ls-files '*.py')
echo "pylint"
pylint $(git ls-files '*.py')
echo "mypy"
mypy $(git ls-files '*.py')
echo "pytests"
pytest tests -s
