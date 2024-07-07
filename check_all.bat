@echo off

call conda activate venv\ || pause && exit /b

echo #########################################
echo mypy
echo ..checking tools
call python -m mypy tools\ || pause && exit /b
echo ..checking bin
call python -m mypy bin\ || pause && exit /b
echo ..checking word2vec_mt
call python -m mypy src\word2vec_mt\ || pause && exit /b
echo.

echo #########################################
echo pylint
echo ..checking tools
call python -m pylint tools\ || pause && exit /b
echo ..checking bin
call python -m pylint bin\ || pause && exit /b
echo ..checking word2vec_mt
call python -m pylint src\word2vec_mt\ || pause && exit /b
echo.

echo #########################################
echo sphinx api documentation
call python tools\sphinx_api_doc_maker.py || pause && exit /b
echo.

echo #########################################
echo project validation
call python tools\validate_project.py || pause && exit /b
echo.

echo #########################################
echo sphinx
cd docs
call make html || cd .. && pause && exit /b
cd ..
echo.
