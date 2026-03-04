# Run this from PowerShell: .\format.ps1
# Black
.\.venv\Scripts\black . --exclude "/(\.venv|chunkr)/"
# isort
.\.venv\Scripts\isort . --skip .venv --skip chunkr
# flake8
.\.venv\Scripts\python.exe -m flake8 . --exclude=.venv,chunkr
