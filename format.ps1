# Windows PowerShell version (format.ps1)
# Run .\format.ps1 in bash
.\.venv\Scripts\black . --exclude .venv
.\.venv\Scripts\isort . --skip .venv
.\.venv\Scripts\python.exe -m flake8 . --exclude=.venv
