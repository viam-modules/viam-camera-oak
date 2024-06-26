#!/bin/bash
set -e

eval "$(pyenv init --path)"
eval "$(pyenv virtualenv-init -)"
UNAME_S=$(uname -s)
UNAME_M=$(uname -m)

if [ -z "$PYENV_VERSION" ]; then
    echo "Warning: PYENV_VERSION is not set. Defaulting to Python 3.11."
    PYENV_VERSION="3.11"
fi

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
# Dynamically find the Python version directory inside .venv/lib
PYTHON_LIB_PATH=$(find .venv/lib -type d -name "python3.*" -print -quit)
AHRS_UTILS_PATH="$PYTHON_LIB_PATH/site-packages/ahrs/utils"
python -m PyInstaller --add-data "$AHRS_UTILS_PATH:ahrs/utils" --onefile --hidden-import="googleapiclient" src/main.py
tar -czvf dist/archive.tar.gz dist/main
