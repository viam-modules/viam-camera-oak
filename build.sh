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

python3 -m venv .build-env
source .build-env/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
# Dynamically find the Python version directory inside .build-env/lib
PYTHON_LIB_PATH=$(find .build-env/lib -type d -name "python3.*" -print -quit)
AHRS_UTILS_PATH="$PYTHON_LIB_PATH/site-packages/ahrs/utils"
python3 -m PyInstaller --add-data "$AHRS_UTILS_PATH:ahrs/utils" --onefile --hidden-import="googleapiclient" src/main.py
tar -czvf dist/archive.tar.gz dist/main
