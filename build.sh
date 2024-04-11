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

# Need to tweak pyenv if darwin/amd64 due to Dockerfile pyenv not building with --enable-shared
if [ "$UNAME_S" = "Darwin" ] && [ "$UNAME_M" = "x86_64" ]; then
    # Check if the required Python version is already installed
    if pyenv versions --bare | grep -Fxq $PYENV_VERSION; then
        echo "$PYENV_VERSION is already installed, rebuilding with shared libraries."
        if pyenv uninstall -f $PYENV_VERSION; then
            echo "Successfully uninstalled $PYENV_VERSION."
        else
            echo "Failed to uninstall $PYENV_VERSION. It may not have been installed correctly."
            exit 1
        fi
    fi

    # Set env var to build Python with shared library support
    export PYTHON_CONFIGURE_OPTS="--enable-shared"

    # Re-check if Python version needs to be installed (in case uninstall failed without exiting)
    if ! pyenv versions --bare | grep -Fxq $PYENV_VERSION; then
        pyenv install $PYENV_VERSION
    fi

    pyenv local $PYENV_VERSION
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
