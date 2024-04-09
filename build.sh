#!/bin/bash

UNAME=$(uname -s)
if [ "$UNAME" = "Linux" ]
then
    echo "Installing venv on Linux"
    # sudo apt-get install -y python3-venv
fi
if [ "$UNAME" = "Darwin" ]
then
    echo "Installing venv on Darwin"
    # brew reinstall python3
    # brew install python3-venv
fi

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
# Dynamically find the Python version directory inside .venv/lib
PYTHON_LIB_PATH=$(find .venv/lib -type d -name "python3.*" -print -quit)
AHRS_UTILS_PATH="$PYTHON_LIB_PATH/site-packages/ahrs/utils"
python3 -m PyInstaller --add-data "$AHRS_UTILS_PATH:ahrs/utils" --onefile --hidden-import="googleapiclient" src/main.py
tar -czvf dist/archive.tar.gz dist/main