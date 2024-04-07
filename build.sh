#!/bin/bash

# Desired Python version
DESIRED_PYTHON_VERSION="3.11"

# Function to find Python path
find_python() {
    which python$1 || which python3 || which python || return 1
}

# Check for the desired Python version and get its path
PYTHON_PATH=$(find_python $DESIRED_PYTHON_VERSION)
if [ $? -ne 0 ]; then
    echo "Python $DESIRED_PYTHON_VERSION is not installed."
    exit 1
fi

echo "Using Python at $PYTHON_PATH"

# Check Python version explicitly
VERSION_CHECK=$($PYTHON_PATH -c "import platform; print(platform.python_version().startswith('$DESIRED_PYTHON_VERSION'))")
if [ $VERSION_CHECK != "True" ]; then
    echo "The Python version is not $DESIRED_PYTHON_VERSION."
    exit 1
fi

# Create virtual environment with the found Python
$PYTHON_PATH -m venv .venv
source .venv/bin/activate

# Ensure pip is installed in the venv
python -m ensurepip

# Install requirements
pip install -r requirements.txt

# Use the Python executable from the venv for PyInstaller
python -m PyInstaller --add-data .venv/lib/python$DESIRED_PYTHON_VERSION/site-packages/ahrs/utils:ahrs/utils --onefile --hidden-import="googleapiclient" src/main.py

# Create a distribution archive
tar -czvf dist/archive.tar.gz dist/main
