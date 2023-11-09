#!/bin/bash

echo "[Module setup] Setting up module."
OS=$(uname -s)
cd "$(dirname "$0")"

if command -v pip3 &> /dev/null; then
    echo "[Module setup] python3-pip installation found."
else
    echo "[Module setup] python3-pip installation not found."
    if [[ "$OS" == "Linux" ]]; then
        echo "[Module setup] Attempting to install python3-pip on $OS machine."
        sudo apt update
        sudo apt install python3-pip -y
    elif [[ "$OS" == "Darwin"* ]]; then
        echo "[Module setup] Attempting to install python3-pip on $OS machine."
        brew update
        brew install python3
    else
        echo "[Module setup] Error: Could not identify $OS for installation."
        exit 1
    fi
fi

if pip3 freeze | grep -q "virtualenv"; then
    echo "[Module setup] python3-venv installation found."
else
    if [[ "$OS" == "Linux" ]]; then
        echo "[Module setup] python3-venv installation not found. Attempting to install python3-venv."
        sudo apt update
        sudo apt install python3-venv -y
    elif [[ "$OS" == "Darwin"* ]]; then
        pip3 install virtualenv -y
    else
        echo "[Module setup] Error: Could not identify $OS for installation."
        exit 1
    fi
fi

echo "[Module setup] Making virtual environment and installing dependencies."
python3 -m venv viam-env
source viam-env/bin/activate
pip3 install -r requirements.txt

echo "[Module setup] Setup complete. Starting module process."
# Uses `exec` so that termination signals reach the Python process; handled by Stoppable protocol
exec -a viam-oak-d python3 -m src.main "$@"
