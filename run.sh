#!/bin/bash
#!/usr/bin/env python3.8

echo "[Module setup] Setting up module."

cd "$(dirname "$0")"

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if command -v pip3 &> /dev/null; then
        echo "[Module setup] python3-pip installation found."
    else
        echo "[Module setup] python3-pip installation not found. Attempting to install python3-pip."
        sudo apt update
        sudo apt install python3-pip
    fi

    if pip3 freeze | grep -q "virtualenv"; then
        echo "[Module setup] python3-venv installation found."
    else
        echo "[Module setup] python3-venv installation not found. Attempting to install python3-venv."
        sudo apt update
        sudo apt install python3-venv
    fi
fi

if [ -f "$(pwd)/.installed" ]; then
    echo "[Module setup] Dependencies installed. Activating venv."
    source viam-env/bin/activate
    pip3 install --upgrade -r requirements-prod.txt
else
    echo "[Module setup] Installing virtual environment and dependencies."
    python3 -m pip3 install --user virtualenv
    python3 -m venv viam-env
    source viam-env/bin/activate
    pip3 install --upgrade -r requirements-prod.txt
    if [ $? -eq 0 ]; then
        touch "$(pwd)/.installed"
    fi
fi

echo "[Module setup] Setup complete. Starting module process."
# Uses `exec` so that termination signals reach the Python process; handled by Stoppable protocol
exec -a viam-oak-d python3 -m src.main "$@"
