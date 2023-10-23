#!/bin/bash
cd "$(dirname "$0")"

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if apt list --installed | grep -q "python3-pip"; then
        echo "[Script] python3-pip installation found"
    else
        echo "[Script] python3-pip installation not found; attempting to install python3-pip"
        sudo apt update
        sudo apt install python3-pip
    fi

    if apt list --installed | grep -q "python3-venv"; then
        echo "[Script] python3-venv installation found"
    else
        echo "[Script] python3-venv installation not found; attempting to install python3-venv"
        sudo apt update
        sudo apt install python3-venv
    fi
fi

if [ -f "$(pwd)/.installed" ]; then
    source viam-env/bin/activate
else
    echo "[Script] Installing virtual environment and dependencies..."
    python3 -m pip install --user virtualenv
    python3 -m venv viam-env
    source viam-env/bin/activate
    pip3 install --upgrade -r requirements.txt
    if [ $? -eq 0 ]; then
        touch "$(pwd)/.installed"
    fi
fi

# Uses `exec` so that termination signals reach the Python process; handled by Stoppable protocol
exec python3 -m src.main "$@"
