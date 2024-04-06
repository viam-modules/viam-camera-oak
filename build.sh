#!/bin/bash
set -e

error_exit()
{
    # Check if the script exits with a non-zero status
    EXIT_STATUS=$?
    if [ "$EXIT_STATUS" -ne 0 ]; then
        echo "Script exited due to an error (status code: $EXIT_STATUS). Please check the output for error messages."
    fi
}

# Trap EXIT signal to call error_exit function
trap error_exit EXIT

UNAME=$(uname -s)

if [ "$UNAME" = "Linux" ]; then
    echo "Installing venv on Linux"
    sudo apt-get install -y python3-venv
elif [ "$UNAME" = "Darwin" ]; then
    echo "Install venv by installing brew package"
    brew install python@3.11
else
    echo "Unsupported operating system: $UNAME"
    exit 1
fi

python3.11 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
python3.11 -m PyInstaller --add-data ".venv/lib/python3.11/site-packages/ahrs/utils:ahrs/utils" --onefile --hidden-import="googleapiclient" src/main.py
tar -czvf dist/archive.tar.gz dist/main
