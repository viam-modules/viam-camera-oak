#!/bin/sh
cd "$(dirname "$0")"

# Create a virtual environment to run our code
VENV_NAME="viam-oak-d-venv"
PYTHON="$VENV_NAME/bin/python"
LOG_PREFIX="[Viam OAK-D setup]"
ENV_ERROR="$LOG_PREFIX Error: this module requires Python >=3.8.1, pip3, and virtualenv to be installed."

if ! python3 -m venv "$VENV_NAME" >/dev/null 2>&1; then
    echo "$LOG_PREFIX Warning: failed to create virtualenv."
    if command -v apt-get >/dev/null; then
        echo "$LOG_PREFIX Detected Debian/Ubuntu. Attempting to install python3-venv automatically."
        SUDO="sudo"
        if ! command -v "$SUDO" >/dev/null; then
            SUDO=""
        fi
        if ! apt info python3-venv >/dev/null 2>&1; then
            echo "$LOG_PREFIX Package info not found, trying apt update."
            "$SUDO" apt -qq update >/dev/null
        fi
        "$SUDO" apt install -qqy python3-venv >/dev/null 2>&1
        if ! python3 -m venv "$VENV_NAME" >/dev/null 2>&1; then
            echo "$ENV_ERROR" >&2
            exit 1
        fi
    else
        # some other OS we cannot get python3-venv for in script
        echo "$ENV_ERROR" >&2
        exit 1
    fi
else
    echo "$LOG_PREFIX Created/found virtualenv."
fi

# Remove -U if viam-sdk should not be upgraded whenever possible
# -qq suppresses extraneous output from pip
echo "$LOG_PREFIX Installing/upgrading Python packages."
if ! "$PYTHON" -m pip3 install -r requirements.txt -Uqq; then
    echo "$LOG_PREFIX Error: pip3 failed to install requirements.txt." 
    exit 1
fi

# Be sure to use `exec` so that termination signals reach the python process,
# or handle forwarding termination signals manually
echo "$LOG_PREFIX Starting module."
exec "$PYTHON" -m src.main "$@"
