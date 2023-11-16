#!/bin/sh
cd "$(dirname "$0")"

# Create a virtual environment to run our code
VENV_NAME="viam-oak-d-venv"
PYTHON="$VENV_NAME/bin/python"
LOG_PREFIX="[Viam OAK-D setup]"
ENV_ERROR="$LOG_PREFIX Error: this module requires Python >=3.8.1, pip3, and virtualenv to be installed."

if ! python3 -m venv "$VENV_NAME" >/dev/null 2>&1; then
    echo "$LOG_PREFIX Warning: failed to create virtualenv."
    echo "$ENV_ERROR" >&2
    exit 1
else
    echo "$LOG_PREFIX Created/found virtualenv."
fi

# Remove -U if viam-sdk should not be upgraded whenever possible
# -qq suppresses extraneous output from pip
echo "$LOG_PREFIX Installing/upgrading Python packages."
if ! "$PYTHON" -m pip install -r requirements.txt -Uqq; then
    echo "$LOG_PREFIX Error: pip failed to install requirements.txt. Please make sure pip is properly installed on your machine." >&2
    exit 1
fi

# Be sure to use `exec` so that termination signals reach the python process,
# or handle forwarding termination signals manually
echo "$LOG_PREFIX Starting module."
exec "$PYTHON" -m src.main "$@"
