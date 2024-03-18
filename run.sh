#!/bin/sh
cd "$(dirname "$0")"
LOG_PREFIX="[Viam OAK module setup]"

echo "$LOG_PREFIX Starting the module."

os=$(uname -s)
arch=$(uname -m)
appimage_path="./viam-camera-oak--aarch64.AppImage"
# Run appimage if Linux aarch64
if [ "$os" = "Linux" ] && [ "$arch" = "aarch64" ] && [ -f "$appimage_path" ]; then
    echo "$LOG_PREFIX Detected system Linux AArch64 and appimage. Attempting to start appimage."
    chmod +x "$appimage_path"
    exec "$appimage_path" "$@"
else
    echo "$LOG_PREFIX No usable appimage was found."
fi

# Else, try running with a virtual environment and source
VENV_NAME="viam-oak-venv"
PYTHON="$VENV_NAME/bin/python"

echo "$LOG_PREFIX Running the module using virtual environment. This requires Python >=3.8.1, pip3, and venv to be installed."

if ! python3 -m venv "$VENV_NAME" >/dev/null 2>&1; then
    echo "$LOG_PREFIX Error: failed to create venv. Please use your system package manager to install python3-venv." >&2
    exit 1
else
    echo "$LOG_PREFIX Created/found venv."
fi

# -qq suppresses extraneous output from pip
echo "$LOG_PREFIX Installing/upgrading Python packages."
if ! "$PYTHON" -m pip install -r requirements.txt -qq; then
    echo "$LOG_PREFIX Error: pip failed to install requirements.txt. Please use your system package manager to install python3-pip." >&2
    exit 1
fi

# Be sure to use `exec` so that termination signals reach the python process,
# or handle forwarding termination signals manually
echo "$LOG_PREFIX Starting module."
exec "$PYTHON" -m src.main "$@"
