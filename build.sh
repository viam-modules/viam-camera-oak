#!/bin/bash

UNAME=$(uname -s)

if [ "$UNAME" = "Linux" ]; then
    echo "Installing venv on Linux"
    export DEBIAN_FRONTEND=noninteractive
    sudo apt-get update && sudo apt-get install -y --no-install-recommends software-properties-common
    sudo add-apt-repository ppa:deadsnakes/ppa -y
    sudo apt-get update
    sudo apt-get install -y --no-install-recommends python3.12
    
    # sudo apt-get install -y --no-install-recommends python3.12-venv
    
elif [ "$UNAME" = "Darwin" ]; then
    echo "Installing venv on Darwin"
    # brew update && brew upgrade
    # brew reinstall python@3.12
    # brew link --overwrite python@3.12
fi

python3.12 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
python3 -m PyInstaller --add-data "$(python3 -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())')/ahrs/utils:ahrs/utils" --onefile --hidden-import="googleapiclient" src/main.py
tar -czvf dist/archive.tar.gz dist/main