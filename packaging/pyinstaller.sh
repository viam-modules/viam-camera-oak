#!/bin/bash
# This script is run by the make file in the working dir of viam-camera-oak-d
pip3 install --upgrade pip
PYINSTALLER_COMPILE_BOOTLOADER=1 MACOSX_DEPLOYMENT_TARGET=14.0 pip3 install pyinstaller --no-binary pyinstaller
# pip3 install pyinstaller
pip3 install -r requirements.txt
python3 -m pip install --upgrade google-api-python-client  # make sure that pip is linked to pip3 and not pip2 for this pkg
# pyinstaller --onefile --hidden-import="googleapiclient" --add-data="src:." src/main.py
pyinstaller ./packaging/pyinstaller.spec
