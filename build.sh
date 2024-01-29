#!/bin/bash
UNAME=$(uname -s)

if [ "$UNAME" = "Linux" ]
then
    echo "Installing venv on Linux"
    sudo apt-get install -y python3.11-venv
fi
if [ "$UNAME" = "Darwin" ]
then
    echo "Installing venv on Darwin"
    brew install python3.11-venv
fi
python3 -m venv .venv
. .venv/bin/activate
pip3 install -r requirements.txt
python3 -m PyInstaller --add-data /root/project/.venv/lib/python3.11/site-packages/ahrs/utils/WMM2020:ahrs/utils/WMM2020 --onefile --hidden-import="googleapiclient" src/main.py
tar -czvf dist/archive.tar.gz dist/main