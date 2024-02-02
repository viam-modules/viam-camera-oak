#!/bin/bash
UNAME=$(uname -s)

if [ "$UNAME" = "Linux" ]
then
    echo "Installing venv on Linux"
    sudo apt-get install -y python3.11-venv
    SITE_PKG=/root/project/.venv/lib/python3.11/site-packages/ahrs
fi
if [ "$UNAME" = "Darwin" ]
then
    echo "Installing venv on Darwin"
    brew install python3.11-venv
    SITE_PKG=~/Library/Python/3.11/lib/python/site-packages
fi
python3 -m venv .venv
. .venv/bin/activate
pip3 install -r requirements.txt
ls -R 
python3 -m PyInstaller --add-data $(SITE_PKG)/ahrs/utils:ahrs/utils --onefile --hidden-import="googleapiclient" src/main.py
tar -czvf dist/archive.tar.gz dist/main