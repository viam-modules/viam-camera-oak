#!/bin/sh
cd `dirname $0`

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if apt list --installed | grep -q "python3-pip"; then
        echo "python3-pip installation found; installing dependencies"
        # Add your actions for when python3-pip is already installed
    else
        echo "python3-pip installation not found; attempting to install python3-pip"
        sudo apt update
        sudo apt install python3-pip
        # Add your actions for when python3-pip needs to be installed
    fi
fi


if [ -f .installed ]
  then
    source viam-env/bin/activate
  else
    python3 -m pip install --user virtualenv
    python3 -m venv viam-env
    source viam-env/bin/activate
    pip3 install --upgrade -r requirements.txt
    if [ $? -eq 0 ]
      then
        touch .installed
    fi
fi

# Be sure to use `exec` so that termination signals reach the python process,
# or handle forwarding termination signals manually
exec python3 -m src.main $@