#!/bin/sh
python3 -m venv venv
if [[ $(uname -s) == Linux  ]]; then
    source venv/bin/activate
else
    source venv/Scripts/activate
fi
pip3 install -r requirements.txt

python analysis.py