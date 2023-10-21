#!/bin/bash
python3 -m pip install -r requirements.txt
python3 -m pip install gdown 
git clone https://github.com/D-X-Y/AutoDL-Projects
cd AutoDL-Projects
python3 -m pip install .
cd ../