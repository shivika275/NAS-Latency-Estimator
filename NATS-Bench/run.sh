#!/bin/bash
python3 -m pip install -r requirements.txt
python3 -m pip install gdown 
git clone https://github.com/D-X-Y/AutoDL-Projects
cd AutoDL-Projects
python3 -m pip install .
cd ../
gdown https://drive.google.com/drive/u/0/folders/1zjB6wMANiKwB2A1yil2hQ8H_qyeSe2yt?sort=13&direction=a

python3 SMR_torch.py