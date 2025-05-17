#!/bin/bash

python3 -m venv venv
venv/bin/pip install -r requirements.txt
venv/bin/pip install -r requirements-pytorch.txt

SITE_PACKAGES=$(venv/bin/python3 -c "import site; print(site.getsitepackages()[0])")
LV_LIB="${SITE_PACKAGES}/lang2vec/data"
LV_DISTANCES="${LV_LIB}/distances.zip"
curl http://www.cs.cmu.edu/~aanastas/files/distances.zip --output "${LV_DISTANCES}"
unzip "${LV_DISTANCES}" -d "${LV_LIB}"

./install.sh
