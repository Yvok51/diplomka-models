#!/bin/bash


SITE_PACKAGES=$(venv/bin/python3 -c "import site; print(site.getsitepackages()[0])")
LV_LIB="${SITE_PACKAGES}/lang2vec/lang2vec.py"
sed -i 's/feature_database = np.load(filename, encoding="latin1").item()/feature_database = np.load(filename, encoding="latin1", allow_pickle=True).item()/g' "${LV_LIB}"