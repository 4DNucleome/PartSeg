#!/usr/bin/env bash

if [ -d /tmp/PartSeg ]; then
    cd /tmp/PartSeg
    git pull
else
    git clone https://github.com/4DNucleome/PartSeg.git /tmp/PartSeg
    cd /tmp/PartSeg
fi

export PATH=/home/partseg/python/bin:${PATH}
export LD_LIBRARY_PATH=/home/partseg/python/lib:${LD_LIBRARY_PATH}

pip3 install numpy cython pyinstaller pytest pytest-qt pytest-xvfb tox
pip3 install .

python3 /tmp/create_build.py /tmp/PartSeg