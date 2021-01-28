#!/usr/bin/env bash
mkdir -p ~/cache/
wget -N https://4dnucleome.cent.uw.edu.pl/PartSeg/Downloads/test_data.tbz2 -P ~/cache/

tar -jxf ~/cache/test_data.tbz2 -C .
