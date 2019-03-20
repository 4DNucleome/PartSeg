#!/usr/bin/env bash

cd /tmp
wget https://www.python.org/ftp/python/3.6.8/Python-3.6.8.tar.xz
tar xf Python-3.6.8.tar.xz
cd Python-3.6.8
./configure --prefix=/opt/python --enable-shared
make
make install