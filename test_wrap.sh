#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd napari
git bisect start
git bisect bad
git bisect good v0.4.14
git bisect
git bisect run bash ../test_napari.sh
git status
