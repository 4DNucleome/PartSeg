#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

set -euo pipefail
shopt -s inherit_errexit

cd $SCRIPT_DIR
python -m pytest package/tests
python -m pytest package/tests
python -m pytest package/tests
