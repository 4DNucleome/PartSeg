#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

set -euo pipefail

cd $SCRIPT_DIR
python -m pytest package/tests --no-cov
python -m pytest package/tests --no-cov
python -m pytest package/tests --no-cov
