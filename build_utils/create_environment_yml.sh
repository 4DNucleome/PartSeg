#!/usr/bin/env bash

set -euo pipefail

dependencies=$(python build_utils/pyproject_toml_to_yaml.py)

cat <<EOF > environment.yml
name: test
channels:
  - conda-forge
dependencies:
$dependencies
  - pyqt
  - python=3.12
EOF
