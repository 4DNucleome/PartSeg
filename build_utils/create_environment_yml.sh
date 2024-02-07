#!/usr/bin/env bash

set -euo pipefail

dependencies=$(python build_utils/pyproject_toml_to_yaml.py)

cat <<EOF > environment.yml
name: test
channels:
  - conda-forge
dependencies:
$dependencies
  - python=3.11
EOF
