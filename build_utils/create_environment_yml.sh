#!/usr/bin/env bash

set -euo pipefail

dependencies=$(bash build_utils/setup_cfg_to_yaml.sh 'install_requires =')

cat <<EOF > environment.yml
name: test
channels:
  - conda-forge
dependencies:
$dependencies
  - python=3.9
  - pip:
    - fonticon-fontawesome6
EOF
