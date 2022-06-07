#!/usr/bin/env bash

set -euo pipefail

dependencies=$(bash build_utils/setup_cfg_to_yaml.sh 'install_requires =')

cat <<EOF > environment.yml
name: test
channels:
  - conda-forge
dependencies:
$dependencies
  - PySide2>=5.12.3
  - python=3.9
EOF
