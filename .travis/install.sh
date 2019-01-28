#!/usr/bin/env bash

if [[ $TRAVIS_OS_NAME == 'osx' ]]; then

    # Install some custom requirements on macOS
    # e.g. brew install pyenv-virtualenv
    brew update
    brew upgrade pyenv

    case "${TOXENV}" in
        py36)
            pyenv install 3.6.8
            pyenv global 3.6.8
            # Install some custom Python 3.6 requirements on macOS
            ;;
        py37)
            pyenv install 3.7.2
            pyenv global 3.7.2
            # Install some custom Python 3.7 requirements on macOS
            ;;
    esac
    pip install pytest
else
    echo "linux"
    # Install some custom requirements on Linux
fi

pip install pip==18.1
pip install pyinstaller cython numpy
pip install ./package