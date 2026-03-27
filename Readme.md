# PartSeg

![Contributions](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)
![Tests](https://github.com/4DNucleome/PartSeg/workflows/Tests/badge.svg?branch=develop)
[![PyPI version](https://badge.fury.io/py/PartSeg.svg)](https://badge.fury.io/py/PartSeg)
[![Anaconda version](https://anaconda.org/conda-forge/partseg/badges/version.svg)](https://anaconda.org/conda-forge/partseg)
[![Python Version](https://img.shields.io/pypi/pyversions/partseg.svg)](https://pypi.org/project/partseg)
[![Documentation Status](https://readthedocs.org/projects/partseg/badge/?version=latest)](https://partseg.readthedocs.io/en/latest/?badge=latest)
[![Azure Pipelines Build Status](https://dev.azure.com/PartSeg/PartSeg/_apis/build/status/4DNucleome.PartSeg?branchName=develop)](https://dev.azure.com/PartSeg/PartSeg/_build/latest?definitionId=1&branchName=develop)
[![DOI](https://zenodo.org/badge/166421141.svg)](https://zenodo.org/badge/latestdoi/166421141)
[![Publication DOI](https://img.shields.io/badge/Publication%20DOI-10.1186%2Fs12859--021--03984--1-blue)](https://doi.org/10.1186/s12859-021-03984-1)
[![Licence: BSD3](https://img.shields.io/github/license/4DNucleome/PartSeg)](https://github.com/4DNucleome/PartSeg/blob/master/License.txt)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v1.json)](https://github.com/charliermarsh/ruff)
[![CodeQL](https://github.com/4DNucleome/PartSeg/actions/workflows/codeql-analysis.yml/badge.svg?branch=develop)](https://github.com/4DNucleome/PartSeg/actions/workflows/codeql-analysis.yml)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/f9b0f1eb2c92486d9efd99ed5b2ef326)](https://www.codacy.com/gh/4DNucleome/PartSeg/dashboard?utm_source=github.com&utm_medium=referral&utm_content=4DNucleome/PartSeg&utm_campaign=Badge_Grade)
[![codecov](https://codecov.io/gh/4DNucleome/PartSeg/branch/develop/graph/badge.svg?token=nbAbkOAe1C)](https://codecov.io/gh/4DNucleome/PartSeg)
[![DeepSource](https://deepsource.io/gh/4DNucleome/PartSeg.svg/?label=active+issues&show_trend=true&token=RuuHPIzqyqGaU-bKtOKPFWTg)](https://deepsource.io/gh/4DNucleome/PartSeg/?ref=repository-badge)

PartSeg is a GUI and a library for segmentation algorithms. PartSeg also provide napari plugins for IO and labels measurement.

This application is designed to help biologist with segmentation based on threshold and connected components.

![interface](images/roi_analysis.png)

## Tutorials

- Tutorial: **Chromosome 1 (as gui)** [link](https://github.com/4DNucleome/PartSeg/blob/master/tutorials/tutorial-chromosome-1/tutorial-chromosome1_16.md)
- Data for chromosome 1 tutorial [link](https://4dnucleome.cent.uw.edu.pl/PartSeg/Downloads/PartSeg_samples.zip)
- Tutorial: **Different neuron types (as library)** [link](https://github.com/4DNucleome/PartSeg/blob/master/tutorials/tutorial_neuron_types/Neuron_types_example.ipynb)

## Installation

- From binaries:

  - [Windows](https://github.com/4DNucleome/PartSeg/releases/latest/download/PartSeg-windows.zip) (build on Windows 10)
  - [Linux](https://github.com/4DNucleome/PartSeg/releases/latest/download/PartSeg-linux.zip) (build on Ubuntu 20.04)
  - [macOS](https://github.com/4DNucleome/PartSeg/releases/latest/download/PartSeg-macos.zip) (build on macOS 13)
  - [macOS arm](https://github.com/4DNucleome/PartSeg/releases/latest/download/PartSeg-macos-arm64.zip) (build on macOS 14)
    There are reported problems with permissions systems on macOS. If you have a problem with starting the application, please try to run it from the terminal.

- With pip:

  - From pypi: `pip install PartSeg[all]`
  - From repository: `pip install git+https://github.com/4DNucleome/PartSeg.git`

- With conda:

  - `conda install -c conda-forge partseg`
  - `mamba install -c conda-forge partseg` - As mamba is faster than conda

- With napari:

  If you do not know how to setup python environment on your system you may use [napari](https://napari.org/) to run PartSeg.
  It is a GUI for scientific image analysis. PartSeg is also a plugin for napari so could be installed from plugin dialog.
  To install napari bundle please download it [napari bundle](https://github.com/napari/napari/releases/latest)
  and follow [installation instructions](https://napari.org/stable/tutorials/fundamentals/installation.html#install-as-a-bundled-app).

Installation troubleshooting information could be found in wiki: [wiki](https://github.com/4DNucleome/PartSeg/wiki/Instalation-troubleshoot).
If this information does not solve problem you can open [issue](https://github.com/4DNucleome/PartSeg/issues).

### Qt 6 support

PartSeg development branch support (and stable since 0.15.0) has experimental Qt6 support. Test are passing but not whole GUI code is covered by tests. Inf you Find any problem please report it.

## Running

If you downloaded binaries, run the `PartSeg` (or `PartSeg.exe` for Windows) file inside the `PartSeg` folder

If you installed from repository or from pip, you can run it with `PartSeg` command or `python -m PartSeg`.
First option does not work on Windows.

PartSeg export few commandline options:

- `--no_report` - disable error reporting
- `--no_dialog` - disable error reporting and error dialog. Use only when running from terminal.
- `roi` - skip launcher and start *ROI analysis* gui
- `mask`- skip launcher and start *ROI mask* gui

## napari plugin

PartSeg provides napari plugins for io to allow reading projects format in napari viewer.

## Save Format

Saved projects are tar files compressed with gzip or bz2.

Metadata is saved in data.json file (in json format).
Images/masks are saved as \*.npy (numpy array format).

## Interface

Launcher. Choose the program that you will launch:

![launcher](images/launcher.png)

Main window of Segmentation Analysis:

![interface](images/roi_analysis.png)

Main window of Segmentation Analysis with view on measurement result:

![interface](images/roi_analysis2.png)

Window for creating a set of measurements:

![statistics](images/measurement.png)

Main window of Mask Segmentation:

![mask interface](images/roi_mask.png)

## Laboratory

Laboratory of Functional and Structural Genomics
[http://4dnucleome.cent.uw.edu.pl/](http://4dnucleome.cent.uw.edu.pl/)

## Cite as

Bokota, G., Sroka, J., Basu, S. et al. PartSeg: a tool for quantitative feature extraction
from 3D microscopy images for dummies. BMC Bioinformatics 22, 72 (2021).
[https://doi.org/10.1186/s12859-021-03984-1](https://doi.org/10.1186/s12859-021-03984-1)
