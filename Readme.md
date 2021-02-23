# PartSeg

![Tests](https://github.com/4DNucleome/PartSeg/workflows/Tests/badge.svg?branch=master)
[![PyPI version](https://badge.fury.io/py/PartSeg.svg)](https://badge.fury.io/py/PartSeg)
[![Documentation Status](https://readthedocs.org/projects/partseg/badge/?version=latest)](https://partseg.readthedocs.io/en/latest/?badge=latest)
[![Azure Pipelines Build Status](https://dev.azure.com/PartSeg/PartSeg/_apis/build/status/4DNucleome.PartSeg?branchName=master)](https://dev.azure.com/PartSeg/PartSeg/_build/latest?definitionId=1&branchName=master)
[![DOI](https://zenodo.org/badge/166421141.svg)](https://zenodo.org/badge/latestdoi/166421141)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/4DNucleome/PartSeg.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/4DNucleome/PartSeg/alerts/)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/4DNucleome/PartSeg.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/4DNucleome/PartSeg/context:python)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/6c6bef8ebb6a4785a7a1a2da88524661)](https://www.codacy.com/manual/Czaki/PartSeg?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=4DNucleome/PartSeg&amp;utm_campaign=Badge_Grade)
[![Requirements Status](https://requires.io/github/4DNucleome/PartSeg/requirements.svg?branch=develop)](https://requires.io/github/4DNucleome/PartSeg/requirements/?branch=master)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![codecov](https://codecov.io/gh/4DNucleome/PartSeg/branch/master/graph/badge.svg?token=nbAbkOAe1C)](https://codecov.io/gh/4DNucleome/PartSeg)

PartSeg is a GUI and a library for segmentation algorithms.

This application is designed to help biologist with segmentation based on threshold and connected components.

![interface](images/roi_analysis.png)

## Tutorials

-   Tutorial: **Chromosome 1 (as gui)** [link](https://github.com/4DNucleome/PartSeg/blob/master/tutorials/tutorial-chromosome-1/tutorial-chromosome1_16.md)
-   Data for chromosome 1 tutorial [link](https://4dnucleome.cent.uw.edu.pl/PartSeg/Downloads/PartSeg_samples.zip)
-   Tutorial: **Different neuron types (as library)** [link](https://github.com/4DNucleome/PartSeg/blob/master/tutorials/tutorial_neuron_types/Neuron_types_example.ipynb)

## Installing

-   From binaries:
    -   [Windows](https://4dnucleome.cent.uw.edu.pl/PartSeg/Downloads/PartSeg-lastest-windows.zip) (build on Windows 10)
    -   [Linux](https://4dnucleome.cent.uw.edu.pl/PartSeg/Downloads/PartSeg-lastest-linux.zip) (build on Ubuntu 18.04)
    -   [MacOS](https://4dnucleome.cent.uw.edu.pl/PartSeg/Downloads/PartSeg-lastest-macos.zip) (build on MacOS Mojave)

-   With pip:
    -   From pypi: `pip install PartSeg[pyqt]`
    -   From repository: `pip install git+https://github.com/4DNucleome/PartSeg.git`

## Running

If you downloaded binaries, run the `PartSeg` (or `PartSeg.exe` for Windows) file inside the `PartSeg` folder

If you installed from repository or from pip, you can run it with `PartSeg` command or `python -m PartSeg`.
First option does not work on Windows.

PartSeg export few commandline options:

-   `--no_report` - disable error reporting
-   `--no_dialog` - disable error reporting and error dialog. Use only when running from terminal.
-   `segmentation_analysis` - skip launcher and start analysis gui
-   `segmentation` - skip launcher and start segmentation gui

## napari plugin

PartSeg provides napari plugins for io to allow reading projects format in napari viewer.

## Save Format

Saved projects are tar files compressed with gzip or bz2.

Metadata is saved in data.json file (in json format).
Images/masks are saved as *.npy (numpy array format).

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
