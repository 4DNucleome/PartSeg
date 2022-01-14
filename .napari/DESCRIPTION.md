# Description

PartSeg is a specialized GUI for feature extraction from multichannel light microscopy images, but also most of its features are available as napari Widgets.
Information about PartSeg as standalone program or library are [here](https://github.com/4DNucleome/PartSeg)

## Who is This For?

This plugin is for 2D and 3D segmentation of distinct objects from images and measuring various parameters of these objects.

This plugin process everything in memory, so it may not work with images stored in dask arrays. Currently, this plugin does not support the processing of Time data.

## How-to Guide

The example data is stored [here](https://4dnucleome.cent.uw.edu.pl/PartSeg/Downloads/test_data.tbz2).

https://user-images.githubusercontent.com/3826210/149146100-973498c7-7d2e-4298-a3c8-3051912e3183.mp4

The above video presents simple segmentation and measurement of various parameters of this segmentation (ROI).

As bellow described, algorithms are the result of porting PartSeg utilities to napari
then detailed description could be found in PartSeg documentation/

### ROI Extraction (Segmentation, pixel labeling)

The PartSeg is focused on the reproducible ROI Extraction process and offers two groups of algorithms:

*   __ROI Mask Extraction__ set of algorithms (from PartSeg ROI Analysis) to work on a whole stack and mainly used for extracting nucleus or cell from a stack.
*   __ROI Analysis Extraction__ set of algorithms (from PartSeg ROI Mask) for detailed segmentation on the level of a single nucleus.
    If possible, they use an inner caching mechanism to improve performance while adjusting parameters.

Algorithms from both groups should support masking.
(perform ROI extraction only on the mask layer's area has non-zero values).

Parameters of ROI Extraction could be saved for later reuse (in the program) or exported to JSON and imported in another instance.
With an accuracy of up to channel selection, they are identical to PartSeg,
so importing should work both ways, but the channel selection step needs to be repeated.

The list of available algorithms could be extensible using the PartSeg plugin mechanism.

### Measurement widgets

PartSeg offers two measurement widgets:

#### Measurement

Interface to PartSeg measurement engine.
In this widget, there are two tabs. **Measurement settings* that allow
to define, delete, import, and export set of measurements

![Measurement Settings](https://i.imgur.com/cfuXRRD.png)

and **Measurement** for performing measures using an already defined set.

![Measurement](https://i.imgur.com/4LzvqRp.png)

The list of available measurements could be extensible using the PartSeg plugin mechanism.

#### Simple Measurement

![Simple Measurement](https://i.imgur.com/Rnq6lF5.png)

Set of measurements that could be calculated per component respecting data voxel size.
In comparison to  *Measurement* list of available measures is limited to ones that do not need
*Mask* information and could be calculated per component.

This widget is equivalent to the PartSeg ROI Mask Simple Measurement window.

### Search label

Widget to find the layer with the given number By highlighting it or zooming on it. The highlight widget uses white color, so the highlight may not be visible if the label has a bright color.

https://user-images.githubusercontent.com/3826210/149330875-84939278-abfb-459a-bde6-e27b6186aa06.mp4

### Mask create

Transform labels layer into another labels layer with the possibility to dilate, and filling small holes

![Mask create widget](https://i.imgur.com/FIJGLjb.png)

## Reader plugins

In this plugin, there are also all PartSeg readers and writers.
The most important readers are this, which allows loading PartSeg projects to napari.
The one which could impact a user workflow is tiff reader.
In comparison to the napari default one, there are two essential differences.
Napari's built-in plugin loads data as they are in a file.
PartSeg plugin read file metadata and return data in TZYX order.
PartSeg reader returns each channel as a separate layer.
PartSeg reader also tries to parse voxel size metadata and set scale parameters to nanometers' size.
