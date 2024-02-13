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

As below described, algorithms are the result of porting PartSeg utilities to napari
then detailed description could be found in PartSeg documentation/

## ROI Extraction (Segmentation, pixel labeling)

The PartSeg is focused on the reproducible ROI Extraction process and offers two groups of algorithms:

- __ROI Mask Extraction__ set of algorithms (from PartSeg ROI Analysis) to work on a whole stack and mainly used for extracting nucleus or cell from a stack.

- __ROI Analysis Extraction__ set of algorithms (from PartSeg ROI Mask) for detailed segmentation on the level of a single nucleus.
  If possible, they use an inner caching mechanism to improve performance while adjusting parameters.

Algorithms from both groups should support masking.
(perform ROI extraction only on the mask layer's area has non-zero values).

Parameters of ROI Extraction could be saved for later reuse (in the program) or exported to JSON and imported in another instance.
With an accuracy of up to channel selection, they are identical to PartSeg,
so importing should work both ways, but the channel selection step needs to be repeated.

The list of available algorithms could be extensible using the PartSeg plugin mechanism.

## Measurement widgets

PartSeg offers two measurement widgets:

### Measurement

Interface to PartSeg measurement engine.
In this widget, there are two tabs. \**Measurement settings* that allow
to define, delete, import, and export set of measurements

![Measurement Settings](https://i.imgur.com/cfuXRRD.png)

and **Measurement** for performing measures using an already defined set.

![Measurement](https://i.imgur.com/4LzvqRp.png)

The list of available measurements could be extensible using the PartSeg plugin mechanism.

### Simple Measurement

![Simple Measurement](https://i.imgur.com/Rnq6lF5.png)

Set of measurements that could be calculated per component respecting data voxel size.
In comparison to  *Measurement* list of available measures is limited to ones that do not need
*Mask* information and could be calculated per component.

This widget is equivalent to the PartSeg ROI Mask Simple Measurement window.

## Search label

Widget to find the layer with the given number By highlighting it or zooming on it. The highlight widget uses white color, so the highlight may not be visible if the label has a bright color.

https://user-images.githubusercontent.com/3826210/154669409-cdac9be8-3dbf-4a0e-a66f-af8a44aed0fb.mp4

## Mask create

Transform labels layer into another labels layer with the possibility to dilate, and filling small holes

![Mask create widget](https://i.imgur.com/FIJGLjb.png)

## Algorithm group widgets

The central concept of PartSeg is the group of algorithms that shares the same interface.
This allows to create workflows where author of the workflow does not need to know which algorithm for
a given task is the best. The user could select the best one and use it.
For example workflow author may specify only that he expect thresholding operator and user
could select any of available. (currently used only in PartSeg ROI Extraction)

Currently there are five algorithm groups and each of them has its own widget:

### Threshold

![Threshold widget](https://github.com/4DNucleome/PartSeg/assets/3826210/1d200722-8f26-4124-8053-52111e44172b)

### Noise filtering

![Noise filtering widget](https://github.com/4DNucleome/PartSeg/assets/3826210/f8f51bd1-c993-44c6-9fa7-90beab896eaa)

### Double threshold

where area of interest is defined by positive value and core objects (start for sprawl) by value 2:

![Double threshold widget](https://github.com/4DNucleome/PartSeg/assets/3826210/628d7b1d-40d8-4947-8bbe-d10651ddc9ce)

### Border smooth for smoothing Labels border

![Border smooth widget](https://github.com/4DNucleome/PartSeg/assets/3826210/bfab6f9f-e3ee-4df3-a2b7-4689ba4b4d3f)

### Watershed for watershed segmentation

![Watershed widget](https://github.com/4DNucleome/PartSeg/assets/3826210/1f42254b-9d58-499d-9306-9bac228ab4d2)

Part of the flow methods require information if the central object is brighter or darker than the area to split.
Examples of such methods are MSO and Path.

## Helpful widgets

There are also two widgets to simplify work with labels:

### Connected component

![image](https://github.com/4DNucleome/PartSeg/assets/3826210/6f7c06fb-5903-4b2c-aca9-60a0c2a5cc3b)

### Split core objects

It is for extract core objects returned by double threshold widget

![image](https://github.com/4DNucleome/PartSeg/assets/3826210/1a081773-1c5f-4e24-9efe-5166d4c7ac2b)

## Label selector

This is widget that allows to create custom label mapping. It could be useful to prepare
publications figures. If image hase more labels that selected label mapping contains it
cycle colors.
Created mapping will be stored between sessions

This widget will not work with really big labels numbers.

Create labels mapping:
![Label mapping creation](https://user-images.githubusercontent.com/3826210/233070662-22a2b016-1397-4a21-bac2-2588c096a702.png)

Apply label mapping:
![Label mapping apply](https://user-images.githubusercontent.com/3826210/233070664-372cf038-6658-4b86-94f6-ade3cb3df9a3.png)

More details about usage could be found [here](https://partseg.readthedocs.io/en/latest/interface-overview/interface-overview.html#create-labels)

## Copy labels

![Copy labels widget](https://github.com/4DNucleome/PartSeg/assets/3826210/3d159b28-a88f-4831-82b0-ad43c08d3405)

Copy selected Labels from current layer along z-axis.

## Image Colormap

This widget allows to create custom colormap for image.

Create colormaps will be stored between sessions.

Create new colormap:
![Create colormap](https://user-images.githubusercontent.com/3826210/233070666-79558119-7d91-4ccf-8e8f-0e55119ace98.png)

Apply colormap:
![Apply colormap](https://user-images.githubusercontent.com/3826210/233070668-b7633574-e12b-4037-acc8-eee1a70eead6.png)

More details about usage could be found [here](https://partseg.readthedocs.io/en/latest/interface-overview/interface-overview.html#color-map-creator)

## Reader plugins

In this plugin, there are also all PartSeg readers and writers.
The most important readers are this, which allows loading PartSeg projects to napari.
The one which could impact a user workflow is tiff reader.
In comparison to the napari default one, there are two essential differences.
Napari's built-in plugin loads data as they are in a file.
PartSeg plugin read file metadata and return data in TZYX order.
PartSeg reader returns each channel as a separate layer.
PartSeg reader also tries to parse voxel size metadata and set scale parameters to nanometers' size.
