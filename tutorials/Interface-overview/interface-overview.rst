=======================================
Graphical User Interface (GUI) overview
=======================================

:Author: Grzegorz Bokota
:Version: $Revision: 1 $
:Copyright: This document has been placed in the public domain.



.. contents:: Table of Contents





ROI Analysis GUI
----------------

In this section we describe main window of "ROI Analysis".


.. image::  images/main_window.png
   :alt: Main Roi Analysis GUI

1.  Colorbar
2.  Two copy of Image view. Preview of data.
    Image view is described in `Image View`_.
3.  Algorithm parameters. Here you set parameters for segmentation.

Measurement
~~~~~~~~~~~

Profile, Pipeline, Project
~~~~~~~~~~~~~~~~~~~~~~~~~~

Batch processing
~~~~~~~~~~~~~~~~

Mask Segmentation GUI
---------------------


Common elements
---------------

Image view
~~~~~~~~~~

Algorithm settings
~~~~~~~~~~~~~~~~~~
This is widget for chose algorithm and set it parameters.

.. image:: images/algorithm_settings.png
   :alt: Algorithm settings

1. This is drop down list on which user can select algorithm.
2. In this area user set parameters of algorithms.
3. In this area there are show additional information produced by algorithm.

Mask manager
~~~~~~~~~~~~
This widget/dialog allows to set parameters of transferring
segmentation into mask.

.. image:: images/mask_manager.png
   :alt: Mask Manager

1. Select to use dilation (2d or 3d) with set
   its radius. If dilation is in 3d then z radius is calculated
   base on image spacing.
2. If fill holes in mask. Hole is background part
   not connected to border of image. If Maximum size is set to -1
   then all holes are closed.
3. **Save components instead** of producing binary mask.
   **Clip previous mask** is useful when using positive radius in Dilate mask
   and want to fit in previous defined mask.
4. Negate produced mask.
5. Show calculated dilation radius for current image.
6. Undo last masking operation.
7. Create new mask or go to previously undone one.
8. TODO
9. TODO

Multiple files widget
~~~~~~~~~~~~~~~~~~~~~
This is widget to manage work on multiple files without need
to reload it from disc.

Each element of top level list is one file.
For each saved

.. image:: images/multiple_files_widget.png
   :alt: Multiple files widget

1.  List of opened files.
2.  Save current image state to be reloadable.
3.  Remove saved state.
4.  Load multiple files to PartSeg.
5.  When click **Save State** open popup with option to set
    custom name instead of default one.
