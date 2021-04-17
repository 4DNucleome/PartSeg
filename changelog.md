# Changelog

## 0.13.4

-   Bugfix for outdated profile/pipeline preview

## 0.13.3

-   Fix saving roi_info in multiple files and history

## 0.13.2

-   Fix showing label in select label tab

## 0.13.1

-   Add Haralick measurements
-   Add obsep file support

## 0.13.0

-   Add possibility of custom input widgets for algorithms
-   Switch to napari Colormaps instead of custom one
-   Add points visualization
-   Synchronization widget for builtin (View menu) napari viewer
-   Drop Python 3.6

## 0.12.7

-   Fixes for napari 0.4.6

## 0.12.6

-   Fix prev_mask_get
-   Fix cache mechanism on mask change
-   Update PyInstaller build

## 0.12.5

-   Fix bug in pipeline execute

## 0.12.4

-   Fix ROI Mask windows related build (signal not properly connected)

## 0.12.3

-   Fix ROI Mask

## 0.12.2

-   Fix windows bundle

## 0.12.1

-   History of last opened files
-   Add ROI annotation and ROI alternatives
-   Minor bugfix

## 0.12.0

-   Toggle multiple files widget in View menu
-   Toggle Left panel in ROI Analysis in View Menu
-   Rename Mask Segmentation to ROI Mask
-   Add documentation for interface
-   Add Batch processing tutorial
-   Add information about errors to batch processing output file
-   Load image from the batch prepare window
-   Add search option in part of list and combo boxes
-   Add drag and drop mechanism to load list of files to batch window.

## 0.11.5

-   add side view to viewer
-   fix horizontal view for Measurements result table

## 0.11.4

-   bump to napari 0.3.8 in bundle
-   fix bug with not presented segmentation loaded from project
-   add frame (1 pix) to image cat from base one based on segmentation
-   pin to Qt version to 5.14

## 0.11.3

-   prepare for napari 0.3.7
-   split napari io plugin on multiple part
-   better reporting for numpy array via sentry
-   fix setting color for mask marking

## 0.11.2

-   Speedup image set in viewer using async calls
-   Fix bug in long name of sheet with parameters

## 0.11.1

-   Add screenshot option in View menu
-   Add Voxels measurements

## 0.11.0

-   Make sprawl algorithm name shorter
-   Unify capitalisation of measurement names
-   Add simple measurements to mask segmentation
-   Use napari as viewer
-   Add possibility to preview additional output of algorithms (In View menu)
-   Update names of available Algorithm and Measurement to be more descriptive.

## 0.10.8

-   fix synchronisation between viewers in Segmentation Analysis
-   fix batch crash on error during batch run, add information about file on which calculation fails
-   add changelog preview in Help > About

## 0.10.7

-   in measurements, on empty list of components mean will return 0

## 0.10.6

-   fix border rim preview
-   fix problem with size of image preview
-   zoom with scroll and moving if rectangle zoom is not marked

## 0.10.5

-   make PartSeg PEP517 compatible.
-   fix multiple files widget on Windows (path normalisation)

## 0.10.4

-   fix slow zoom

## 0.10.3

-   deterministic order of elements in batch processing.

## 0.10.2

-   bugfixes

## 0.10.1

-   bugfixes

## 0.10.0

-   Add creating custom label coloring.
-   Change execs interpreter to python 3.7.
-   Add masking operation in Segmentation Mask.
-   Change license to BSD.
-   Allow select root type in batch processing.
-   Add median filter in preview.

## 0.9.7

-   fix bug in compare mask

## 0.9.6

-   fix bug in loading project with mask
-   upgrade PyInstaller version (bug  GHSA-7fcj-pq9j-wh2r)

## 0.9.5

-   fix bug in loading project in "Segmentation analysis"

## 0.9.4

-   read mask segmentation projects
-   choose source type in batch
-   add initial support to OIF and CZI file format
-   extract utils to PartSegCore module
-   add automated tests of example notebook
-   reversed mask
-   load segmentation parameters in mask segmentation
-   allow use sprawl in segmentation tool
-   add radial split of mask for measurement
-   add all measurement results in batch, per component sheet

## 0.9.3

-   start automated build documentation
-   change color map backend and allow for user to create custom color map.
-   segmentation compare
-   update test engines
-   support of PySide2

## 0.9.2.3

-   refactor code to make easier create plugin for mask segmentation
-   create class base updater for update outdated algorithm description
-   fix save functions
-   fix different bugs

## 0.9.2.2

-   extract static data to separated package
-   update marker of fix range and add mark of gauss in channel control

## 0.9.2.1

-   add VoteSmooth and add choosing of smooth algorithm

## 0.9.2

-   add pypi base check for update

-   remove resetting image state when change state in same image

-   in stack segmentation add options to picking components from segmentation's

-   in mask segmentation add:

    -   preview of segmentation parameters per component,
    -   save segmentation parameters in save file
    -   new implementation of batch mode.

## 0.9.1

-   Add multiple files widget

-   Add Calculating distances between segmented object and mask

-   Batch processing plan fixes:

    -   Fix adding pipelines to plan
    -   Redesign mask widget

-   modify measurement backend to allow calculate multi channel measurements.

## 0.9

Begin of changelog
