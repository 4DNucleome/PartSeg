# Changelog 

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
