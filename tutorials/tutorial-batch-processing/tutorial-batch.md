# Batch processing


## Create workflow for batch processing

1a. Select how mask file for segmentation will be recognized. 
In this example in section USE MASK FROM, select Suffix- use suffix: “_mask”
All masks created by Mask Segmentation component of Partseg contain _mask suffix
Confirm by Add mask button. File mask is now added to the workflow

1b. Select SEGMENTATION profile “chromosome1” created in chromosome 1 segmentation tutorial
All settings used in the selected profile are visible in the INFORMATION PANEL.
Confirm by Add Profile/Pipeline button. Segmentation profile is now added to the workflow.

1c. Select SET OF MEASUREMENTS created in in chromosome 1 segmentation tutorial, called “chromosome1_measure”.
All measurements within selected set are visible in the INFORMATION PANEL.
Select channel (x) and units (um) that will be used in quantification.
Confirm by Add Set of measurements mask button. Segmentation profile is now added to the workflow
1d. SAVE created Workflow under chosen name (e.g. Chromosome1_batch) by Save button
New workflow is now added to a list of workflows. 
Note: to change any element of the existing workflow check box UPDATE ELEMENT in workflow edition mode. Now select each of the elements that need replacement and upadate using REPLACE buttons in each section. 
    2. Select input files
2a. Using “Select file” option manually select all components saved by Mask Segmentation
  
Using “Select directory” option select directory where files are present and add /*[0-9].tif to path line. Files created by Mask Segmentation module of Partseg are saved as tifs, which names composed of a name of original 3D microscopic picture and component number at the end. Find All option finds all tiff files, which name ends with 0-9 number, confirm by Add button. Now a list of files appears in lower panel. This list can be expanded by adding individual files or all files from a different directory as is described above. Files can be also removed individually (remove file option) or globally (remove all option).

2b. Select created workflow in BATCH WORKFLOW (here use Chromosome1_batch)
2c. Select name and a place where resulted xls file will be stored. 
2d. Number of concurrent process can be increased depending on computer power and current usage.
2e. Start calculation by confirming with process button. A new popup window will show you summary 

Confirm with execute button 

Workflow edition
