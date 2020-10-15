# Batch processing


## Create a workflow for a batch processing

Zdjęcie batch workflow z wprowadzonymi ustawieniami opisanymi w przykladzie z ponumerownymi wedlug opisu krokami


1. Select how the mask file for segmentation will be recognized. 
In this example in section USE MASK FROM, select Suffix- use suffix: “_mask” (1a)
All masks created by the Mask Segmentation component of Partseg contain _mask suffix
Confirm with the "Add mask" button (1b). File mask is now added to the workflow (1c)

2. Select SEGMENTATION profile “chromosome1” created in chromosome 1 segmentation tutorial
All settings used in the selected profile are visible in the INFORMATION PANEL (2a).
Confirm with the "Add Profile/Pipeline" button (2b). The segmentation profile is now added to the workflow (2c).

3. Select SET OF MEASUREMENTS created in chromosome 1 segmentation tutorial, called “chromosome1_measure”.
All measurements within the selected set are visible in the INFORMATION PANEL (3a).
Select channel (x) and units (um) that will be used in quantification.
Confirm with the "Add Set of measurements" button (3b). The segmentation profile is now added to the workflow (3c).
4. SAVE created Workflow under the chosen name (e.g. Chromosome1_batch) with the "Save" button (4a)
A new workflow is now added to a list of workflows (4b). 
Note: to change any element of the existing workflow check box "UPDATE ELEMENT" (4c) in the workflow edition mode. 
Now select each of the elements that need replacement and update using the "REPLACE" buttons (4d) in each section. 


## Execute created workflow
Zdjęcie panelu z zanaczonymi miejscami z tekstu

1. Using the “Select file” option manually select list of components (without mask files) saved by a Mask Segmentation (1a)
   Using “Select directory” (1b) option select directory where files are present and add `/*[0-9].tif` to the path line (1c). 
   Files created by the Mask Segmentation module of Partseg are saved as tifs, which names composed of the name of the original 3D microscopic picture and component number at the end. 
   "Find All" (1d) option finds all tiff files, which name ends with 0-9 number, confirm by the "Add" button (1e). Now a list of files appears in the lower panel (1f). 
   This list can be expanded by adding individual files or all files from a different directory as is described above. Files can be also removed individually ("Remove file" option (1g)) or globally ("Remove all" option(1h)).

2. Select created workflow in BATCH WORKFLOW (here use Chromosome1_batch (2))
3. Select the name and a place where the resulted xls file will be stored (3). 
4. Number of the concurrent process can be increased depending on computer power and current usage (4).
5. Start calculation by confirming with the "Process" button (5). 
Zdjęcie z panelu 
6. A new popup window will show you a summary. Confirm with the "Execute" button (6)
7. The progress of processing can be followed in the Single batch progress status bar (7a and 7b)


## Data curation 

 animowany gif?
1. In case some errors are detected during batch processing, a list of files and descriptions of errors appears in the bottom left panel (8). 
Information on errors is automatically sent out and helps us to improve the next version of Partseg. 
2. For files with complete segmentation failure a separate spreadsheet entitled "Erorrs" with a list of files will be created to inform the user. 
3. Files can be transferred directly from the spreadsheet to the “input file” widget for batch processing. 
4. After choosing the workflow used for the batch processing, each file can be opened with a right click of a mouse and verified in "ROI Analysis". 
5. Correct segmentation parameters, create a new segmentation profile, and based on it, a new batch workflow, and again process files from the list.
6. The results for each nucleus and each component are present in the spreadsheet saved at the location selected by the user. 
7. Use data filtration in the spreadsheet to check for incorrectly segmented and measured structures. Resulting in a list of files, which contain incorrect data can be processed as in 2.
 All spreadsheets created by batch processing include information about settings and parameters used for segmentation and measurement. 
