# PartSeg - gui for segmentation algorithms

This application is designed to help biologist with segmentation 
based on threshold and connected components 

## Run from code
* PartSeg - `python src/launcher_.py`


## Non standard python libraries dependencies
* tifffile
* SimpleITK
* Matplotlib
* Numpy
* PyQt5
* appdirs
* h5py

## Project Web Page
http://nucleus3d.cent.uw.edu.pl/PartSeg  (binaries here)

## Save Format
Saved project are tar files compressed with gzip or bz2 

Metadata are saved in data.json file (in json format)
images/mask are saved as *.npy (numpy array format)


## Interface
![launcher](images/launcher.png)
![interface](images/analysis_gui.png)
![interface](images/analysis_gui2.png)
![statistics](images/statisitcs.png)
![statistics](images/statisitcs.png)
![mask interface](images/mask_gui.png)



## Laboratory
Laboratory of functional and structural genomics
http://4dnucleome.cent.uw.edu.pl/


