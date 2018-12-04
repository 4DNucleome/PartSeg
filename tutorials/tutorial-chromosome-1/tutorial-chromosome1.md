# Chromosome 1 territory analysis 

## Dataset 

Dataset for this tutorial can be download from [link](http://nucleus3d.cent.uw.edu.pl/PartSeg/)

This dataset contains `wpisać` cutted nucleus with marked chromosome 1 territory.

1. Data are comes from confocal microscope. Voxel size is 77x77x210 nm   
2. Data contains four channels: 
    * channel 0  - chromosome 1 territory (without centromere). Marker do not bind uniformly to 
    chromatin so this data cannot be used to estimate chromatin density.  
    * channel 1 - chromosome 1 telomere 3'
    * channel 2 - chromosome 1 telomere 5'
    * channel 3 - chromatin density inside nucleus
3. Data are deconvoluted

## Sample data process

### Preparation

1. Open PartSeg
2. Press "Analysis" button
3. Press "Advanced" on top of the window
4. Choose "Statistic settings" tab 
5. 

### Data process
1. Load data:
    1. Push **Open** button or press **ctrl+O** (cmd+O on mac) 
    2. Choose `image with mask` file filter
    3. Choose data to load
    4. In next widow choose matching file with `_mask` suffix
    
## Apendix - segment nucleus from stack
