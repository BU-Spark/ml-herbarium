# Herbarium_Project


### Label Segmentation
#### *Usage*
Use CRAFT's heatmap to detect text on the Herbarium specimen. Next, expand the resulting bounding box, merge overlapping ones, then crop out the label which contains specimen type, name, location, etc. by identifying the largest bounding box on the images. 

#### *Results* 
[Sample Labels](https://drive.google.com/file/d/1YqlqDSl7fUcgLrR02slxLUG_mLXbevEF/view?usp=sharing) 

[Grouth Truth](https://github.com/mzheng27/Herbarium_Project/blob/main/words.txt) 

#### *Testing* 
1. Environment Setup: 
      * Clone the Label_Segmentation Branch (has an environment.yml file) 
      * conda env create --name envname --file=environment.yml
      * activate conda env envname
2. Update path in test.py: 
      * Change line 49 to where you placed the original specimen scans  
      * Change line 50 to an arbitrary local directory
      * Change line 226 to where you want to store the cropped labels
3. Run > python test.py
      * Results will be saved to the directoy specified in line 226. 
