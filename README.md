# Herbarium_Project

### Scraping Scripts
#### *Usage*
Use these scritps to scrape plant data to use for training and validation

#### *Results*
The results are the included files: `species.json` (species images) and `species_names.txt` (this is a python set binary that can be opened using pickle)

#### *Testing*
1. Environment Setup (for both `scrape_images.py` and  `scrape_plant_names.py`): 
      * Make sure you have sellenium Installed. you may need an **OS-specific chrome webdriver** that is in the same directory as these files. The files can be found in the 'herb_dat' folder
      * Beware that these scripts could take DAYS to run, so just use them as a way to test that they actually run. Note that for the `species_names.txt`,
        we saved different sections of the set of plant names to avoid any memory issues, and created `species_names.txt` separately on a local machine
        using the union of these sets
      * Run `scrape_images.py` or  `scrape_plant_names.py`

### Label Segmentation
#### *Usage*
Use CRAFT's heatmap to detect text on the Herbarium specimen. Next, expand the resulting bounding box, merge overlapping ones, then crop out the label which contains specimen type, name, location, etc. by identifying the largest bounding box on the images. 

#### *Results* 
[Sample Labels](https://drive.google.com/file/d/1YqlqDSl7fUcgLrR02slxLUG_mLXbevEF/view?usp=sharing) 

[Grouth Truth](https://github.com/mzheng27/Herbarium_Project/blob/main/words.txt) 

#### *Testing* 
1. Environment Setup: 
      * Clone the Label_Segmentation Branch (has an environment.yml file) 
      * `conda env create --name envname --file=environment.yml`
      * `activate conda env envname`
2. Update path in `test.py`: 
      * Change `line 49` to where you placed the original specimen scans  
      * Change `line 50` to an arbitrary local directory
      * Change `line 226` to where you want to store the cropped labels
3. Run > `python test.py`
      * Results will be saved to the directoy specified in line 226. 
