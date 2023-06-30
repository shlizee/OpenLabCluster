# OpenLabCluster
## Active Learning Based Clustering and Classification of Animal Behaviors in Videos Based on Automatically Extracted Kinematic Body Keypoints

## Usage
### Installation
#### Install from pip
Create a new environment with conda and install the package


For **Linux** start the environment with [spec-file.txt](https://drive.google.com/file/d/1nlOqspBrnl5kiErudW6NtHTt3nFCTPJH/view?usp=sharing)

	conda create --name OpenLabCluster --file spec-file.txt
	conda activate OpenLabCluster
	pip install openlabcluster
	
for **Mac OS** and **Windows**

	conda create --name OpenLabCluster python=3.7
	conda activate OpenLabCluster
	pip install 'openlabcluster[gui]'
	 
	
<!---##### Troubleshooting for Linux installation
If, for some reason, `wxPython` fails to install on Linux, run `sudo apt install libsdl2-dev build-essential libgtk-3-dev make gcc libgtk-3-dev libwebkitgtk-dev libwebkitgtk-3.0-dev libgstreamer-gl1.0-0 freeglut3 freeglut3-dev python-gst-1.0 python3-gst-1.0 libglib2.0-dev ubuntu-restricted-extras libgstreamer-plugins-base1.0-dev`. Then, use `pip install -U -f https://extras.wxpython.org/wxPython4/extras/linux/gtk3/<your operating system>  wxPython` to install it. To determine your exact link, go to https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ and select the right folder (copy the link from the search bar). 
	
#### Install the Required Package from Environment File	
Git clone the entire package
Create a new  enviornment.yml file
If you are using **Linux**
	
	conda env create -f environment.yml
	
if you are using **Mac-os**
	
	conda env create -f env-mac2.yml
--->
	
	
### Execution

Run the following for **Linux**
	
	python -m openlabcluster

Run the following for **Mac-OS**

	conda install python.app
	pythonw -m openlabcluster

Run the following for **Windows**

	pythonw.exe -m openlabcluster
		
### Run a Demo

#### Create the Demo Project
1. Download the *example* folder from [here](https://drive.google.com/file/d/15ei6_raZzNCbKU07sqWTxQdrOFCuSaJe/view?usp=sharing) including the extracted keypoints of Home-Cage Mouse dataset (from *Automated home-cage behavioural phenotyping of mice* Hueihan Jhuang et al., 2010) and follow the instructions in the README file in the example folder.
2. Go to *your_download-dir/example* folder run 
		
        python3 prepare_video_list.py
   which generates video_segments_names.text file.
   
3. Launch OpenLabCluster GUI (see Execution above)
4. Set Project Name: e.g., demo
5. Click *Load Preprocessed Keypoints*, choose datafile: your_download_dir/openlabcluster_example/demo_data.h5
6. Click *Load Video Segments Name List*, choose the file: your_download_dir/openlabcluster_example/video_segments_names.text
7. Uncheck *Check to use GPU*, if GPU is not available.
8. Set Feature Length = 16
9. Click *OK* to create the project

#### Start the Demo Project:
1. Go to **Manage Project** panel, 
2. Choose *Load Project*
3. Select the config file as *filedirectory/project_name/config.yaml*
4. Click **OK**, then go to **Cluster Map**


#### Cluster Map:
1. Click **Start Clustering** button to start unsupervised clustering. 
2. Click **Go To Classification** when unsupervised clustering is finished, then go to **Behavior Classification Map** panel.

#### Behavior Classification Map: 
1. The scatter plot indicating sample clusters is initialized on the bottom left, with suggested samples for annotation.
2. Label samples on the bottom right panel.
3. Click **Run Classification**, and start classification.


### Manage Project (Start a New Project or Load Earlier Project) Detailed Description
#### Start a New Project
1. Project Name - the name of the project
2. If you have only videos, use markerless pose estimators (e.g., DeepLabCut) to extract keypoints for each video. If you already have DeepLabCut-like formatted files, select them with "Load Keypoints Data."
3. List video names in the "Load Video Segments Names List" file. There should be one video for each keypoint file. Make sure the videos and keypoints files are in the same order.
4. Optional: Set a directory for the project (the default is the working directory of this project).
5. Keep the GPU box checked if you have a GPU on your computer and you would like to use it for training.
6. Enter the features length (number of body parts * number of dimensions per body part. For example, 5 keypoints in 2D would be 5*2 =10).
7. Choose "OK" to create the project.
8. If you want to edit the config (i.e., change class names, change video cropping), press "Edit Config File."

#### Loading a Project
1. Select the config.yaml file generated when you created that project.
2. Press "OK."


### Cluster Map (Unsupervised Learning)
#### Set the Training Parameters
1. Update Cluster Map Every (Epochs): The frequency to update the Cluster Map (bottom left panel in the figure), e.g., set 1 to update Cluster Map every training epoch, or 5 to update every 5 epochs.
2. Save Cluster Map Every (Epochs): The frequency to save the Cluster Map.
3. Maximum Epochs: The maximum number of epochs to be performed unsupervised training.
4. Cluster Map Dimension: Possible choices are 2d or 3d. For 2d, the Cluster Map will be shown in two dimensions; otherwise, it will be three dimensions.
5. Dimension Reduction Methods: Possible choices are PCA, tSNE, and UMAP. The GUI will use the chosen method to perform dimension reduction and show results in Cluster Map.


#### Buttons
After setting the parameters, you can perform the analysis:

1. Start Clustering: Perform an unsupervised sequence regeneration task.
2. Stop Clustering: Usually, the clustering will stop when it reaches the maximum epochs, but if you want to stop at an intermediate stage, click this button.
3. Continue Clustering: If you stopped the clustering at some stage and want to perform clustering with earlier clustering results, click this button.
4. Go to Classification: After the unsupervised clustering, we go to the next step, which includes: i) annotation suggestion, ii) sample annotating, and iii) semi-supervised action classification with labeled samples.

### Behavior Classification Map
#### Set the Training Parameters
1. Selection Method: In this part, your selection will decide which method GUI uses to select samples for annotation. There are four possible choices: Marginal Index (MI), Core Set (CS), Cluster Center (TOP), Cluster Random (Rand), and Uniform.
2. \# Samples per Selection: the number of samples you want to label in the current selection stage. You can select or deselect samples in the behavior classification map.
3. Maximum Epochs: The maximum epoch the network will be trained when performing the action recognition.
4. Cluster Map Dimension: You can choose 2d or 3d. If it is 2d, the Cluster Map will be shown in two-dimensional space; otherwise, it is three-dimensional space.
5. Dimension Reduction Method: Possible choices are PCA, tSNE, and UMAP. The GUI will use the chosen reduction method to perform dimension reduction and show results in the Cluster Map.


#### Buttons:
   
1. Run Classification: Save annotation results and train the action recognition model.
   
2. Stop Classification: Stop training.
   
3. Next Selection: Suggest a new set of samples based on with indicated active learning method. **Notice**: If you change Cluster Map Dimension or Dimension Reduction Method, click the **Next Selection** to show suggested samples.

4. Get Results: Get the Behavior Classification Map (predicted class label) from the trained model on unlabeled samples. 

#### Plots
1. Behavior Classification Plot:
   
   **Visualization mode**: The points representing each action segment is shown in black for visualization. It can be visualized in 2D or 3D with different dimension reduction methods.
 
   **Annotation mode**: In this mode, dots for each action segment in a different color are shown in the Behavior Classification plot (only in 2D).
      Red: current sample for annotating, and the video segment is shown on the right. 
      Blue: the suggested samples for annotating in this iteration (deselect them by clicking the point).
      Green: Samples have been annotated.

   Buttons:
   
     * Zoom: Zoom in or zoom out the plot.
     * Pan: Move the plot around.
   
   
2. Videos Panel:
   
   Left panel: The corresponding video of the action segment, which is shown in the Behavior Classification Map in red.
   
   Right panel: The class name and class id. According to the video, one can label the action segment.

   Buttons:

     * Previous: Load the previous video.
     * Play: Play the video.
     * Next: Go to the next video.
     
### Attribution
OpenLabCluster code was developed in University of Washington UW NeuroAI Lab by Jingyuan Li and Moishe Keselman. OpenLabCluster interface is inspired by <a href="https://github.com/DeepLabCut" target="_blank">DeepLabCut</a>, which code is used as a backbone for user interface panels, interaction with the back-end, logging, and visualization. For specific usage please see [third_party/deeplabcut](third_party/deeplabcut) folder. OpenLabCluster also uses <a href="https://github.com/google/active-learning" target="_blank">Google Active Learning Playground</a> code for implementation of the K-center selection method in Core-Set active learning option. For specific usage please see [third_party/kcenter](third_party/kcenter) folder.
 
Related projects and publications:
 
#### OpenLabCluster
```
@article{li2022openlabcluster,
  title={OpenLabCluster: Active Learning Based Clustering and Classification of Animal Behaviors in Videos Based on Automatically Extracted Kinematic Body Keypoints},
  author={Li, Jingyuan and Keselman, Moishe and Shlizerman, Eli},
  journal={bioRxiv},
  pages={2022--10},
  year={2022},
  publisher={Cold Spring Harbor Laboratory}
}
```

#### DeepLabCut
```
@article{Mathisetal2018,
    title = {DeepLabCut: markerless pose estimation of user-defined body parts with deep learning},
    author = {Alexander Mathis and Pranav Mamidanna and Kevin M. Cury and Taiga Abe  and Venkatesh N. Murthy and Mackenzie W. Mathis and Matthias Bethge},
    journal = {Nature Neuroscience},
    year = {2018},
    url = {https://www.nature.com/articles/s41593-018-0209-y}}
```
#### DeepLabCut2.x
```
 @article{NathMathisetal2019,
    title = {Using DeepLabCut for 3D markerless pose estimation across species and behaviors},
    author = {Nath*, Tanmay and Mathis*, Alexander and Chen, An Chi and Patel, Amir and Bethge, Matthias and Mathis, Mackenzie W},
    journal = {Nature Protocols},
    year = {2019},
    url = {https://doi.org/10.1038/s41596-019-0176-0}}
```
 
#### DeeperCut
```
@article{insafutdinov2016eccv,
    title = {DeeperCut: A Deeper, Stronger, and Faster Multi-Person Pose Estimation Model},
    author = {Eldar Insafutdinov and Leonid Pishchulin and Bjoern Andres and Mykhaylo Andriluka and Bernt Schiele},
    booktitle = {ECCV'16},
    url = {http://arxiv.org/abs/1605.03170}}
```

#### Google Active Learning Playground
```
@misc{activelearningplayground,
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/google/active-learning}}}
 ```