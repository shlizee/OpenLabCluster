"""
OpenLabCluster: Active Learning Based Clustering and Classification of Animal Behaviors in Videos Based on Automatically Extracted Kinematic Body Keypoints
Developed in UW NeuroAI Lab by Moishe Keselman.
"""

import os

guistate = os.environ.get("OLClight", default="False")

# If module is executed directly (i.e. `python -m openlabcluster.__init__`) launch straight into the GUI
if guistate == "False":
    print("Starting GUI...")
    import openlabcluster
    openlabcluster.launch_olc()
else:
    print("You are in OLClight mode, GUI cannot be started.")
