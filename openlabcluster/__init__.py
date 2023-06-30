"""
OpenLabCluster: Active Learning Based Clustering and Classification of Animal Behaviors in Videos Based on Automatically Extracted Kinematic Body Keypoints
Developed in UW NeuroAI Lab by Moishe Keselman.
"""


import os
import matplotlib as mpl
# Check if in the DEBUG mode
DEBUG = True and "DEBUG" in os.environ and os.environ["DEBUG"]


# Set the enviornment to use OpenLabCluster with GUI or in light mode
if os.environ.get("OLClight", default=False) == "True":
    print(
        "OLC loaded in light mode; you cannot use any GUI (labeling, relabeling and standalone GUI)"
    )
    # Use anti-grain geometry engine #https://matplotlib.org/faq/usage_faq.html
    mpl.use(
        "AGG"
    )
else:
    # Standard use [wxpython supported]
    mpl.use("WxAgg")
    from openlabcluster.user_interface.run_gui import launch_olc


#
from openlabcluster.third_party.deeplabcut.new import (
    create_new_project,
)
from openlabcluster.training_utils import (
    train_unsup_network
)

from openlabcluster.version import __version__, VERSION
