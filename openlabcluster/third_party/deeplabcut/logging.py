"""
This file is adapted from original file available at
https://github.com/DeepLabCut/DeepLabCut/blob/2472d40a4b1a96130984d9f1bff070f15f5a92a9/deeplabcut/pose_estimation_tensorflow/util/logging.py
"""

"""
Adapted from DeeperCut by Eldar Insafutdinov
https://github.com/eldar/pose-tensorflow
"""
import logging
import os


def setup_logging():
    """
    Sets up the logging format.
    """
    FORMAT = "%(asctime)-15s %(message)s"
    logging.basicConfig(
        filename=os.path.join("log.txt"),
        filemode="w",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        format=FORMAT,
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger("").addHandler(console)
