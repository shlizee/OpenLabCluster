"""
OpenLabCluster: Active Learning Based Clustering and Classification of Animal Behaviors in Videos Based on Automatically Extracted Kinematic Body Keypoints
Copyright (c) 2022-2023 University of Washington. Developed in UW NeuroAI Lab by Jingyuan Li. 
"""
import os
from pathlib import Path

import umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from threading import Thread

DIMENSION_REDUCTION_DICT = {'PCA': PCA, 'tSNE': TSNE, 'UMAP': umap.UMAP}


class transform_hidden(Thread):
    """
    Wrapper to call extract_hid function
    """
    def __init__(self, config,
                 reducer_name='PCA',
                 dimension=2,
                 ):
        """
        Initializes the class of extracting hidden states.
        Inputs:
            config: the config file directory.
            reducer_name: the name of dimension reduction method (options: from "PCA", "tSNE", "UMAP").
            dimension: the number of dimensions to keep after dimension reduction.
        """
        Thread.__init__(self)
        self.config = config
        self.reducer_name = reducer_name
        self.dimension = dimension
        self.run()

    def run(self):
        """
        Extracts hidden states
        """
        from openlabcluster.user_interface.plot_hidden import extract_hid
        tmp = extract_hid(self.config, reducer_name=self.reducer_name, dimension=self.dimension)
        self.transform = tmp.transformed

    def transformed(self):
        """
        Returns the hidden states
        """
        return self.transform
