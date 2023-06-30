"""
OpenLabCluster: Active Learning Based Clustering and Classification of Animal Behaviors in Videos Based on Automatically Extracted Kinematic Body Keypoints
Copyright (c) 2022-2023 University of Washington. Developed in UW NeuroAI Lab by Jingyuan Li.
"""
import numpy as np

def SampleFromCluster(train_id_list, dis_list, mi_list, sample_method, num_sample):
    """ Selects a set of keypoint sequences for annotation using active learning
    Inputs:
      train_id_list: a list of lists. Sequences with the same cluster assignments are grouped into the same sub-list
      dis_list:  a list of lists including distance of keypoint sequences to their cluster center (grouped in the same way as the train_id_list)
      mi_list: a list of lists including marginal index values of keypoint sequences (grouped in the same way as the train_id_list)
      sample method: the name of active learning method
                    Options: "Marginal Index (MI)", "Core Set (CS)", "Cluster Center (Top)",
                            "Cluster Random (Rand)", "Uniform (Uni)"
      num_sample: the number of samples to select.

    Outputs:
      toLabel: the ids of selected samples for labeling.
    """
    num_class = len(train_id_list)
    toLabel = []
    # Different active learning methods.
    if sample_method == 'random':
      all_id = []
      for i in train_id_list:
        all_id += i.tolist()
      np.random.shuffle(all_id)
      toLabel = all_id[:num_sample]

    if sample_method == 'ktop':
      print('ktop selection')
      for i in range(len(train_id_list)):
        index = train_id_list[i]
        distance = np.argsort(dis_list[i])
        if len(distance) > 0:
          toLabel = toLabel + [index[distance[0]]]

    if sample_method == 'krandom':
      print('krandom selection')
      for i in range(len(train_id_list)):
        index = train_id_list[i]
        np.random.shuffle(index)
        if len(index) > 0:
          toLabel = toLabel + [index[0]]

    if sample_method == 'kmi':
      print('kmi selection')
      for i in range(len(train_id_list)):
        index = train_id_list[i]
        mi = mi_list[i]
        sort_mi = np.argsort(mi)
        if len(index) > 0:
          toLabel = toLabel + [index[sort_mi[0]]]

    return toLabel
