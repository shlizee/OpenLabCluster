"""
OpenLabCluster: Active Learning Based Clustering and Classification of Animal Behaviors in Videos Based on Automatically Extracted Kinematic Body Keypoints
Copyright (c) 2022-2023 University of Washington. Developed in UW NeuroAI Lab by Jingyuan Li.
"""

import numpy as np
import torch
from sklearn import preprocessing
from sklearn.cluster import KMeans

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def remove_labeled_cluster(train_set, train_id, labeled, mi: list = []):
    """Removes keypoint sequences from the dataset if they have been annotated
    Inputs:
        train_set: a list of keypoint sequences
        train_id: a list, contains ids of each sequence in the train_set
        labeled: a list, contains ids of annotated keypoint sequences
        mi (optional): a list, contains marginal index values for each sequence in the training_set
    Outputs:
        train_set: a list, contains keypoint sequences which have not been annotated
        train_id: a list, contains ids of each sequence in the updated train_set
        mi (optional): a list, contains precomputed marginal index values for each sequence in the updated train_set (if mi is provided)
    """
    if len(labeled) != 0:

        del_id = []
        for i in range(len(train_id)):
            if train_id[i] in labeled:
                del_id = del_id + [i]
        train_set = np.delete(train_set, del_id, axis=0)
        train_id = np.delete(train_id, del_id, axis=0)
        if len(mi) > 0:
            mi = np.delete(mi, del_id, axis=0)
    if len(mi) > 0 or len(train_id) == 0:  # for the case all sequence is labeled
        return train_set, train_id, mi
    else:
        return train_set, train_id


def iter_kmeans_cluster(train_set,
                        train_id, ncluster=10,
                        mi=[]):
    """Groups keypoint sequences into clusters using KMeans clustering
    Inputs:
        train_set:  a list of keypoint sequences
        train_id: a list, contains ids of each sequence in the train_set
        nclusters: the number of clusters for KMeans clustering
        mi (optional): a list, contains precomputed marginal index values for each sequences in the train_set
    Outputs:
        train_id_list: a list of lists. Sequences with the same cluster assignments are grouped into the same sub-list
        dis_list:  a list of lists including distance of keypoint sequences to their cluster center (grouped in the same way as the train_id_list)
        mi_list: a list of lists including marginal index values of keypoint sequences (grouped in the same way as the train_id_list)
        cluster_label: KMeans cluster assignments
    """
    train_set = preprocessing.normalize(train_set)
    kmeans = KMeans(ncluster, init='k-means++', max_iter=500, random_state=0).fit(train_set)
    # Group labels of kmeans outputs.
    pre_train = kmeans.predict(train_set)
    distance = kmeans.transform(train_set)
    DisToCenter = []

    # Gathers the distance of sequences to assigned cluster center
    for i in range(len(pre_train)):
        DisToCenter.append(distance[i, pre_train[i]])
    DisToCenter = np.asarray(DisToCenter)

    # Gathers the train_id, distance and mi of the same cluster together
    train_id_list = []
    dis_list = []
    mi_list = []
    cluster_label = np.zeros(len(train_id))
    for i in range(ncluster):
        clas_poss = pre_train == i
        cluster_label[clas_poss] = i
        train_id_list.append(train_id[clas_poss])
        dis_list.append(DisToCenter[clas_poss])
        if len(mi) > 0:
            mi_list.append(mi[clas_poss])

    return train_id_list, dis_list, mi_list, cluster_label
