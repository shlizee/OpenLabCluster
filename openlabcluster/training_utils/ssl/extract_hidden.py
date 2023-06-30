"""
OpenLabCluster: Active Learning Based Clustering and Classification of Animal Behaviors in Videos Based on Automatically Extracted Kinematic Body Keypoints
Copyright (c) 2022-2023 University of Washington. Developed in UW NeuroAI Lab by Jingyuan Li.
"""
import torch
import numpy as np
import torch.nn as nn


def extract_hidden_ordered(model, data_train, feature_size, alpha, device):
    """Extracts hidden states of the model
    Inputs:
        model: an encoder-decoder model
        data_train: the data loader containing keypoint sequences
        feature_size: the dimension of hidden states
        alpha: 0 indicates the model doesn't have a classification layer, larger than 0 otherwise
        device: str, either 'cuda' or 'cpu'
    Outputs:
        hidden_train_tmp: extracted hidden states of sequences in the data_train
        label_list_train: labels for the input samples
        label_train_semi: indicates labeling status for input sequences, 0 for unlabeled, otherwise labeled
        pre_label_list (return if alpha > 0): a list of predicted labels by the classifier
        mi (return if alpha > 0): marginal index computed from classifier outputs for each sequence in data_train
    """
    feature_size = feature_size * 2
    train_length = len(data_train.dataset)
    hidden_train_tmp = torch.empty((train_length, feature_size)).to(device)
    label_train_semi = np.zeros(train_length, dtype=int)
    label_list_train = np.zeros(train_length, dtype=int)
    if alpha != 0: # Having the classification head.
        softmax = nn.Softmax(dim=1)
        mi = torch.empty(train_length).to(device)
        pre_label_list = torch.zeros(train_length).to(device)

    for ith, (ith_data, seq_len, label, semi, index) in enumerate(data_train):
        input_tensor = ith_data.to(device)
        if alpha == 0:
            en_hi, de_out = model(input_tensor, seq_len)
            cla_pre = None
        else:
            en_hi, de_out, cla_pre = model(input_tensor, seq_len)
            cla_prob = softmax(cla_pre).detach()
            cla = torch.argmax(cla_prob, dim=-1) + 1.0
            cla_prob = torch.sort(cla_prob, dim=-1)[0]

            mi[index] = cla_prob[:, -1] - cla_prob[:, -2]
            pre_label_list[index] = cla * 1.0

        label_list_train[index] = np.asarray(label)
        label_train_semi[index] = semi
        hidden_train_tmp[index, :] = en_hi[0, :, :].detach()
    if device == 'cuda':
        hidden_train_tmp = hidden_train_tmp.cpu().numpy()
        if alpha != 0:
            pre_label_list = pre_label_list.cpu().numpy()
            mi = mi.cpu().numpy()
    label_list_train = label_list_train.tolist()

    if alpha == 0:
        return hidden_train_tmp, label_list_train, label_train_semi
    else:
        return hidden_train_tmp, label_list_train, label_train_semi, pre_label_list, mi
