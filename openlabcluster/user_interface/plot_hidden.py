"""
OpenLabCluster: Active Learning Based Clustering and Classification of Animal Behaviors in Videos Based on Automatically Extracted Kinematic Body Keypoints
Copyright (c) 2022-2023 University of Washington. Developed in UW NeuroAI Lab by Jingyuan Li.
"""
import argparse
from pathlib import Path
import torch.nn as nn
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap

# Third-party import
from openlabcluster.third_party.deeplabcut import auxiliaryfunctions
from openlabcluster.third_party.deeplabcut.logging import setup_logging

DIMENSION_REDUCTION_DICT = {'PCA': PCA, 'tSNE': TSNE, 'UMAP':umap.UMAP}

class extract_hid():
    """
    Extracts the encoder hidden states and performs dimension reduction
    """
    def __init__(self, config_yaml, dimension, reducer_name):
        """Initializes the class of hidden states extraction
        Inputs:
            config_yaml: the config file directory
            dimension: the number of dimensions to keep after the dimension reduction
            reducer_name: the method to be used for dimension reduction (options: from "PCA", "tSNE", "UMAP")
        """
        from openlabcluster.training_utils.ssl.utilities import load_model
        import torch
        import os
        from torch.utils.data import Dataset
        from torch import optim

        # Switches to folder of config_yaml (for logging)
        os.chdir(
            str(Path(config_yaml).parents[0])
        )
        setup_logging()

        # Gets the default parameters.
        cfg = auxiliaryfunctions.read_config(config_yaml)
        root_path = cfg["project_path"]
        batch_size = cfg['batch_size']
        model_type = cfg['tr_modelType']
        model_name = cfg['tr_modelName']
        label_path = os.path.join(cfg['project_path'],cfg['label_path'], 'label.npy')
        if not os.path.exists(label_path):
            label_path = None
        fix_weight = cfg['fix_weight']
        fix_state = cfg['fix_state']
        teacher_force = cfg['teacher_force']
        phase = 'PC'
        feature_length = cfg['feature_length']
        hidden_size = cfg['hidden_size']
        batch_size = cfg['batch_size']
        en_num_layers = cfg['en_num_layers']
        de_num_layers = cfg['de_num_layers']
        cla_num_layers = cfg['cla_num_layers']
        learning_rate = cfg['learning_rate']
        device = cfg['device']
        # Class labels start from 1, with 0 indicates the non-labeled samples.
        cla_dim = cfg['cla_dim']
        self.num_class = cla_dim[-1]

        # Gets dataset and dataloader.
        if len(cfg['train'])!=0:
            from openlabcluster.training_utils.ssl.data_loader import SupDataset, pad_collate_iter
            dataset_train = SupDataset(root_path, cfg['data_path'], cfg['train'], label_path)

        dataset_size_train = len(dataset_train)
        self.dataset_size_train = dataset_size_train

        indices_train = list(range(dataset_size_train))
        print("training data length: %d" % (len(indices_train)))
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
                                                   shuffle=False, collate_fn=pad_collate_iter)

        # Initializes the model and loads from the most current checkpoint.
        from openlabcluster.training_utils.ssl.SeqModel import seq2seq, SemiSeq2Seq
        if model_type == 'seq2seq' or not model_type:
            model = seq2seq(feature_length, hidden_size, feature_length, batch_size,
                        en_num_layers, de_num_layers, fix_state, fix_weight, teacher_force, device).to(device)
            alpha= 0
        elif model_type == 'semi_seq2seq':
            model = SemiSeq2Seq(feature_length, hidden_size, feature_length, batch_size,
                                cla_dim, en_num_layers, de_num_layers, cla_num_layers, fix_state, fix_weight, teacher_force, device).to(device)
            alpha = 0.5
        with torch.no_grad():
            for child in list(model.children()):
                print(child)
                for param in list(child.parameters()):
                    if param.dim() == 2:
                        nn.init.uniform_(param, a=-0.05, b=0.05)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
        if model_name:
            if os.path.exists(model_name):
                model, _ = load_model(model_name, model, optimizer, device)

        # Extracts the hidden states using the pretrained model.
        from openlabcluster.training_utils.ssl.extract_hidden import extract_hidden_ordered
        hidd = extract_hidden_ordered(model, train_loader, hidden_size, alpha, device)
        self.hidarray = hidd[0]

        # Performs dimension reduction on the extracted hidden states.
        dim_reducer = DIMENSION_REDUCTION_DICT[reducer_name]
        if dimension == '2d':
            reducer = dim_reducer(n_components=2)
        else:
            reducer = dim_reducer(n_components=3)
        self.transformed = reducer.fit_transform(self.hidarray)
        self.semilabel = hidd[2]
        self.gt_label = hidd[1]
        if alpha!=0:
            self.pred_label = hidd[-2]
            self.mi = hidd[-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to yaml configuration file.")
    cli_args = parser.parse_args()

