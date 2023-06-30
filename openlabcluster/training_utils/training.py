"""
OpenLabCluster: Active Learning Based Clustering and Classification of Animal Behaviors in Videos Based on Automatically Extracted Kinematic Body Keypoints
Copyright (c) 2022-2023 University of Washington. Developed in UW NeuroAI Lab by Jingyuan Li.
"""

import wx
from pathlib import Path
from threading import Thread
from wx.lib.pubsub import pub

# Third party import
from openlabcluster.third_party.deeplabcut.logging import setup_logging

# OpenLabCluster import
from openlabcluster.training_utils.ssl.SeqModel import seq2seq
from openlabcluster.training_utils.ssl.seq_train import training, clustering_knn_acc


class train_unsup_network(Thread):
    def __init__(self, config, canvas,
                 displayiters=None,
                 saveiters=None,
                 maxiters=None,
                 continue_training=False,
                 reducer_name='PCA',
                 dimension=2):
        """Initializes a thread to train the sequence-to-sequence model for behavior clustering (Unsupervised)
        Inputs:
            config: the directory of a config file
            canvas: the plot handle
            displayiters: frequency to update the plots
            saveiters: frequency to save the trained model
            maxiters: maximum number of training epochs
            continue_training: if true, uses pretrained model otherwise train from scratch
            reducer_name: method name for dimension reduction, options: "PCA", "tSNE", "UMAP"
            dimension: the number of dimensions to display
        """
        Thread.__init__(self)

        self.canvas = canvas
        self.displayiters = displayiters
        self.maxiters = maxiters
        self.saveiters = saveiters
        self.cfg = config
        self.stop_work_thread = 0
        self.continue_training = continue_training
        self.reducer_name = reducer_name
        self.dimension = dimension
        self.start()  # Starts the thread

    def run(self):
        """
        Starts to train the unsupervised clustering model
        """
        import torch.nn as nn
        import torch
        import os
        import numpy as np
        from torch.utils.data import Dataset, SubsetRandomSampler
        from openlabcluster.third_party.deeplabcut import auxiliaryfunctions
        from torch import optim
        print(self.cfg)
        os.chdir(
            str(Path(self.cfg).parents[0])
        )  # switch to folder of config_yaml (for logging)
        setup_logging()

        cfg = auxiliaryfunctions.read_config(self.cfg)  # load_config(config_yaml)
        num_class = cfg['num_class'][0]
        root_path = cfg["project_path"]
        batch_size = cfg['batch_size']
        model_name = cfg['tr_modelName']
        model_type = cfg['tr_modelType']
        import sys
        if len(cfg['train']) != 0:
            from openlabcluster.training_utils.ssl.data_loader import UnsupData, pad_collate_iter, get_data_paths
            dataset_train = UnsupData(get_data_paths(root_path, cfg['data_path'], cfg['train']))

            dataset_size_train = len(dataset_train)

            indices_train = list(range(dataset_size_train))

            random_seed = 11111

            np.random.seed(random_seed)
            np.random.shuffle(indices_train)

            print("training data length: %d" % (len(indices_train)))
            # Seperates train and validation
            train_sampler = SubsetRandomSampler(indices_train)
            train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
                                                       sampler=train_sampler, collate_fn=pad_collate_iter)

        fix_weight = cfg['fix_weight']
        fix_state = cfg['fix_state']
        teacher_force = cfg['teacher_force']
        phase = 'PC'
        if fix_weight:
            network = 'FW' + phase

        if fix_state:
            network = 'FS' + phase

        if not fix_state and not fix_weight:
            network = 'O' + phase

        # Hyperparameters
        # global variables
        feature_length = cfg['feature_length']
        hidden_size = cfg['hidden_size']
        batch_size = cfg['batch_size']
        en_num_layers = cfg['en_num_layers']
        de_num_layers = cfg['de_num_layers']
        learning_rate = cfg['learning_rate']
        epoch = self.maxiters

        device = cfg['device']
        percentage = 1
        few_knn = False

        # Initializes the model
        model = seq2seq(feature_length, hidden_size, feature_length, batch_size,
                        en_num_layers, de_num_layers, fix_state, fix_weight, teacher_force, device).to(device)

        with torch.no_grad():
            for child in list(model.children()):
                print(child)
                for param in list(child.parameters()):
                    if param.dim() == 2:
                        nn.init.uniform_(param, a=-0.05, b=0.05)

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
        if model_type == 'seq2seq' and self.continue_training:
            if os.path.exists(model_name):
                from openlabcluster.training_utils.ssl.utilities import load_model
                model, optimizer = load_model(model_name, model, optimizer, device)

        loss_type = 'L1'

        if loss_type == 'MSE':
            criterion_seq = nn.MSELoss(reduction='none')

        if loss_type == 'L1':
            criterion_seq = nn.L1Loss(reduction='none')

        criterion_cla = nn.CrossEntropyLoss(reduction='sum')

        alpha = 0

        file_output = open(os.path.join(root_path, cfg['output_path'], '%sA%.2f_P%d_en%d_hid%d.txt' % (
            network, alpha, percentage * 100, en_num_layers, hidden_size)), 'w')
        model_prefix = os.path.join(root_path, cfg['model_path'], '%sA%.2f_P%d_en%d_hid%d' % (
            network, alpha, percentage * 100, en_num_layers, hidden_size))
        model_path = Path(model_prefix).parent
        pre = Path(model_prefix).name
        lambda1 = lambda ith_epoch: 0.95 ** (ith_epoch // 5)
        model_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
        past_loss = sys.float_info.max
        self.train_loader = train_loader
        self.hidden_size = hidden_size
        self.num_class = num_class
        self.alpha = alpha
        self.few_knn = few_knn
        self.device = device

        for ith_epoch in range(epoch):
            past_loss, self.path_model = training(ith_epoch, epoch, train_loader,
                                                  model, optimizer, criterion_seq, criterion_cla, alpha, past_loss,
                                                  model_path, pre,
                                                  model_prefix,
                                                  device)
            if ith_epoch % self.displayiters == 0:
                self.ith_epoch = ith_epoch
                self.model = model
                wx.CallAfter(pub.sendMessage, "plot")
            model_scheduler.step()
            if ith_epoch % 50 == 0:
                filename = file_output.name
                file_output.close()
                file_output = open(filename, 'a')
            if self.path_model:
                auxiliaryfunctions.edit_config(self.cfg, {'tr_modelName': self.path_model, 'tr_modelType': 'seq2seq'})
            else:
                auxiliaryfunctions.edit_config(self.cfg, {'tr_modelType': 'seq2seq'})
            if self.stop_work_thread == 1:
                print('stopped')
                break
        wx.CallAfter(pub.sendMessage, "finish")
        return

    def plot(self):
        """
        Plots sequences in the reduced dimension space
        """
        clustering_knn_acc(
            self.model,
            self.train_loader,
            self.hidden_size,
            self.alpha,
            self.canvas, self.ith_epoch, self.device, self.reducer_name, self.dimension)

    def stop(self):
        """
        Stops training
        """
        self.stop_work_thread = 1
