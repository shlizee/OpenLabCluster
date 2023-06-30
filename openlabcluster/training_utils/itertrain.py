"""
OpenLabCluster: Active Learning Based Clustering and Classification of Animal Behaviors in Videos Based on Automatically Extracted Kinematic Body Keypoints
Copyright (c) 2022-2023 University of Washington. Developed in UW NeuroAI Lab by Jingyuan Li.
"""

import os
import importlib
import logging
import wx
from wx.lib.pubsub import pub
from pathlib import Path
from threading import Thread

# Third party import
from openlabcluster.third_party.deeplabcut.logging import setup_logging
from openlabcluster.third_party.deeplabcut import auxiliaryfunctions

# OpenLabCluster import
from openlabcluster.training_utils.ssl.seq_train import clustering_knn_acc


class train_iter_network(Thread):
    def __init__(self, config, canvas,
                 epochs: int = None,
                 model_name=None, model_type=None, acc_text=None):
        """Initializes the thread for training behavior classification
        Inputs:
            config: the directory of the config file
            canvas: the plot handle
            epochs: maximum number of training epochs
            model_name: the pretrained model directory
            model_type: str, 'seq2seq' for unsupervised learning 'semi_seq2seq' for semi-supervised learning
            acc_test: the training accuracy
        """
        Thread.__init__(self)

        importlib.reload(logging)
        logging.shutdown()
        self.cfg = config
        self.cfg_data = auxiliaryfunctions.read_config(self.cfg)
        if model_name:
            self.model_name = model_name
        else:
            self.model_name = self.cfg_data['tr_modelName']
            if not self.model_name:
                wx.MessageBox('Error Do Cluster Data First', 'Error')

        if model_type:
            self.model_type = model_type
        else:
            self.model_type = self.cfg_data['tr_modelType']

        self.canvas = canvas
        self.stop_work_thread = 0
        self.acc_text = acc_text
        self.epochs = epochs
        # Read file path for pose_config file. >> pass it on
        self.start()  # start the thread

    def run(self):
        """
        Trains the semi-supervised behavior classification model (Initiated by Start)
        """

        import numpy as np
        import sys

        import torch
        import torch.nn as nn
        from torch.utils.data import Dataset, SubsetRandomSampler
        from torch import optim

        from openlabcluster.training_utils.ssl.SeqModel import SemiSeq2Seq
        from openlabcluster.training_utils.ssl.SeqModel import seq2seq
        from openlabcluster.training_utils.ssl.seq_train import training
        from openlabcluster.training_utils.ssl.utilities import load_model

        os.chdir(
            str(Path(self.cfg).parents[0])
        )  # Switches to folder of config_yaml (for logging)
        setup_logging()

        num_class = self.cfg_data['num_class'][0]
        root_path = self.cfg_data["project_path"]
        batch_size = self.cfg_data['batch_size']

        if len(self.cfg_data['train']) != 0:
            from openlabcluster.training_utils.ssl.data_loader import SupDataset, pad_collate_iter
            label_path = os.path.join(self.cfg_data['label_path'], 'label.npy')
            if not os.path.exists(label_path):
                label_path = None
            dataset_train = SupDataset(root_path, self.cfg_data['data_path'], self.cfg_data['train'], label_path)

            dataset_size_train = len(dataset_train)
            indices_train = list(range(dataset_size_train))

            random_seed = 11111

            np.random.seed(random_seed)
            np.random.shuffle(indices_train)

            print("training data length: %d" % (len(indices_train)))
            # Prepares training dataloader
            train_sampler = SubsetRandomSampler(indices_train)
            train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
                                                       sampler=train_sampler, collate_fn=pad_collate_iter)

        fix_weight = self.cfg_data['fix_weight']
        fix_state = self.cfg_data['fix_state']
        teacher_force = self.cfg_data['teacher_force']
        phase = 'PC'
        if fix_weight:
            network = 'FW' + phase

        if fix_state:
            network = 'FS' + phase

        if not fix_state and not fix_weight:
            network = 'O' + phase

        # Hyperparameters
        feature_length = self.cfg_data['feature_length']
        hidden_size = self.cfg_data['hidden_size']
        batch_size = self.cfg_data['batch_size']
        en_num_layers = self.cfg_data['en_num_layers']
        de_num_layers = self.cfg_data['de_num_layers']
        cla_num_layers = self.cfg_data['cla_num_layers']
        learning_rate = self.cfg_data['learning_rate']
        epoch = self.epochs if self.epochs is not None else self.cfg_data["su_epoch"]

        device = self.cfg_data['device']
        percentage = 1
        few_knn = False
        # Global variable
        cla_dim = self.cfg_data['cla_dim']  # 0 non labeled class

        print_every = 1

        model = SemiSeq2Seq(feature_length, hidden_size, feature_length, batch_size,
                            cla_dim, en_num_layers, de_num_layers, cla_num_layers,
                            fix_state, fix_weight, teacher_force, device=device)
        print('network fix state=', fix_state)

        with torch.no_grad():
            for child in list(model.children()):
                print(child)
                for param in list(child.parameters()):
                    if param.dim() == 2:
                        # nn.init.xavier_uniform_(param)
                        nn.init.uniform_(param, a=-0.05, b=0.05)

        if self.model_type == 'seq2seq':
            model_tmp = seq2seq(feature_length, hidden_size, feature_length, batch_size,
                                en_num_layers, de_num_layers,
                                fix_state, fix_weight, teacher_force, device=device)
            optimizer_tmp = optim.Adam(filter(lambda p: p.requires_grad, model_tmp.parameters()), lr=learning_rate)
            model_tmp, _ = load_model(self.model_name, model_tmp, optimizer_tmp, device)
            model.seq = model_tmp
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

        elif self.model_type == 'semi_seq2seq':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
            model, optimizer = load_model(self.model_name, model, optimizer, device)

        loss_type = 'L1'

        if loss_type == 'MSE':
            criterion_seq = nn.MSELoss(reduction='none')

        if loss_type == 'L1':
            criterion_seq = nn.L1Loss(reduction='none')

        criterion_cla = nn.CrossEntropyLoss(reduction='sum')

        alpha = 0.5

        file_output = open(os.path.join(root_path, self.cfg_data['output_path'], '%sA%.2f_P%d_en%d_hid%d.txt' % (
            network, alpha, percentage * 100, en_num_layers, hidden_size)), 'w')
        model_prefix = os.path.join(root_path, self.cfg_data['model_path'], '%sA%.2f_P%d_en%d_hid%d' % (
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
            past_loss, model_name, self.acc = training(ith_epoch, epoch, train_loader,
                                                       model, optimizer, criterion_seq, criterion_cla, alpha, past_loss,
                                                       model_path, pre,
                                                       model_prefix,
                                                       device)
            if ith_epoch % print_every == 0:
                self.ith_epoch = ith_epoch
                self.model = model
                wx.CallAfter(pub.sendMessage, "plot_iter")
            if model_name:
                auxiliaryfunctions.edit_config(self.cfg, {'tr_modelType': 'semi_seq2seq', 'tr_modelName': model_name})
            else:
                auxiliaryfunctions.edit_config(self.cfg, {'tr_modelType': 'semi_seq2seq'})
            model_scheduler.step()
            wx.CallAfter(pub.sendMessage, "update", step=1)
            if ith_epoch % 50 == 0:
                filename = file_output.name
                file_output.close()
                file_output = open(filename, 'a')
            if self.stop_work_thread == 1:
                break
        wx.CallAfter(pub.sendMessage, "finish_iter")
        return

    def plot(self):
        """
        Plots learned hidden states in an embedded space
        """
        clustering_knn_acc(
            self.model,
            self.train_loader,
            self.hidden_size,
            self.alpha,
            self.canvas, self.ith_epoch, self.device, reducer_name='PCA', dimension='2d')

    def stop(self):
        """
        Stops training
        """
        self.stop_work_thread = 1
