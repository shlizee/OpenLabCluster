"""
OpenLabCluster: Active Learning Based Clustering and Classification of Animal Behaviors in Videos Based on Automatically Extracted Kinematic Body Keypoints
Copyright (c) 2022-2023 University of Washington. Developed in UW NeuroAI Lab by Jingyuan Li.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, device='cuda'):
        """Initializes a bidirectional RNN

        Inputs:
            input_size: the feature size of input sequence
            hidden_size: the hidden dimension of the RNN
            num_layers: the number of layers of the RNN
        """
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.num_layers = num_layers
        self.device = device

    def forward(self, input_tensor, seq_len):
        """Performs a forward pass

        Inputs:
            input_tensor: a batch of keypoint sequences (shape: batch size, sequence length, input size)
            seq_len: a list of sequence lengths of input tensors before padding
        Returns:
            hidden: a torch tensor including the hidden states of the RNN at the last time step
        """
        encoder_hidden = torch.Tensor().to(self.device)
        for it in range(max(seq_len)):
            if it == 0:
                enout_tmp, hidden_tmp = self.gru(input_tensor[:, it:it + 1, :])
            else:
                enout_tmp, hidden_tmp = self.gru(input_tensor[:, it:it + 1, :], hidden_tmp)
            encoder_hidden = torch.cat((encoder_hidden, enout_tmp), 1)

        hidden = torch.empty((1, len(seq_len), encoder_hidden.shape[-1])).to(self.device)
        count = 0
        for ith_len in seq_len:
            hidden[0, count, :] = encoder_hidden[count, ith_len - 1, :]
            count += 1
        return hidden


class DecoderRNN(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers):
        """Initializes a unidirectional RNN

        Inputs:
            output_size: input & output feature size of the model
                        (input and output feature size are the same for regeneration task)
            hidden_size: the hidden dimension of the RNN
            num_layers: the number of layers of the RNN
        """
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(output_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input_tensor, hidden):
        """Performs a forward pass

        Inputs:
            input_tensor: a batch of sequences (shape: batch size, sequence length, output size)
            seq_len: a list of sequence lengths of input tensors before padding
        Returns:
            hidden: a torch tensor including the hidden states of the RNN at the last time step
        """
        output, hidden = self.gru(input_tensor, hidden)

        output = self.out(output)
        return output, hidden


class Classification(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers, device ='cuda'):
        """Initializes a multi-layer perceptron for classification

        Inputs:
            in_dim: the dimension of inputs to the classifier
            out_dim: the output feature dimension, the same as the numer of classes
            num_layers: the number of layers of the classifier
        """
        super(Classification, self).__init__()
        self.out_dim = out_dim
        self.layers = num_layers
        self.indim = in_dim
        self.device = device

        nn_list = []
        for i in range(num_layers):
            nn_list.append(nn.Linear(in_dim, self.out_dim[i]).to(self.device))
            if i != num_layers - 1:
                nn_list.append(nn.ReLU().to(self.device))
            in_dim = out_dim[i]
        self.linear = nn.ModuleList(nn_list)

    def weight_init(self):
        """
        Initializes model weights
        """
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, input_tensor):
        """Performs a forward pass

        Inputs:
            input_tensor: input features with shape (batch size, in_dim)
        Returns：
            out：the classifier outputs (logits) with shape (batch size, out_dim)
            inter：the original input_tensor
        """
        for i, l in enumerate(self.linear):
            inter = input_tensor
            y = l(input_tensor)
            input_tensor = y

        out = y
        if self.layers == 1:
            assert inter.size()[-1] == self.indim
        inter = inter[np.newaxis, :]
        return out, inter

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)




