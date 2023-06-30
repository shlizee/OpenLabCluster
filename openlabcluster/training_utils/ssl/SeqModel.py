"""
OpenLabCluster: Active Learning Based Clustering and Classification of Animal Behaviors in Videos Based on Automatically Extracted Kinematic Body Keypoints
Copyright (c) 2022-2023 University of Washington. Developed in UW NeuroAI Lab by Jingyuan Li.
"""
import torch
import torch.nn as nn
from openlabcluster.training_utils.ssl.EnDeModel import EncoderRNN, DecoderRNN, Classification

class seq2seq(nn.Module):
    def __init__(self, en_input_size, en_hidden_size, output_size, batch_size,
                 en_num_layers=3, de_num_layers=1,
                 fix_state=False, fix_weight=False, teacher_force=False, device='cuda'):
        """Initializes a sequence (encoder) to sequence (decoder) model
        Inputs:
            en_input_size: the feature size of the input
            en_hidden_size: the hidden states dimension of the encoder
            output_size: the output size of the decoder
            batch_size: the number of samples in a batch
            en_num_layers: the number of layers for the encoder
            de_num_layers: the number of layers for the decoder
            fix_state: if true, fix hidden states of the decoder with the encoder hidden states at the last time step,
                       otherwise updates the decoder hidden states at each timestep
            fix_weigh: if true, fix the decoder weight during decoding
            teacher_force: if true, uses ground truth sequences as inputs to the decoder
            device: str, either 'cpu' or 'cuda'
        """
        super(seq2seq, self).__init__()
        self.batch_size = batch_size
        self.en_num_layers = en_num_layers
        self.device = device
        self.encoder = EncoderRNN(en_input_size, en_hidden_size, en_num_layers, device).to(self.device)
        self.decoder = DecoderRNN(output_size, en_hidden_size * 2, de_num_layers).to(self.device)
        self.fix_state = fix_state
        self.fix_weight = fix_weight

        if self.fix_weight:
            with torch.no_grad():
                # Fixes decoder weight.
                self.decoder.gru.requires_grad = False
                self.decoder.out.requires_grad = False

        self.en_input_size = en_input_size
        self.teacher_force = teacher_force

    def forward(self, input_tensor, seq_len):
        """Performs a forward pass.
        Inputs:
            input_tensor: a batch of keypoint sequences in shape (batch size, sequence length, input size)
            seq_len: the sequence lengths of input tensors before padding
        """
        self.batch_size = len(seq_len)

        encoder_hidden = self.encoder(
            input_tensor, seq_len)

        decoder_output = torch.Tensor().to(self.device)
        # Decoder part
        if self.teacher_force:
            de_input = torch.zeros([self.batch_size, 1, self.en_input_size], dtype=torch.float)
            de_input = torch.cat((de_input, input_tensor[:, 1:, :]), dim=1).to(self.device)
        else:
            de_input = torch.zeros(input_tensor.size(), dtype=torch.float).to(self.device)

        if self.fix_state:
            # Fix_state: using the same state as the input to decoder model
            de_input = input_tensor[:, 0:1,:]
            for it in range(max(seq_len)):
                deout_tmp, _ = self.decoder(
                    de_input, encoder_hidden)
                deout_tmp = deout_tmp + de_input
                de_input = deout_tmp
                decoder_output = torch.cat((decoder_output, deout_tmp), dim=1)
        else:
            # Updating hidden states after each iteration
            hidden = encoder_hidden
            for it in range(max(seq_len)):
                deout_tmp, hidden = self.decoder(
                    de_input[:, it:it + 1, :], hidden)
                decoder_output = torch.cat((decoder_output, deout_tmp), dim=1)

        return encoder_hidden, decoder_output

class SemiSeq2Seq(nn.Module):
    def __init__(self, en_input_size, en_hidden_size, output_size, batch_size, cla_dim,
                 en_num_layers=3, de_num_layers=1, cl_num_layers=1,
                 fix_state=False, fix_weight=False, teacher_force=False, device='cuda'):
        """Initializes a sequence-to-sequence model with an additional classification layer
        Inputs:
            en_input_size: the feature size of the input
            en_hidden_size: the hidden states dimension of the encoder
            output_size: the output size of the decoder
            batch_size: the number of samples in a batch.
            cla_dim: the output dimension of the classification layer, usually the number of classes.
            en_num_layers: the number of layers for the encoder.
            de_num_layers: the number of layers for the decoder.
            cl_num_layers: the number of layers for the classifier.
            fix_state: if true, fix hidden states of the decoder with the encoder hidden states at the last time step,
                       otherwise updates the decoder hidden states at each timestep.
            fix_weigh: if true, fix the decoder weight during decoding.
            teacher_force: use ground truth sequence as input to the decoder.
            device: str, either 'cpu' or 'cuda'.
        """
        super(SemiSeq2Seq, self).__init__()

        self.seq = seq2seq(en_input_size, en_hidden_size, output_size, batch_size,
                           en_num_layers=en_num_layers, de_num_layers=de_num_layers,
                           fix_state=fix_state, fix_weight=fix_weight, teacher_force=teacher_force, device=device)
        self.classifier = Classification(en_hidden_size * 2, cla_dim, cl_num_layers, device)

    def forward(self, input_tensor, seq_len):
        """Performs a forward pass.
        Inputs:
            input_tensor: a batch of keypoint sequences with shape (batch size, sequence length, input size)
            seq_len: the sequence lengths of input tensors before padding
        Outputs:
            inter: encoder hidden states
            deout: decoder outputs (regenerated input sequence)
            pred:  logits for each class with shape (batch size, the number of classes)
        """
        encoder_hidden, deout = self.seq(input_tensor, seq_len)
        pred, inter = self.classifier(encoder_hidden[0, :, :])

        return inter, deout, pred



