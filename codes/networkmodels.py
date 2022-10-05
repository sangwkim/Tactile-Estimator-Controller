#!/usr/bin/env python

import argparse, torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
from utils_network import *

class EncoderCNN(nn.Module):
    def __init__(self,
                 img_x=200,
                 img_y=200,
                 input_channels=1,
                 fc_hidden1=1024,
                 fc_hidden2=768,
                 drop_p=0.5,
                 CNN_embed_dim=512):
        super(EncoderCNN, self).__init__()

        self.img_x = img_x
        self.img_y = img_y
        self.CNN_embed_dim = CNN_embed_dim

        # CNN architechtures
        self.ch1, self.ch2, self.ch3, self.ch4 = 8, 16, 32, 64
        self.k1, self.k2, self.k3, self.k4 = (5, 5), (3, 3), (3, 3), (
            3, 3)  # 2d kernal size
        self.s1, self.s2, self.s3, self.s4 = (2, 2), (2, 2), (2, 2), (
            2, 2)  # 2d strides
        self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0), (0, 0), (0, 0), (
            0, 0)  # 2d padding

        # conv2D output shapes
        self.conv1_outshape = conv2D_output_size((self.img_x, self.img_y),
                                                 self.pd1, self.k1,
                                                 self.s1)  # Conv1 output shape
        self.conv2_outshape = conv2D_output_size(self.conv1_outshape, self.pd2,
                                                 self.k2, self.s2)
        self.conv3_outshape = conv2D_output_size(self.conv2_outshape, self.pd3,
                                                 self.k3, self.s3)
        self.conv4_outshape = conv2D_output_size(self.conv3_outshape, self.pd4,
                                                 self.k4, self.s4)

        # fully connected layer hidden nodes
        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels,
                      out_channels=self.ch1,
                      kernel_size=self.k1,
                      stride=self.s1,
                      padding=self.pd1),
            nn.BatchNorm2d(self.ch1, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch1,
                      out_channels=self.ch2,
                      kernel_size=self.k2,
                      stride=self.s2,
                      padding=self.pd2),
            nn.BatchNorm2d(self.ch2, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch2,
                      out_channels=self.ch3,
                      kernel_size=self.k3,
                      stride=self.s3,
                      padding=self.pd3),
            nn.BatchNorm2d(self.ch3, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch3,
                      out_channels=self.ch4,
                      kernel_size=self.k4,
                      stride=self.s4,
                      padding=self.pd4),
            nn.BatchNorm2d(self.ch4, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        self.drop = nn.Dropout(self.drop_p)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(
            self.ch4 * self.conv4_outshape[0] * self.conv4_outshape[1],
            self.fc_hidden1)
        self.fc2 = nn.Linear(
            self.fc_hidden1,
            self.CNN_embed_dim)  # output = CNN embedding latent variables

    def forward(self, x_3d):
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            # CNNs
            x = self.conv1(x_3d[:, t, :, :, :])
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = x.view(x.size(0), -1)  # flatten the output of conv
            x = self.drop(x)
            # FC layers
            x = F.relu(self.fc1(x))
            x = self.drop(x)
            x = self.fc2(x)
            cnn_embed_seq.append(x)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        # cnn_embed_seq: shape=(batch, time_step, input_size)

        return cnn_embed_seq


class DecoderRNN(nn.Module):
    def __init__(self,
                 CNN_embed_dim=512,
                 h_RNN_layers=1,
                 h_RNN=512,
                 h_FC_dim=256,
                 drop_p=0.5,
                 output_dim=6):
        super(DecoderRNN, self).__init__()

        self.RNN_input_size = 4 * CNN_embed_dim
        self.h_RNN_layers = h_RNN_layers  # RNN hidden layersssssss
        self.h_RNN = h_RNN  # RNN hidden nodes
        self.h_FC_dim = h_FC_dim
        self.drop_p = drop_p
        self.output_dim = output_dim

        self.LSTM = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=self.h_RNN,
            num_layers=self.h_RNN_layers,
            batch_first=
            True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            dropout=self.drop_p)

        self.fc1 = nn.Linear(self.h_RNN, self.h_FC_dim)
        self.bn = nn.BatchNorm1d(self.h_FC_dim)
        self.fc2 = nn.Linear(self.h_FC_dim, self.output_dim)
        self.drop = nn.Dropout(self.drop_p)

    def forward(self, x1_init, x1, x2_init, x2, device):

        self.LSTM.flatten_parameters()

        RNN_out, (h_n, h_c) = \
            self.LSTM(torch.cat((x1_init, x1, x2_init, x2), 2), None)
        x = self.fc1(RNN_out)
        x = F.relu(x)
        x = self.drop(x)
        x = torch.tanh(self.fc2(x))

        return x

    def forward_single(self, x1_init, x1, x2_init, x2, h_nc, device):
        self.LSTM.flatten_parameters()
        RNN_out, h_nc_ = \
            self.LSTM(torch.cat((x1_init, x1, x2_init, x2), 2), h_nc)
        x = self.fc1(RNN_out)
        x = F.relu(x)
        x = self.drop(x)
        x = torch.tanh(self.fc2(x))
        return x, h_nc_


class DecoderFC(nn.Module):
    def __init__(self,
                 CNN_embed_dim=512,
                 FC_layer_nodes=[512, 512, 256],
                 drop_p=0.5,
                 output_dim=6):
        super(DecoderFC, self).__init__()

        self.FC_input_size = 4 * CNN_embed_dim
        self.FC_layer_nodes = FC_layer_nodes
        self.drop_p = drop_p
        self.output_dim = output_dim

        assert len(FC_layer_nodes) == 3

        self.fc1 = nn.Linear(self.FC_input_size, self.FC_layer_nodes[0])
        self.fc2 = nn.Linear(self.FC_layer_nodes[0], self.FC_layer_nodes[1])
        self.fc3 = nn.Linear(self.FC_layer_nodes[1], self.FC_layer_nodes[2])
        self.fc4 = nn.Linear(self.FC_layer_nodes[2], self.output_dim)
        self.drop = nn.Dropout(self.drop_p)

    def forward(self, x11, x12, x21, x22, device):

        x = self.fc1(torch.cat((x11, x12, x21, x22), -1))
        x = F.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.drop(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.drop(x)
        x = self.fc4(x)
        x = torch.tanh(x)

        return x