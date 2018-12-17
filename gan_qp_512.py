# -*- coding: utf-8 -*-

from __future__ import print_function

__author__ = "Rahul Bhalley"

import torch
import torch.nn as nn

from config import *

###################
# Neural Networks #
###################
# --------------- #
# Generator       #
# --------------- #
# Discriminator   #
# --------------- #
###################


#############
# Generator #
#############

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        block1 = [
            nn.Linear(Z_DIM, 4 * 4 * 4096),
            nn.BatchNorm1d(4 * 4 * 4096),
            nn.ReLU()
        ]

        block2 = [
            nn.ConvTranspose2d(4096, 2048, 4, 2, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU()
        ]

        block3 = [
            nn.ConvTranspose2d(2048, 1024, 4, 2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        ]

        block4 = [
            nn.ConvTranspose2d(1024, 512, 4, 2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        ]

        block5 = [
            nn.ConvTranspose2d(512, 256, 4, 2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        ]

        block6 = [
            nn.ConvTranspose2d(256, 128, 4, 2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        ]

        block7 = [
            nn.ConvTranspose2d(128, 64, 4, 2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        ]

        block8 = [
            nn.ConvTranspose2d(64, N_CHANNELS, 4, 2, padding=1),
            nn.Tanh()
        ]

        all_blocks = block2 + block3 + block4 + block5 + block6 + block7 + block8
        self.main1 = nn.Sequential(*block1)
        self.main2 = nn.Sequential(*all_blocks)

        # Initialize with Xavier Glorot
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        # Free some memory
        del all_blocks, block1, block2, block3, block4, block5, block6, block7, block8

        # Print the summary
        if VERBOSE:
            self.summary()

    def forward(self, x):
        for layer in self.main1:
            x = layer(x)
        x = x.view(-1, 4096, 4, 4)  # Reshape for convolution
        for layer in self.main2:
            x = layer(x)
        return x

    def summary(self):
        x = torch.zeros(BATCH_SIZE, Z_DIM)
        
        # Print the title in a good design
        # for easy recognition.
        print()
        summary_title = '| {} Summary |'.format(self.__class__.__name__)
        for _ in range(len(summary_title)):
            print('-', end='')
        print()
        print(summary_title)
        for _ in range(len(summary_title)):
            print('-', end='')
        print('\n')
        
        # Run forward pass while not tracking history on
        # tape using `torch.no_grad()` for printing the
        # output shape of each neural layer operation.
        print('Input: {}'.format(x.size()))
        with torch.no_grad():
            for layer in self.main1:
                x = layer(x)
                print('Out: {} \tLayer: {}'.format(x.size(), layer))
            x = x.view(-1, 4096, 4, 4)  # Reshape for convolution
            print('Out: {} \tLayer: {}'.format(x.size(), 'Reshape'))
            for layer in self.main2:
                x = layer(x)
                print('Out: {} \tLayer: {}'.format(x.size(), layer))


#################
# Discriminator #
#################

class Discriminator(nn.Module):
    """
    Down scaling by a factor of 2
        combination #1
        k = 6 | s = 2 | p = 2
        combination #2 --- Best for me!
        k = 4 | s = 2 | p = 1
        combination #3 (s = 1.333)
        k = 4 | s = 1 | p = 0
        combination #4 (p = 1.5)
        k = 5 | s = 2 | p = 1
    """

    def __init__(self):
        super(Discriminator, self).__init__()

        block1 = [
            nn.Conv2d(N_CHANNELS, 64, 4, 2, padding=1, bias=False),
            nn.LeakyReLU(0.2)
        ]

        block2 = [
            nn.Conv2d(64, 128, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        ]

        block3 = [
            nn.Conv2d(128, 256, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        ]

        block4 = [
            nn.Conv2d(256, 512, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        ]

        block5 = [
            nn.Conv2d(512, 1024, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2)
        ]

        block6 = [
            nn.Conv2d(1024, 2048, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(2048),
            nn.LeakyReLU(0.2)
        ]

        block7 = [
            nn.Conv2d(2048, 4096, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(4096),
            nn.LeakyReLU(0.2)
        ]
        
        all_blocks = block1 + block2 + block3 + block4 + block5 + block6 + block7
        self.conv = nn.Sequential(*all_blocks)
        self.linear = nn.Linear(4 * 4 * 4096, 1, bias=False)

        # Initialize with Xavier Glorot
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        # Free some memory
        del all_blocks, block1, block2, block3, block4, block5, block6, block7

        # Print summary
        if VERBOSE:
            self.summary()

    def forward(self, x):
        for layer in self.conv:
            x = layer(x)
        x = x.view(-1, self.num_flat_features(x))  # flatten the conv output
        x = self.linear(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def summary(self):
        x = torch.zeros(BATCH_SIZE, N_CHANNELS, IMG_DIM, IMG_DIM)
        
        # Print the title in a good design
        # for easy recognition.
        print()
        summary_title = '| {} Summary |'.format(self.__class__.__name__)
        for _ in range(len(summary_title)):
            print('-', end='')
        print()
        print(summary_title)
        for _ in range(len(summary_title)):
            print('-', end='')
        print('\n')
        
        # Run forward pass while not tracking history on
        # tape using `torch.no_grad()` for printing the
        # output shape of each neural layer operation.
        print('Input: {}'.format(x.size()))
        with torch.no_grad():
            for layer in self.conv:
                x = layer(x)
                print('Out: {} \tLayer: {}'.format(x.size(), layer))
            x = x.view(-1, self.num_flat_features(x))  # flatten the conv output
            print('Out: {} \tlayer: {}'.format(x.size(), 'Reshape: Flatten'))
            x = self.linear(x)
            print('Out: {} \tLayer: {}'.format(x.size(), layer))
'''
g_model = Generator()
g_model.summary()

d_model= Discriminator()
d_model.summary()
#'''