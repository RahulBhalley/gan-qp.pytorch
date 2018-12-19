# -*- coding: utf-8 -*-

from __future__ import print_function

__author__ = "Rahul Bhalley"

# Some configuration for networks
IMG_DIM = 256 # [64, 128, 256, 512]
Z_DIM = 128
BATCH_SIZE = 64
NORM = 'l1' # [l1, l2]
TRAIN = False

# Some other configurations
DATASET = 'celeba'
N_CHANNELS = 3
BEGIN_ITER = 0
TOTAL_ITERS = 1000000
ITERS_PER_LOG = 1
VERBOSE = False

print('------------------')
print('| Configurations |')
print('------------------')
print('')
print('IMG_DIM:         {}'.format(IMG_DIM))
print('Z_DIM:           {}'.format(Z_DIM))
print('BATCH_SIZE:      {}'.format(BATCH_SIZE))
print('NORM:            {}'.format(NORM))
print('DATASET:         {}'.format(DATASET))
print('N_CHANNELS:      {}'.format(N_CHANNELS))
print('BEGIN_ITER:      {}'.format(BEGIN_ITER))
print('TOTAL_ITERS:     {}'.format(TOTAL_ITERS))
print('ITERS_PER_LOG:   {}'.format(ITERS_PER_LOG))
print('VERBOSE:         {}'.format(VERBOSE))
print('')
