# -*- coding: utf-8 -*-

from __future__ import print_function

__author__ = "Rahul Bhalley"

# Some configuration for networks
IMG_DIM = 128 # [64, 128, 256, 512]
Z_DIM = 128
BATCH_SIZE = 64
NORM = "l1" # [l1, l2]
TRAIN = False

# Some other configurations
DATASET = "celeba"
N_CHANNELS = 3
BEGIN_ITER = 66000
TOTAL_ITERS = 1000000
ITERS_PER_LOG = 100
VERBOSE = False

print("------------------")
print("| Configurations |")
print("------------------")
print("")
print(f"IMG_DIM:         {IMG_DIM}")
print(f"Z_DIM:           {Z_DIM}")
print(f"BATCH_SIZE:      {BATCH_SIZE}")
print(f"NORM:            {NORM}")
print(f"DATASET:         {DATASET}")
print(f"N_CHANNELS:      {N_CHANNELS}")
print(f"BEGIN_ITER:      {BEGIN_ITER}")
print(f"TOTAL_ITERS:     {TOTAL_ITERS}")
print(f"ITERS_PER_LOG:   {ITERS_PER_LOG}")
print(f"VERBOSE:         {VERBOSE}")
print("")