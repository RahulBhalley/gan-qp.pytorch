# -*- coding: utf-8 -*-

"""Generative Adversarial Network with Quadratic Potential
"""

from __future__ import print_function
from __future__ import division

__author__ = "Rahul Bhalley"

import torch
import torch.nn as nn
import torch.optim as optim

from config import *

if IMG_DIM == 64:
    from gan_qp_64 import *
elif IMG_DIM == 128:
    from gan_qp_128 import *
elif IMG_DIM == 256:
    from gan_qp_256 import *
elif IMG_DIM == 512:
    from gan_qp_512 import *

import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils

import os

if TRAIN:
    # Make experiments reproducible
    _ = torch.manual_seed(12345)

####################
# Make directories #
# - Samples        #
# - Checkpoints    #
####################

if not os.path.exists(NORM):
    os.mkdir(NORM)
# Directory for samples
if not os.path.exists(os.path.join(NORM, "samples", str(IMG_DIM))):
    os.mkdir(os.path.join(NORM, "samples", str(IMG_DIM)))
if not os.path.exists(os.path.join(NORM, "samples", str(IMG_DIM), DATASET)):
    os.mkdir(os.path.join(NORM, "samples", str(IMG_DIM), DATASET))
# Directory for checkpoints
if not os.path.exists(os.path.join(NORM, "checkpoints", str(IMG_DIM))):
    os.mkdir(os.path.join(NORM, "checkpoints", str(IMG_DIM)))
if not os.path.exists(os.path.join(NORM, "checkpoints", str(IMG_DIM), DATASET)):
    os.mkdir(os.path.join(NORM, "checkpoints", str(IMG_DIM), DATASET))
    
####################
# Load the dataset #
####################

def load_data():
    import psutil
    cpu_cores = psutil.cpu_count()

    transform = transforms.Compose(
        [
            transforms.Resize(IMG_DIM),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ],
    )

    print("Loading dataset")
    root = f"/Users/rahulbhalley/.torch/datasets/{DATASET}"
    if DATASET == "cifar-10":
        trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
    elif DATASET == "cifar-100":
        trainset = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=transform)
    elif DATASET == "mnist":
        trainset = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=transform)
    elif DATASET == "fashion-mnist":
        trainset = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
    else:
        trainset = torchvision.datasets.ImageFolder(root=root, transform=transform)
    dataloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=cpu_cores)
    print("Loaded dataset")

    def get_infinite_data(dataloader):
        while True:
            for images, _ in dataloader:
                yield images

    return get_infinite_data(dataloader)

if TRAIN:
    data = load_data()

################################################
# Define device, neural nets, optimizers, etc. #
################################################

# Automatic GPU/CPU device placement
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create models
g_model = Generator().to(device)
d_model = Discriminator().to(device)

# Optimizers
g_optim = optim.Adam(g_model.parameters(), 2e-4, betas=(0.5, 0.999))
d_optim = optim.Adam(d_model.parameters(), 2e-4, betas=(0.5, 0.999))

############
# Training #
############

def train():
    print("Begin training!")
    # Try loading the latest existing checkpoints based on `BEGIN_ITER`
    try:
        # Checkpoint dirs
        g_model_dir = os.path.join(NORM, "checkpoints", DATASET, f"g_model_{BEGIN_ITER}.pth")
        d_model_dir = os.path.join(NORM, "checkpoints", DATASET, f"d_model_{BEGIN_ITER}.pth")
        # Load checkpoints
        g_model.load_state_dict(torch.load(g_model_dir, map_location="cpu"))
        d_model.load_state_dict(torch.load(d_model_dir, map_location="cpu"))
        print(f"Loaded the latest checkpoints from {BEGIN_ITER}th iteration")
        print("NOTE: Set the `BEGIN_ITER` in accordance to saved checkpoints")
        # Free some memory
        del g_model_dir, d_model_dir
    except:
        print(f"Resume: Couldn't load the checkpoints from {BEGIN_ITER}th iteration")

    # Just to see the learning progress
    fixed_z = torch.randn(BATCH_SIZE, Z_DIM).to(device)

    for i in range(BEGIN_ITER, TOTAL_ITERS + 1):
        # Just because I"m encountering some problem with
        # the batch size of sampled data with `torchvision`.
        def safe_sampling():
            x_sample = data.next()
            if x_sample.size(0) != BATCH_SIZE:
                print(f"Required batch size not equal to x_sample batch size: {BATCH_SIZE} != {x_sample.size(0)} | skipping...")
                x_sample = data.next()
            return x_sample.to(device)

        #################
        # Train d_model #
        #################

        # Tune gradient computations
        for param in g_model.parameters():
            param.requires_grad_(False)
        for param in d_model.parameters():
            param.requires_grad_(True)

        for j in range(2):
            z_sample = torch.randn(BATCH_SIZE, Z_DIM).to(device) # Sample prior from Gaussian distribution
            x_sample = safe_sampling()

            with torch.no_grad():
                x_fake = g_model(z_sample)
            x_real_score = d_model(x_sample)
            x_fake_score = d_model(x_fake)

            # Zerofy the gradients
            d_optim.zero_grad()

            # Compute loss
            d_loss = x_real_score - x_fake_score
            if NORM == "l1":
                d_norm = 10 * (x_sample - x_fake).abs().mean()
            elif NORM == "l2":
                d_norm = 10 * ((x_sample - x_fake)**2).mean().sqrt()
            d_loss = - d_loss + 0.5 * d_loss**2 / d_norm
            d_loss = d_loss.mean()

            # Compute gradients
            d_loss.backward()

            # Update the network(s)
            d_optim.step()

        #################
        # Train g_model #
        #################

        # Tune gradient computations
        for param in g_model.parameters():
            param.requires_grad_(True)
        for param in d_model.parameters():
            param.requires_grad_(False)

        for j in range(1):
            z_sample = torch.randn(BATCH_SIZE, Z_DIM).to(device) # Sample prior from Gaussian distribution
            x_sample = safe_sampling()

            x_fake = g_model(z_sample)
            x_real_score = d_model(x_sample)
            x_fake_score = d_model(x_fake)

            # Zerofy the gradients
            g_optim.zero_grad()

            # Compute loss
            g_loss = x_real_score - x_fake_score
            g_loss = g_loss.mean()

            # Compute gradients
            g_loss.backward()

            # Update the network(s)
            g_optim.step()

        ##################
        # Log statistics #
        ##################

        if i % ITERS_PER_LOG == 0:
            # Print statistics
            print(f"iter: {i}, d_loss: {d_loss}, g_loss: {g_loss}")
            # Save image grids of fake and real images
            with torch.no_grad():
                samples = g_model(fixed_z)
            samples_dir = os.path.join(NORM, "samples", str(IMG_DIM), DATASET, f"test_{i}.png")
            real_samples_dir = os.path.join(NORM, "samples", str(IMG_DIM), DATASET, "real.png")
            vutils.save_image(samples, samples_dir, normalize=True)
            vutils.save_image(x_sample, real_samples_dir, normalize=True)
            # Checkpoint directories
            g_model_dir = os.path.join(NORM, "checkpoints", str(IMG_DIM), DATASET, f"g_model_{i}.pth")
            d_model_dir = os.path.join(NORM, "checkpoints", str(IMG_DIM), DATASET, f"d_model_{i}.pth")
            # Save all the checkpoints
            torch.save(g_model.state_dict(), g_model_dir)
            torch.save(d_model.state_dict(), d_model_dir)
            # Free some memory
            del g_model_dir, d_model_dir

def infer(epoch, n=10):
    g_model.eval()
    try:
        g_model_dir = os.path.join(NORM, "checkpoints", str(IMG_DIM), DATASET, f"g_model_{epoch}.pth")
        g_model.load_state_dict(torch.load(g_model_dir, map_location="cpu"))
    except:
        print("Couldn't load the checkpoint of `g_model`")

    for i in range(n):
        with torch.no_grad():
            z_sample = torch.randn(BATCH_SIZE // 4, Z_DIM).to(device) # Sample prior from Gaussian distribution
            samples = g_model(z_sample)
        samples_dir = os.path.join(NORM, "samples", str(IMG_DIM), DATASET, f"sample_{i}.png")
        vutils.save_image(samples, samples_dir, normalize=True, nrow=4)
        print(f"Saved data: {samples_dir}")

#################
# Interpolation #
#################

def slerp(start, end, val):
    start_norm = start / torch.norm(start, dim=1, keepdim=True)
    end_norm = end / torch.norm(end, dim=1, keepdim=True)
    omega = torch.acos((start_norm * end_norm).sum(1))
    so = torch.sin(omega)
    result = (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * start + (torch.sin(val * omega) / so).unsqueeze(1) * end
    return result

def interpolate(epoch, mode="lerp", n_latents=1):
    g_model.eval()
    try:
        g_model_dir = os.path.join(NORM, "checkpoints", str(IMG_DIM), DATASET, "g_model_" + str(epoch) + ".pth")
        g_model.load_state_dict(torch.load(g_model_dir, map_location="cpu"))
        print(f"Loaded the checkpoint of `g_model` at {epoch} epoch")
    except:
        print("Couldn't load the checkpoint of `g_model`")

    print(f"Performing {mode} interpolation")
    z_start = torch.randn(BATCH_SIZE, Z_DIM).to(device)
    for a in range(n_latents):
        z_end = torch.randn(BATCH_SIZE, Z_DIM).to(device)
        z_saver = z_end
        # Perform interpolation
        b = 0
        for i in torch.arange(0., 1., 0.05):
            with torch.no_grad():
                if mode == "lerp":
                    z_point = torch.lerp(z_start, z_end, i.item())
                elif mode == "slerp":
                    z_point = slerp(z_start, z_end, i.item())
                sample = g_model(z_point)
            samples_dir = os.path.join(NORM, "samples", str(IMG_DIM), DATASET, f"{mode}_{a}_{b}.png")
            vutils.save_image(sample, samples_dir, normalize=True)
            print(f"Saved data: {samples_dir}")
            b += 1
        a += 1
        z_start = z_saver

if TRAIN:
    # Train the GAN-QP
    train()
else:
    # Sample from the GAN-QP
    # infer(epoch=66000)
    interpolate(epoch=66000, mode="slerp", n_latents=20)