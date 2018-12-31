# Generative Adversarial Network with Quadratic Potential

This is a minimal [PyTorch](https://pytorch.org/) code for [GAN-QP](https://arxiv.org/abs/1811.07296) *without gradient vanishing* which has *no Lipschitz constraint* like W-GANs on critic network. Also it is not trained to minimize Wasserstein divergence! 

It's a totally different GAN with stable training even with high resolution data without requiring careful hyper-parameters and network architecture configurations!

Once again thanks [Jianlin Su](https://github.com/bojone/gan-qp), creator of GAN-QP, for suggesting this code as [PyTorch](https://pytorch.org/) implementation of [GAN-QP](https://arxiv.org/abs/1811.07296)!

## Experiments

I performed my own experiments on couple of datasets:
- CelebFaces
- LSUN Bedrooms

I trained the images on [128](https://github.com/rahulbhalley/gan-qp.pytorch/blob/master/gan_qp_128.py), [256](https://github.com/rahulbhalley/gan-qp.pytorch/blob/master/gan_qp_256.py), and [512](https://github.com/rahulbhalley/gan-qp.pytorch/blob/master/gan_qp_512.py) sized [GAN-QP](https://arxiv.org/abs/1811.07296).

**Note**: Training is not yet complete! And maybe that's why results are not that good. I'll try to update these images when I'm done.

### 128 x 128 Resolution
#### CelebFaces
![](https://raw.githubusercontent.com/rahulbhalley/gan-qp.pytorch/master/imgs/celeba-128.png)

#### LSUN Bedrooms
![](https://raw.githubusercontent.com/rahulbhalley/gan-qp.pytorch/master/imgs/lsun-128.png)

### 256 x 256 Resolution
#### CelebFaces
![](https://raw.githubusercontent.com/rahulbhalley/gan-qp.pytorch/master/imgs/celeba-256.png)

### 512 x 512 Resolution
#### CelebFaces
![](https://raw.githubusercontent.com/rahulbhalley/gan-qp.pytorch/master/imgs/celeba-512.png)

## References
- GAN-QP: A Novel GAN Framework without Gradient Vanishing and Lipschitz Constraint [[arXiv](https://arxiv.org/abs/1811.07296)]
- Original [GAN-QP](https://github.com/bojone/gan-qp) implementation
