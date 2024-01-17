# Wasserstein Variational Auto-Encoder Demonstration
## Scope
This repo implements the Wasserstein Variational Auto-Encoder (VAE) as described in [this paper](https://openreview.net/forum?id=HkL7n1-0b).
Compared to other repos it also implements a probabilistic encoder, which makes the algorithm more robust with respect to the dimension of the latent space, as described [here](https://arxiv.org/pdf/1802.03761). 
## Notebooks
There are two notebooks. 

[This notebook](/circle_data.ipynb) randomly samples data points from a cycle and adds some gaussian noise, so that the variation of the data is contained in a one-dimensional manifold. 

[The second notebook](/mnist.ipynb) implements a CNN encoder/decoder as described in the original paper. 
## Structure
- [distributions_dists.py](/distribution_dists.py) implements the MMD distance. 
- [data.py](/data.py) defines a torch dataset with the circle data. 
- [vae.py](/vae.py) contains the encoders and decoders. 
