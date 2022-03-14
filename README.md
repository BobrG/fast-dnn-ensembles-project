# sk_fast_dnn_ensembles
Through the void: interpolating path for auto-encoders's parameters
# Project Team

Gleb Bobrovskikh   
Nikita Marin    
Maria Lysyuk    
Olga Grebenkova    

# Proposal
In paper ( https://arxiv.org/pdf/1802.10026.pdf ) , the authors propose a simple
computational algorithm to perform fast ensembling of deep neural networks. The
key idea of the algorithm is to find a low-loss curve between local minima θ1, θ2
on the neural network loss surface. The intermediate points θ (neural networks) on
 this curve can be used as elements of the ensemble. In this project, we will test
 this algorithm on a specific type of neural networks – autoencoders. We will find
out whether such “low-loss” paths exists between the local minima of autoencoder
models.

# Specific Tasks & Our results
- Fetch CelebA 64x64 aligned images of faces dataset.
- Prepare an convolutional autoencoder architecture for this dataset. You
may use any available implementation on the internet. Use 128 as the
bottleneck dimension.
- Train your architecture 2 times from different random starts and save all
the final checkpoints θ (weights of the autoencoder):
(a) 3 times by using mean-squared-error reconstruction loss;
(d) 3 times by using perceptual loss on VGG features (see this
implementation);
Report reconstruction losses for each autoencoder as well as provide
visual examples of reconstructed images.
- Implement the algorithm for finding a low-loss path on the loss surface
connecting a pair of given points θ1, θ2 (neural networks) by using Bezier
of Polychain curve (on your choice). You may use the paper’s
implementation;
- Within each group of 2 trained autoencoders corresponding to one of the
considered losses, apply the algorithm to connect θi , θj (for each pair i, j, 3
pairs in group). 

You have to report:
(a) Loss along the fitted curves for each pair (3 pairs × 2 experiments = 6 
curves). On the plot also represent the loss along the trivial connection of
θi , θj – segment. Your plot should look like Figures 2a, 2b of the paper.
(b) Pick a set of N = 4 images. For each curve connecting θi , θj , plot the
image by the autoencoder while its weight go along the curve from θi , θj
(ideally your reconstructions should be good along the entire curve). For
the sanity check, also plot reconstructions along the trivial connection of
θi , θj by a segment.
 - Analyse the results. Do auto-encoders also have good interpolating path
similar to the networks tested in the paper?
