# Through the void: interpolating path for auto-encoders’s parameters
Project for Skoltech Machine Learning 2022 Course

## Project Team

* Gleb Bobrovskikh   
* Nikita Marin    
* Maria Lysyuk    
* Olga Grebenkova    

## Motivation
In paper ( https://arxiv.org/pdf/1802.10026.pdf ) , the authors propose a simple
computational algorithm to perform fast ensembling of deep neural networks. The
key idea of the algorithm is to find a low-loss curve between local minima θ1, θ2
on the neural network loss surface. The intermediate points θ (neural networks) on
 this curve can be used as elements of the ensemble. In this project, we will test
 this algorithm on a specific type of neural networks – autoencoders. We will find
out whether such “low-loss” paths exists between the local minima of autoencoder
models.

## Specific Tasks & Our results
:white_check_mark: Fetch CelebA 64x64 aligned images of faces dataset.
:white_check_mark: Prepare an convolutional autoencoder architecture for this dataset. You
may use any available implementation on the internet. Use 128 as the
bottleneck dimension.
:white_check_mark: Train architecture 2 times from different random starts and save all
the final checkpoints θ (weights of the autoencoder):
(a) 3 times by using mean-squared-error reconstruction loss;
(d) 3 times by using perceptual loss on VGG features;

:white_check_mark: Implement the algorithm for finding a low-loss path on the loss surface
connecting a pair of given points θ1, θ2 (neural networks) by using Bezier curve
:white_check_mark: Within each group of 2 trained autoencoders corresponding to one of the
considered losses, the algorithm to connect θi , θj applied.
