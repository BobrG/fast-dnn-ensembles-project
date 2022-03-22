# Through the void: interpolating path for auto-encoders’s parameters
This repository contains a PyTorch implementation of the curve-finding for autoencoders. The project inspired by
[Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs](https://arxiv.org/abs/1802.10026)

by Timur Garipov, Pavel Izmailov, Dmitrii Podoprikhin, Dmitry Vetrov and Andrew Gordon Wilson (NIPS 2018, Spotlight).

## Project Team

* Gleb Bobrovskikh   
* Nikita Marin    
* Maria Lysyuk    
* Olga Grebenkova    

## Motivation
Although performance of deep neural networks(DNNs)
made a big leap forward during last decade, it is still a
challenge to train them. It happens because the loss surfaces
of DNN are highly non-convex and can depend on millions
of parameters.Usually it is supposed that the loss surfaces of DNN have
multiple isolated local optima. In
contrast to this idea, paper [Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs](https://arxiv.org/abs/1802.10026)  provide new training procedures which can in
fact find paths of near-constant accuracy between the modes
of large deep neural networks. Furthermore, it can be shown
that these curves can be rather simple as a polygonal chain
of two line segments.This geometric
discovery may have great impact in research into multilayer
networks including improving the training and creating bet-
ter ensembles. However, the existence of such ”low-loss”
curves was proved only for ResNet and VGG models.

In our project we focused on autoencoders — a specific type of a neural network, which is mainly de-
signed to encode the input into a compressed and mean-
ingful representation, and then decode it back such that
the reconstructed input is similar as possible to the origi-
nal one. Using the proposed method we carry out exper-
iments with convolutional autoencoder on two different
losses with: Mean Squared Error and perceptual loss on
VGG features.

# Dependencies
* [PyTorch](http://pytorch.org/)
* [torchvision](https://github.com/pytorch/vision/)
* [tabulate](https://pypi.python.org/pypi/tabulate/)

# Dataset downloading
```bash

chmod +x download.sh
./download.sh
unzip -q CelebA_128crop_FD.zip?dl=0 -d ./data/
              
```

# Usage

The code in this repository implements the curve-finding procedure for autoencoders with examples on the CelebA dataset.

## Curve Finding


### Training the autoencoders

To run the curve-finding procedure, you first need to train the two autoencoders that will serve as the end-points of the curve. You can train the endpoints using the following command

```bash
python3 main.py --data-dir=<DIR> \
                 --batch-size=<BATCH_SIZE> \
                 --loss=<LOSS> \
                 ----save-every-epoch=<SAVE> \
                 --checkpoint=<CHECK> \
                 --epochs=<EPOCHS> \
                 --lr=<LR_INIT> \
                 --seed=<SEED> \
              
```

Parameters:

* ```DIR``` &mdash; path to the data directory (default: './data/celeba')
* ```BATCH_SIZE``` &mdash; batch size for train and test (default: 64)
* ```LOSS``` &mdash; loss: mse or vgg perceptual loss
* ```EPOCHS``` &mdash; number of training and testing epochs (default: 30)
* ```LR_INIT``` &mdash; initial learning rate (default: 1e-4)
* ```SEED``` &mdash; random seed to initialize pytorch (default: 42)
* ```CHECK``` &mdash; path to model checkpoint
* ```SAVE``` &mdash; save model weights and images every N epochs


### Training the curves

Once you have two checkpoints to use as the endpoints you can train the curve connecting them using the following comand.

```bash
python3 train.py --dir=<DIR> \
                 --dataset=<DATASET> \
                 --data_path=<PATH> \
                 --transform=<TRANSFORM>
                 --model=<MODEL> \
                 --epochs=<EPOCHS> \
                 --lr=<LR_INIT> \
                 --wd=<WD> \
                 --loss =<LOSS>\
                 --curve=<CURVE>[Bezier|PolyChain] \
                 --num_bends=<N_BENDS> \
                 --init_start=<CKPT1> \ 
                 --init_end=<CKPT2> \
                 [--fix_start] \
                 [--fix_end] \
                 [--use_test]
```

Parameters:

* ```CURVE``` &mdash; desired curve parametrization [Bezier|PolyChain] 
* ```N_BENDS``` &mdash; number of bends in the curve (default: 3)
* ```CKPT1, CKPT2``` &mdash; paths to the checkpoints to use as the endpoints of the curve
* * ```LOSS``` &mdash; loss: mse or vgg perceptual loss

Use the flags `--fix_end --fix_start` if you want to fix the positions of the endpoints; otherwise the endpoints will be updated during training. See the section on [training the endpoints](https://github.com/izmailovpavel/curves-dnn-loss-surfaces/blob/master/README.md#training-the-endpoints)  for the description of the other parameters.


### Evaluating the curves

To evaluate the found curves, you can use the following command
```bash
python3 eval_curve.py --dir=<DIR> \
                 --dataset=<DATASET> \
                 --data_path=<PATH> \
                 --transform=<TRANSFORM>
                 --model=<MODEL> \
                 --wd=<WD> \
                 --loss =<LOSS>\
                 --curve=<CURVE>[Bezier|PolyChain] \
                 --num_bends=<N_BENDS> \
                 --ckpt=<CKPT> \ 
                 --num_points=<NUM_POINTS> \
                 [--use_test]
```
Parameters
* ```CKPT``` &mdash; path to the checkpoint saved by `train.py`
* ```NUM_POINTS``` &mdash; number of points along the curve to use for evaluation (default: 61)
* ```LOSS``` &mdash; loss: mse or vgg perceptual loss


`eval_curve.py` outputs the statistics on train and test loss and error along the curve. It also saves a `.npz` file containing more detailed statistics at `<DIR>`.


## Specific Tasks & Our results
- ☑️ Fetch CelebA 64x64 aligned images of faces dataset.
- ☑️ Prepare an convolutional autoencoder architecture for this dataset. Use 128 as the
bottleneck dimension.
- ☑️ Train architecture 2 times from different random starts and save all
 the final checkpoints θ (weights of the autoencoder):  
(a) 3 times by using mean-squared-error reconstruction loss;  
(d) 3 times by using perceptual loss on VGG features;  

- ☑️ Implement the algorithm for finding a low-loss path on the loss surface
connecting a pair of given points θ1, θ2 (neural networks) by using Bezier curve
- ☑️ Within each group of 2 trained autoencoders corresponding to one of the
considered losses, the algorithm to connect θi , θj applied.
