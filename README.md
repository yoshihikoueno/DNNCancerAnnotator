# AI to predict prostate cancer annotations using DNN (Deep Neural Network)

## Framework
<img src="https://www.gstatic.com/devrel-devsite/vbb62cc5a3e8f17e37bae4792b437f28f787df3f9cf9732cbfcc99b4f4ff41a54/tensorflow/images/lockup.svg" alt="tensorflow" width="300">
Estimator API

## Dataset
Private datasets consists of t2-weighted MRI images.

## Models
 * UNet
 * Pix2Pix
 * GAN

## Data augmentations
 * Flipping
 * Warping

## Data Format
 Compatible with both 2D and 3D

## Getting Started
### Install
 ```bash
 # Clone
 git clone https://github.com/yoshihikoueno/DNNCancerAnnotator
 cd DNNCancerAnnotator
 python3 setup.py
 ```

### Train, Evaluate, Predict
 ```bash
 # train
 python3 -m runs.train

 # see the usage
 python3 -m runs.train -h

 # evaluate
 python3 -m runs.evaluate

 # predict
 python3 -m runs.predict
 ```
