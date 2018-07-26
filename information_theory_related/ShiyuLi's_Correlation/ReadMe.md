# Insight Into Correlation of Feature Maps
## Introduction
This experiment aims at Visualizaing the correlations of the feature maps of the CNN. And it's at the very initial stages.
## Experiment Setup
We use the simple LeNet5 with MNIST dataset. It will converge at around 10 epochs but I do the training for 400 epochs. And we also logged the gradient, normed_weight (not been used yet) and **the output of each layer** on the test set.
## Requirements
_The version annoted is what I'm currently using_
- Keras         2.20.0
- Tensorflow    1.9.0
- Numpy         1.14.0
- Matplotlib    2.20.0
- Seaborn
- PIL
- imageio

## Code
- `Utils.py` is for loading and preprocessing the datasets
- `loggingreport.py` is the callback class which will log the activity during the training
- `CNN_SaveActivation.py` construct and train the LeNet5 Network and save the activity to 'rawdata/' directory. **The Log File might be very large (16GB for 400 epochs)**
- `stat_data.py` loading the saved log and calculating the correlation, and output pictures and animation file.
- `convertGIF.py` create gif animation from a series of pictures.

## Usage
First train and log the epochs with
    
    CNN_SaveActivation.py

Then run

    stat_data.py

And you will get the animations and the heatmap as well as the activation plot of each feature map in folder 'img_out'

## LICENSE
This code is modified from https://github.com/artemyk/ibsgd
>Andrew Michael Saxe, Yamini Bansal, Joel Dapello, Madhu Advani, Artemy Kolchinsky, Brendan Daniel Tracey, David Daniel Cox, On the Information Bottleneck Theory of Deep Learning, _ICLR 2018_.
