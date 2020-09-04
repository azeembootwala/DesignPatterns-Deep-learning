# Convolutional Neural Networks

Cobvolutional neural networks can be viewed as consisting two parts, a frontend and a backend, the backend is a simple feedforward neural network where as the frontend is kind of a preprocessor to the backend in other words "feature extractor".
Since the goal of this repository is not to learn the theory behind deep learning rather to build concept of how can we design a neural network in a resuable way. 

## CNN Basic Sequential Model
So we will now design a basic sequential model in two parts, a Convolutional frontend and feed forward backend, we start by adding a convolutional layer of 16 layers as the first layer using Conv2D class object. 

The size of each filter will be 3 x 3 and a stride of 2. So we pick an image size of 128 x 128, since the Conv2D class needs specifying the number of color chanels. We will represent the image as (128, 128, 1).

