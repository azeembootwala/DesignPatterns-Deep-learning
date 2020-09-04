# Model Resnet
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, LeakyReLU, ReLU, MaxPool2D, Add

activation = LeakyReLU()

def residual_block(n_filters, X):
    """ Create a Residual Block of Convolutions
    n_filters: number of filters
    X        : Input to the block 

    """
    shortcut = X
    X = Conv2D(n_filters, kernel_size=(3,3), strides=(1,1),padding='same')(X)
    X = activation(X)
    X = Conv2D(n_filters, kernel_size=(3,3), strides=(1,1), padding='same')(X)
    X = activation(X)
    X = Add([shortcut,X])
    return X

def convolutional_block(n_filters, X):
    """ Create a Block of Convolution without pooling
    n_filters: Number of filters 
    X        : Input to the block
    """
    X = Conv2D(n_filters, kernel_size=(3,3), strides=(2,2), padding='same')(X)
    X = activation(X)
    X = Conv2D(n_filters, kernel_size=(3,3), strides=(2,2), padding='same')(X)
    X = activation(X)
    return X

def bottleneck_block(n_filters, X):
    """ Create a BolltenecK Block of Convolutions
    n_filters: Number of filters
    X        : Input of the block
    """
    shortcut = X
    X = Conv2D(n_filters, kernel_size=(1,1), strides=(1,1), padding='same')(X)
    X = activation(X)
    X = Conv2D(n_filters, kernel_size=(3,3), strides=(1,1), padding='same')(X)
    X = activation(X)
    X = Conv2D(n_filters * 4,kernel_size=(1,1),strides=(1,1),padding='same')(X)
    X = Add([shortcut, X])
    return X


def build_resnet50():
    # Build Resnet 50  using procedurally generated loops
    pass
