# Model Resnet50
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, LeakyReLU, ReLU, MaxPool2D, Add, BatchNormalization, AveragePooling2D, \
                                    ZeroPadding2D, Flatten, Dense


def convolutional_block(n_filters, X, s=2):
    """ Create a Block of Convolution without pooling
    n_filters: Number of filters 
    X        : Input to the block
    """
    shortcut = X
    X = Conv2D(n_filters, kernel_size=(1,1), strides=(s,s), padding='valid')(X)
    X = BatchNormalization(axis = 3)(X)
    X = LeakyReLU()(X)
    X = Conv2D(n_filters, kernel_size=(3,3), strides=(1,1), padding='same')(X)
    X = BatchNormalization(axis = 3)(X)
    X = LeakyReLU()(X)
    X = Conv2D(n_filters * 4, kernel_size=(1,1), strides=(1,1), padding= 'valid')(X)
    X = BatchNormalization(axis = 3)(X)

    shortcut = Conv2D(n_filters * 4, kernel_size=(1,1), strides=(s,s), padding='valid')(shortcut)
    shortcut = BatchNormalization(axis = 3)(shortcut)

    X = Add()([shortcut,X])
    X = LeakyReLU()(X)

    return X

def bottleneck_block(n_filters, X):
    """ Create a BolltenecK Block of Convolutions
    n_filters: Number of filters
    X        : Input of the block
    """
    shortcut = X
    X = Conv2D(n_filters, kernel_size=(1,1), strides=(1,1), padding='same')(X)
    X = BatchNormalization(axis = 3)(X)
    X = LeakyReLU()(X)
    X = Conv2D(n_filters, kernel_size=(3,3), strides=(1,1), padding='same')(X)
    X = BatchNormalization(axis = 3)(X)
    X = LeakyReLU()(X)
    X = Conv2D(n_filters * 4, kernel_size=(1,1),strides=(1,1),padding='same')(X)
    X = BatchNormalization(axis = 3)(X)
    X = Add()([shortcut, X])
    X = LeakyReLU()(X)
    return X


def build_resnet50(input_shape):
    # Build Resnet 50  using procedurally generated loops
    
    inputs = Input(input_shape)
    X = ZeroPadding2D((3,3))(inputs)

    # X = AveragePooling2D((2,2))(X)

    X = Conv2D(64, kernel_size=(7,7), strides=(2,2), padding='valid')(X)
    X = BatchNormalization(axis = 3)(X)
    X = LeakyReLU()(X)
    X = MaxPool2D((3,3), strides=(2,2))(X)

    X = convolutional_block(64, X, s=1)
    X = bottleneck_block(64, X)
    X = bottleneck_block(64, X)

    X = convolutional_block(128, X, s=2)
    X = bottleneck_block(128, X)
    X = bottleneck_block(128, X)
    X = bottleneck_block(128, X)

    X = convolutional_block(256, X, s=2)
    X = bottleneck_block(256, X)
    X = bottleneck_block(256, X)
    X = bottleneck_block(256, X)
    X = bottleneck_block(256, X)
    X = bottleneck_block(256, X)

    X = convolutional_block(512, X, s=2)
    X = bottleneck_block(512, X)
    X = bottleneck_block(512, X)

    X = AveragePooling2D((4,4))(X)

    model = Model(inputs = inputs, outputs = X, name='Resnet50')

    model.summary()
    return model

if __name__ == '__main__':
    build_resnet50((160, 160, 3))