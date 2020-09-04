import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, ReLU, Activation


def make_conv_sequential_model(input_shape):
    # input_shape = (128,128,1)
    model = Sequential()

    model.add(Conv2D(16, kernel_size=(2,2), strides=(2,2), padding='same', input_shape=input_shape)
    model.add(ReLu())
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(Flatten())
    

