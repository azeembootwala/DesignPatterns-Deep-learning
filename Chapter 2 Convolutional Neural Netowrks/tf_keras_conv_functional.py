import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, ReLU, Flatten


def make_conv_functional_model(input_shape):

    inputs = Input(shape=(input_shape))
    X = Conv2D(16, kernel_size=(3,3), strides=(2,2), padding='same')(inputs)
    X = ReLU()(X)
    X = MaxPool2D(pool_size=(2,2), strides=(2,2))(X)
    X = Flatten()(X)
    X = Dense(512, activation = 'relu')(X)
    outputs = Dense(26, activation='softmax')(X)

    model = Model(inputs, outputs)
    model.summary()
    return model

if __name__ == '__main__':
    make_conv_functional_model((128,128,1))

