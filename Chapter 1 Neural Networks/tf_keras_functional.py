# This script uses the TF.Keras functional Api 

import numpy as np
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Flatten


def make_functional_model(input_shape):
    inputs = Input(input_shape)

    x = Flatten()(inputs)
    x = Dense(128, activation = 'relu')(x)
    x = Dense(512, activation = 'relu')(x)
    outputs = Dense(10, activation = 'softmax')(x)

    return Model(inputs, outputs)

def model_branch():
    # An example of how a model structure looks for a branched NN
    pass 

def model_multi_label():
    # An example of a model structure for multi-label classification
    pass

def train():
    model = make_functional_model((28,28))
    from tensorflow.keras.datasets import mnist
    (xtrain, ytrain), (x_test, ytest) = mnist.load_data()

    model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])

    model.fit(xtrain, ytrain, epochs = 10, batch_size = 32, validation_split = 0.1, verbose = 1)

if __name__ == '__main__':
    train()