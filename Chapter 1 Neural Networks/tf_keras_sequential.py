# In this script we will layout a simple format of a tf keras sequential api. 
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Activation

# We will first define the input shape for an image size of 28x28 

def make_model(input_shape):
    # This function demonstrates the vanilla method of using the tf.keras sequential api 

    model = Sequential()

    # Here we assume that our model is a rank 2 Tensor 
    model.add(Flatten(input_shape = input_shape))

    model.add(Dense(128, activation = 'relu'))

    model.add(Dense(128, activation = 'relu'))

    model.add(Dense(10, activation = 'softmax'))

    model.summary()
    return model

def make_model_list_form(input_shape):
    # Another way of using the sequential Api However this method less flexible
    model = Sequential([Flatten(input_shape = input_shape),
                        Dense(128, activation = 'relu'), 
                        Dense(128, activation = 'relu'),
                        Dense(10,  activation = 'softmax')])

    model.summary()

    return model

def train():
    model1 = make_model((28,28))
    model2 = make_model_list_form((28,28))

    # Load data function
    from tensorflow.keras.datasets import mnist

    (xtrain, ytrain), (x_test, ytest) = mnist.load_data()

    model1.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
    model1.fit(xtrain, ytrain, epochs = 10, batch_size = 32, validation_split = 0.1, verbose = 1)


if __name__ == '__main__':
    train()

