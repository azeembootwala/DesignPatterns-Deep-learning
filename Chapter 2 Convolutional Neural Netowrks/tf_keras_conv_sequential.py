import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, ReLU, Activation


def make_conv_sequential_model(input_shape):
    # input_shape = (128,128,1)

    # FrontEnd
    model = Sequential()

    model.add(Conv2D(16, kernel_size=(3,3), strides=(2,2), padding='same', input_shape=input_shape))
    model.add(ReLU())
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(Flatten())

    # Backend
    model.add(Dense(512))
    model.add(ReLU())
    model.add(Dense(26))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')

    model.summary()
    return model

if __name__ == "__main__":
    make_conv_sequential_model((128,128,1))


