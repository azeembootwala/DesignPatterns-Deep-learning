import numpy as np 
from tensorflow.keras import Model, Input 
from tensorflow.keras.layers import Dense



def model_branch():
    # An example of how a model structure looks for a branched NN
    input_shape = (3,)
    inputs = Input(input_shape)
    x = Dense(10, activation = 'relu')(inputs)
    x = Dense(10, activation = 'relu')(x)

    output1 = Dense(5, activation = 'softmax')(x)
    output2 = Dense(2, activation = 'softmax')(x)

    model = Model(inputs, [output1 , output2])

    model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam', metrics = ['accuracy'])
    model.summary()
    return model

def model_multi_label():
    # An example of a model structure for multi-label classification
    pass


def train():
    model = model_branch()

if __name__ == '__main__':
    train()