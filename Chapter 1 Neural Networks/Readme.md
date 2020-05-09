# Neural Networks 

In this section we will deal with a very simple example of a feed forward neural network. 
Be aware that the focus of this repopsitory is not to learn the theory behind deep learning, but to build on the concepts of design patterns. 

In General we will use the TF 2.X which has Keras married into it. 
There are two distinctive styles when coding a feed forward network in TF.Keras 

                    1. Sequential API Method 
                    2. Functional API Method 


Feed Forward  are like functions in procedural programming. The inputs are passed as parameters (i.e input layers ) , the function performs  a sequential set of actions based on the inputs (i.e hidden layers ) and outputs as a result 
(i.e Output layers)

## Sequential API Method

This method is easier to read but obviously less flexible. Essentially you create a Sequential class object, and then add one layer at a time. An Example code goes as follows:

``` 
    from tensorflow.keras import Sequential

    model = Sequential()
    model.add(.....first layer ....)
    model.add(.... second layer....)
    model.add(....Output layer.... ) 
```

This can aslo be wrriten in an another way: You can pass in the layers as a list at the same time you instantiate the Sequential object: 

```
from tensorflow.keras import Sequential

model = Sequential([.....first layer.....,
                    .....second layer.....,
                    ..... output layer.....
                    ])
```

## Functional API Method 
As comparred to the sequential API the finctional API is differnet, it is more flexible and scalable. You first define the layers and then you tie all the layers to gether with a Model object:

```
from tensorflow.keras import Model
from tensorflow.keras.layers impory layers

input  = layers.(.....first layer.....)
hidden = layers.(.....hidden layer.....)( .... the layer to bind to ... in this case Input)
output = layers.(.....output layer.....)(..... the layer to bind to ... in this case hidden)

model = Model(input, output)
````
Using the functional api we can leverage the use of polymorphism i.e we instantiate a layer and that layer is a callable too. 





