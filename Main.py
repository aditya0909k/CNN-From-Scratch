import tensorflow as tf
import numpy as np
from CNN import CNN
from Convolution import Convolution
from FullyConnected import FullyConnected
from Pooling import Pooling
from ReLU import ReLU
from Softmax import Softmax

class Main:
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train, X_test = X_train / 255.0, X_test / 255.0 #normalize

    X_train = np.expand_dims(X_train, axis=-1) #28x28 --> 28x28x1
    X_test = np.expand_dims(X_test, axis=-1) 

    X_train, y_train = X_train[:1000], y_train[:1000]
    X_test, y_test = X_test[:1000], y_test[:1000]


    ''' Let's try 3 different CNN architectures '''

    single_layer_CNN = CNN([Convolution(num_filters=3, filter_size=5, depth=1, reg=0.005), #24x24x3
                            ReLU(),
                            Pooling(2), #12x12x3
                            FullyConnected(input_size=432, output_size=10), #432x10
                            Softmax()], lr=0.005) 
    

    no_activation_CNN = CNN([Convolution(num_filters=3, filter_size=5, depth=1), #24x24x3
                            Pooling(2), #12x12x3
                            Convolution(num_filters=3, filter_size=3, depth=3), #10x10x3
                            Pooling(2), #5x5x3
                            FullyConnected(input_size=75, output_size=10), #75x10
                            Softmax()], lr=0.001)
    
    deep_CNN = CNN([Convolution(num_filters=10, filter_size=7, depth=1, reg=0.0001), #22x22x10
                    ReLU(),
                    Pooling(2), #11x11x10
                    Convolution(num_filters=10, filter_size=5, depth=10, reg=0.0001), #7x7x10
                    ReLU(),
                    Convolution(num_filters=15, filter_size=3, depth=10), #5x5x15
                    ReLU(),
                    FullyConnected(input_size=375, output_size=10, reg=0.0001), #375x10
                    Softmax()], lr=0.002) 

    ''' Train model'''
    print("Now training: Single Layer CNN\n")
    single_layer_CNN.train(X_train, y_train, epochs=50)
    print("Now training: No Activiation Function CNN\n")
    no_activation_CNN.train(X_train, y_train, epochs=50)
    print("Now training: Deep CNN\n")
    deep_CNN.train(X_train, y_train, epochs=50)

    ''' Test model '''
    single_acc = single_layer_CNN.test(X_test, y_test)
    noa_acc = no_activation_CNN.test(X_test, y_test)
    deep_acc = deep_CNN.test(X_test, y_test)

    print(f"Barebone CNN Test Accuracy: {single_acc:.2f}%")
    print(f"No Activation CNN Test Accuracy: {noa_acc:.2f}%")
    print(f"Deep CNN Test Accuracy: {deep_acc:.2f}%")


Main()