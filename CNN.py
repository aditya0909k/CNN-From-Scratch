import numpy as np
from Convolution import Convolution
from FullyConnected import FullyConnected

class CNN:
    def __init__(self, layers, lr=0.001):
        self.layers = layers
        self.lr = lr

    def forward(self, feature, label):
        output = feature
        regularization_loss = 0

        for layer in self.layers:
            if isinstance(layer, Convolution) or isinstance(layer, FullyConnected): #if we have a layer in which we update parameters, add the reg loss
                regularization_loss += layer.regularization()
            output = layer.forward(output)

        epsilon = 1e-5
        output = np.clip(output, epsilon, 1-epsilon) #clip gradient so model learns better and no log(0)

        loss = -np.log(output[label]) + regularization_loss #take cross entropy loss with L2 regularization    
        accuracy = 1 if np.argmax(output) == label else 0

        return output, loss, accuracy
    
    def backwards(self, gradient, lr):
        for layer in reversed(self.layers):
            gradient = layer.backwards(gradient, lr) #backward pass and update gradient with new gradient each time
    
    def train(self, features, labels, epochs=100):
        for epoch in range(epochs):
            shuffle = np.random.permutation(len(features)) #shuffle data each time
            features = features[shuffle]
            labels = labels[shuffle]

            total_accuracy = 0
            total_loss = 0

            for i, feature in enumerate(features):
                label = labels[i]
                output, loss, accuracy = self.forward(feature, label)
                total_accuracy += accuracy
                total_loss += loss

                dL_dy = np.zeros(10)
                dL_dy[label] = -1 / output[label] #-1 * yi/pi --> yi will be 0 for all that isnt output[label]
                self.backwards(dL_dy, self.lr)

            if (epoch+1) % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs} | Loss: {total_loss / len(features):.2f} | Accuracy: {total_accuracy / len(features) * 100:.2f}%')
        print("-------------------------------------------")

    def test(self, features, labels):
        total_accuracy = 0
        for i, feature in enumerate(features):
            label = labels[i]
            _, _, accuracy = self.forward(feature, label)
            total_accuracy += accuracy

        return total_accuracy / len(features) * 100