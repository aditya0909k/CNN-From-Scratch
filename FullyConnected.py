import numpy as np

class FullyConnected:
    def __init__(self, input_size, output_size, reg=0.0001):
        self.reg = reg
        self.weights = np.random.randn(input_size, output_size) / input_size
        self.biases = np.zeros(output_size)

    def forward(self, input):
        self.last_input_shape = input.shape
        self.flat_input = input.flatten() #flatten tensor or a FC layer (.flatten() on 1d does nothing)
        output =  np.dot(self.flat_input, self.weights) + self.biases #dot product perceptron
        return output

    def backwards(self, gradient, lr):
        column_input = self.flat_input[:, np.newaxis] #reshape from (n, ) to (n, 1)
        vector_gradient = gradient[np.newaxis, :] #reshape from (, n) to (1, n)

        dL_dw = np.dot(column_input, vector_gradient) #gradient wrt weights is the input * prev gradient
        dL_db = gradient #bias gradient remains same
        dL_dx = np.dot(self.weights, gradient) #gradient wrt input is weights * prev gradient

        self.weights -= (lr * dL_dw) + (self.reg * self.weights)
        self.biases -= lr * dL_db
        return dL_dx.reshape(self.last_input_shape) #reshape to the previous layer

    def regularization(self):
        return ((self.reg / 2) * np.sum(self.weights**2))