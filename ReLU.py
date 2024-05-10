import numpy as np

class ReLU:
    def forward(self, input):
        self.last_input = input
        output = np.maximum(0, input) #relu activation
        return output
    
    def backwards(self, gradient, lr):
        dL_dr = self.last_input > 0 #relu gradient, if x>0 = 1, otherwise 0
        dL_dx = dL_dr * gradient #only keep gradients for positive values
        return dL_dx