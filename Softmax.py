import numpy as np

class Softmax:
    def forward(self, input):
        self.last_input = input
        exp = np.exp(input)
        output = exp / np.sum(exp) #softmax equation
        return output

    def backwards(self, gradient, lr):
        transformation = np.exp(self.last_input)
        s_total = np.sum(transformation)
        softmax_output = transformation / s_total #recompute softmax on the 10x1 input vector

        jacobian = np.zeros((len(self.last_input), len(self.last_input))) #jacobian matrix for softmax loss

        for i in range(len(self.last_input)):
            for j in range(len(self.last_input)):
                if i == j:
                    jacobian[i][j] = softmax_output[i] * (1 - softmax_output[i]) #equation if logits are same
                else:
                    jacobian[i][j] = -softmax_output[i] * softmax_output[j] #equation if not same

        dL_dz = np.dot(jacobian, gradient)
        return dL_dz
    