import numpy as np

class Pooling:
    def __init__(self, pool_size):
        self.pool_size = pool_size

    def forward(self, input):
        self.last_input = input
        in_h, in_w , in_d = input.shape
        out_h, out_w = in_h // self.pool_size, in_w // self.pool_size
        output = np.zeros((out_h, out_w, in_d))

        for i in range(out_h):
            for j in range(out_w):
                for k in range(in_d):
                    region = input[(i*self.pool_size):(i*self.pool_size + self.pool_size), (j*self.pool_size):(j*self.pool_size + self.pool_size), k] #take region of pool for current channel
                    output[i, j, k] = np.max(region) #maxpool this region

        return output
    
    def backwards(self, gradient, lr):
        out_h, out_w = gradient.shape[:2]
        dL_dx = np.zeros(self.last_input.shape)

        for i in range(out_h):
            for j in range(out_w):
                for k in range(self.last_input.shape[2]):
                    region = self.last_input[(i*self.pool_size):(i*self.pool_size + self.pool_size), (j*self.pool_size):(j*self.pool_size + self.pool_size), k] #get the current region we are looking at
                    if region.size == 0:
                        continue
                    index = np.unravel_index(np.argmax(region), region.shape) #assign index (i, j) as the location in the region with the maximum value
                    dL_dx[(i*self.pool_size + index[0]), (j*self.pool_size + index[1]), k] = gradient[i, j, k] #set this location with the incoming gradient at this location, keep others as 0
        return dL_dx