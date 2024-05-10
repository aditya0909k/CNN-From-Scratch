import numpy as np

class Convolution:
    def __init__(self, filter_size, num_filters, depth, reg=0.0001):
        self.filter_size = filter_size
        self.num_filters = num_filters
        self.depth = depth
        self.reg = reg
        self.filters = np.random.randn(num_filters, filter_size, filter_size, depth) / (filter_size**2)
        self.biases = np.zeros(num_filters)

    def forward(self, input):
        self.last_input = input 
        in_h, in_w, _ = input.shape
        out_h, out_w = in_h - self.filter_size + 1, in_w - self.filter_size + 1
        output = np.zeros((out_h, out_w, self.num_filters))

        for i in range(out_h):
            for j in range(out_w):
                for k in range(self.num_filters):
                    region = input[i:(i+self.filter_size), j:(j+self.filter_size), :] #get region for all channels
                    cur_filters = self.filters[k] #take the current filter
                    output[i, j, k] = np.sum(region * cur_filters) + self.biases[k] #convolve

        return output

    def backwards(self, gradient, lr):
        out_h, out_w = gradient.shape[:2]
        dL_dw = np.zeros(self.filters.shape)
        dL_db = np.zeros(self.biases.shape)
        dL_dx = np.zeros(self.last_input.shape)

        for i in range(out_h):
            for j in range(out_w):
                for k in range(self.num_filters):
                    region = self.last_input[i:(i+self.filter_size), j:(j+self.filter_size), :]
                    dL_dw[k] += gradient[i, j, k] * region  #multiply vals by gradients, add to wk
                    dL_db[k] += gradient[i, j, k] #sum gradients to update bias bk
                    dL_dx[i:(i+self.filter_size), j:(j+self.filter_size), :] += gradient[i, j, k] * self.filters[k] #for each region along all channels, add filter * gradient at that point

        self.filters -= (lr * dL_dw) + (self.reg * self.filters)
        self.biases -= lr * dL_db
        return dL_dx

    def regularization(self):
        return ((self.reg / 2) * np.sum(self.filters**2))