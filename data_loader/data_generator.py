import numpy as np

class DataGenerator:
    def __init__(self, config):
        self.config = config
        x = np.random.random((500,1))*10

        # load data here
        self.input = x
        self.y = pow(x,3)

    def next_batch(self, batch_size):
        idx = np.random.choice(500, batch_size)
        yield self.input[idx], self.y[idx]
