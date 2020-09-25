import numpy as np

class Perceptron:
    def __init__(self, dim, lrate):
        self.w = -1 + 2 * np.random.rand(dim)
        self.b = -1 + 2 * np.random.rand()
        self.eta = lrate

    def predict(self, x):
        if np.dot(self.w, x) + self.b >= 0:
            return 1
        return 0

    def train(self, x, y, epochs=100):
        p = x.shape[0]
        for _ in range(epochs):
            for i in range(p):
                y_est = self.predict(x[i])
                self.w += self.eta * (y[i] - y_est) * x[i]
                self.b += self.eta * (y[i] - y_est)