import numpy as np
import matplotlib.pyplot as plt

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

def drawNeuron(model):
    w1, w2, b = model.w[0], model.w[1], model.b
    X = [-2, 2]
    Y = [(1/w2)*(-w1*X[0]-b), (1/w2)*(-w1*X[1]-b)]
    plt.plot(X,Y, '--k')

def drawData(x, y):
    for i in range(y.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'or')
        else:
            plt.plot(x[i][0], x[i][1], 'ob')

x = np.array([[0,0],[0,1],[1,0],[1,1]])
_and = np.array([0,0,0,1])
_or = np.array([0,1,1,1])
_xor = np.array([0,1,1,0])
cmps = {'and':_and, 'or':_or, 'xor':_xor}

learning_rate = float(input('Learning rate(0-1): '))
y = input("Compuerta(and, or, xor): ")

neurona = Perceptron(x.shape[1], learning_rate)

neurona.train(x, cmps[y])

drawNeuron(neurona)
drawData(x, cmps[y])
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.show()