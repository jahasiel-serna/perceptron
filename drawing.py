import matplotlib.pyplot as plt

class Draw:
    def __init__(self, xlim, ylim, title=''):
        plt.title(title)
        plt.xlim(xlim)
        plt.ylim(ylim)

    def drawNeuron(self, model):
        w1, w2, b = model.w[0], model.w[1], model.b
        X = [-2, 2]
        Y = [(1/w2)*(-w1*X[0]-b), (1/w2)*(-w1*X[1]-b)]
        plt.plot(X,Y, '--k')

    def drawData(self, x, y, fig):
        for i in range(y.shape[0]):
            if y[i] == 0:
                plt.plot(x[i][0], x[i][1], fig+'r')
            else:
                plt.plot(x[i][0], x[i][1], fig+'b')

    def show(self):
        plt.show()
