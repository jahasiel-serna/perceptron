import numpy as np
import perceptron
import drawing

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
_and = np.array([0, 0, 0, 1])
_or = np.array([0, 1, 1, 1])
_xor = np.array([0, 1, 1, 0])
cmps = {'and':_and, 'or':_or, 'xor':_xor}


def main():
    learning_rate = float(input('Learning rate(0-1): '))
    y = input("Compuerta(and, or, xor): ")

    neurona = perceptron.Perceptron(x.shape[1], learning_rate)
    neurona.train(x, cmps[y])

    d = drawing.Draw([-2, 2], [-2, 2], y.upper())
    d.drawNeuron(neurona)
    d.drawData(x, cmps[y], 'o')
    d.show()

if __name__ == "__main__":
    main()