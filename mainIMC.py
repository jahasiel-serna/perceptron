import numpy as np
import perceptron
import drawing

def generateData(len, xmin, xmax):
    x = np.zeros((len, 2))
    for i in range(len):
        x[i,0] = -xmin[0] + (xmax[0]+xmin[0]) * np.random.rand()
        x[i,1] = -xmin[1] + (xmax[1]+xmin[1]) * np.random.rand()
    return x

def classifyData(x, lim):
    y = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        imc = x[i,0] / x[i,1]**2
        if imc >= lim:
            y[i] = 1
        else:
            y[i] = 0
    return y

def normalizeData(x):
    x[:,0] = (x[:,0]-x[:,0].min()) / (x[:,0].max()-x[:,0].min())
    x[:,1] = (x[:,1]-x[:,1].min()) / (x[:,1].max()-x[:,1].min())
    return x

def main():
    default = input('Opciones por defecto? (s/n): ')
    
    xmin = np.array([40, 1])
    xmax = np.array([120, 2.2])
    len = 100
    lim = 25
    learning_rate = 0.1
    xylim = [0, 1]
    normalize = ''

    if default == 'n':
        len = int(input('Cantidad de datos a generar: '))
        xmin[0] = float(input('Menor peso(kg): '))
        xmax[0] = float(input('Mayor peso(kg): '))
        xmin[1] = float(input('Menor altura(m): '))
        xmax[1] = float(input('Mayor altura(m): '))
        lim = float(input('Limite de sobrepeso: '))
        learning_rate = float(input('Tasa de aprendizaje(0-1): '))
        xylim[0] = float(input('Limite inferior de la grafica: '))
        xylim[1] = float(input('Limite superior de la grafica: '))
        normalize = input('Normalizar datos? (s/n): ')

    x = generateData(len, xmin, xmax)
    y = classifyData(x, lim)
    if normalize != 'n':
        x = normalizeData(x)

    neurona = perceptron.Perceptron(x.shape[1], learning_rate)
    neurona.train(x, y)

    d = drawing.Draw(xlim=xylim, ylim=xylim)
    d.drawNeuron(neurona)
    d.drawData(x, y, '.')
    d.show()

if __name__ == "__main__":
    main()