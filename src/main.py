from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sequential import *
from dense import *

## Gist originally developed by @craffel and improved by @ljhuang2017

import matplotlib.pyplot as plt
import numpy as np

def draw_neural_net(ax, left, right, bottom, top, layer_sizes):
    '''
    Draw a neural network cartoon using matplotilb.
    
    :usage:
        >>> fig = plt.figure(figsize=(12, 12))
        >>> draw_neural_net(fig.gca(), .1, .9, .1, .9, [4, 7, 2])
    
    :parameters:
        - ax : matplotlib.axes.AxesSubplot
            The axes on which to plot the cartoon (get e.g. by plt.gca())
        - left : float
            The center of the leftmost node(s) will be placed here
        - right : float
            The center of the rightmost node(s) will be placed here
        - bottom : float
            The center of the bottommost node(s) will be placed here
        - top : float
            The center of the topmost node(s) will be placed here
        - layer_sizes : list of int
            List of layer sizes, including input and output dimensionality
    '''
    n_layers = len(layer_sizes)
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)
    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in range(layer_size):
            circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/4.,
                                color='w', ec='k', zorder=4)
            ax.add_artist(circle)
    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                  [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='k')
                ax.add_artist(line)
								
iris = datasets.load_iris()
iris_x = iris.data
iris_y = iris.target.astype(np.float64)
# Map nilai iris_y ke 0-1
iris_y[iris_y==1] = 0.5
iris_y[iris_y==2] = 1
# Buat test data
x_train, x_test, y_train, y_test = train_test_split(iris_x, iris_y, test_size=0.33, random_state=1)

# Seed random numpy agar hasil experimen sama
np.random.seed(2)
model = Sequential()
model.loadModel('iris.json')
# model.add(Dense(4, input_shape=(4, )))
# model.add(Dense(3, activation='sigmoid'))
# model.add(Dense(2, activation='sigmoid'))
# model.add(Dense(1, activation='sigmoid'))
# history = model.fit(iris_x, iris_y.reshape(iris_y.shape[0], 1), epochs=2000, debug=False, lr=0.14, batch_size=30)
# plt.plot(history)

predIris = model.forward(np.reshape(iris_x, (150, 2, 2)))
# Hitung class tiap data
deltaClass0 = np.abs(predIris - 0).reshape(predIris.shape[0])
deltaClass1 = np.abs(predIris - 0.5).reshape(predIris.shape[0])
deltaClass2 = np.abs(predIris - 1).reshape(predIris.shape[0])
deltaClass = np.array([deltaClass0, deltaClass1, deltaClass2])
classPredIris = np.argmin(deltaClass, axis=0)
deltaPred = classPredIris - iris.target
jumlahSalah = np.sum(np.where(deltaPred == 0, 0, 1))
accuracy = (150 - jumlahSalah) / 150
print('Acc : ', accuracy)
print('Jumlah Salah : ', jumlahSalah)

print("Prediction : ")
print(classPredIris.astype('int'))
print("Real : ")
print(iris.target)

print("Saving model")
model.saveModel("iris2.json")

# layer_arr = []
# for layer in model.layers:
#   layer_arr.append(layer.units)

# print(len(layer_arr))

# for layar in layer_arr:
#   print(layar)

# fig = plt.figure(figsize=(12, 12))
# ax = fig.gca()
# ax.axis('off')
# draw_neural_net(ax, .1, .9, .1, .9, layer_arr)
# fig.show()