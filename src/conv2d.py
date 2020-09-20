import numpy as np
from activation import *
from util import conv2d

class Conv2D:
  def __init__(self, num_filter, kernel_shape, pad, stride, input_shape=(1, 1, 3), activation='relu',):
    if (len(kernel_shape) != 2):
      raise ValueError("Kernel shape must be in 2 Dimension")
    self.kernel = np.random.random((num_filter, input_shape[2], kernel_shape[0], kernel_shape[1]))
    self.pad = pad
    self.stride = stride
    self.input_shape = input_shape
    self.output_shape = None

    if (activation == 'relu'):
      self.activation = relu # detector part
      self.activation_deriv = relu_deriv
    else:
      raise ValueError("Activation function " + activation + " does not exist.")

  def updateInputShape(self, input_shape):
    self.input_shape = input_shape
    self.updateWBO()

  def updateWBO(self):
    pass

  def getSaveData(self):
    data = {
      'name': 'Conv2D',
      'kernel': self.kernel,
      'pad': self.pad,
      'stride': self.stride
    }

    return data

  def loadData(self, data):
    pass

  def forward(self, x):
    result = []

    for feature_map in x:
      result.append(conv2d(feature_map, self.kernel, self.pad, self.stride))

    return np.array(result)

  def calcPrevDelta(self, neuron_input, delta, debug=False):
    return delta

  def backprop(self, neuron_input, delta, lr=0.001, debug=False):
    pass

  def updateWeight(self, deltaWeight, debug=False):
    pass