import numpy as np
from activation import *
from util import conv2d

class Conv2D:
  def __init__(self, num_filter, pad, stride, input_shape=(1, 1, 3), activation='relu'):
    self.kernel = np.random.random((num_filter, input_shape[2], w_filter, h_filter)) # w_filter and h_filter TBD @adityaputras
    self.pad = pad
    self.stride = stride

    if (activation == 'relu'):
      self.activation = relu # detector part
      self.activation_deriv = relu_deriv

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