import numpy as np
from activation import *
from util import conv2d

class Conv2D:
  def __init__(self, kernel, pad, stride):
    self.kernel = kernel
    self.pad = pad
    self.stride = stride
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