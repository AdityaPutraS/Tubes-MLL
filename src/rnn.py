import numpy as np
from activation import *


class RNN(object):
  def __init__(self, units, weight_initializer='random', input_shape=None, activation='tanh', return_sequences=False):
    self.units = units
    self.weight_initializer = weight_initializer
    self.input_shape = input_shape
    self.return_sequences = return_sequences
    self.activation_name = activation
    self.output_shape = None
    self.h = [np.zeros((self.units, ))]

    if (self.activation_name == 'relu'):
      self.activation = relu
      self.activation_deriv = relu_deriv
    elif (self.activation_name == 'leaky_relu'):
      self.activation = leaky_relu
      self.activation_deriv = leaky_relu_deriv
    elif (self.activation_name == 'tanh'):
      self.activation = tanh
      self.activation_deriv = tanh_deriv
    elif (self.activation_name == 'sigmoid'):
      self.activation = sigmoid
      self.activation_deriv = sigmoid_deriv
    else:
      raise ValueError("Activation function " + activation + " does not exist.")

    self.bias_xh = self.initWeight((self.units,)) # bias for input to hidden
    # 2 weights
    self.U = None
    self.W = None

    if (self.input_shape != None):
      self.updateWBO()


  def initWeight(self, size):
    if self.weight_initializer == 'random':
      return np.random.random(size)
    elif self.weight_initializer == 'zeros':
      return np.zeros(size)
    elif self.weight_initializer == 'ones':
      return np.ones(size)


  def updateInputShape(self, input_shape):
    self.input_shape = input_shape
    self.updateWBO()


  def updateWBO(self):
    sequence_length = self.input_shape[1]

    # Initialize 2 weight matrices (U, W)
    self.U = self.initWeight((self.units, sequence_length))
    self.W = self.initWeight((self.units, self.units))

    if (self.return_sequences == True):
      # output = returning full output sequence
      self.output_shape = (sequence_length, self.units)
    else:
      assert self.return_sequences == False, "Return Sequences value must be 'True' or 'False'."
      # output = returning last value of output sequence
      self.output_shape = (self.units, )


  def getSaveData(self):
    return {
      "name": "RNN",
      "units": self.units,
      "return_sequences": self.return_sequences,
      "activation": self.activation_name,
      "data": {
        "bias_xh": self.bias_xh.tolist(),
        "U": self.U.tolist(),
        "W": self.W.tolist()
      }
    }


  def loadData(self, data):
    self.bias_xh = np.array(data['bias_xh'].copy())

    self.U = np.array(data['U'].copy())
    self.W = np.array(data['W'].copy())


  def forward(self, x_data):
    # self.U = np.array([[0.1, 0.15, 0.2, 0.3],
    #                   [0.15, 0.2, 0.3, 0.1],
    #                   [0.2, 0.3, 0.1, 0.15]])
    # self.W = np.array([[0.5, 0.5, 0.5],
    #                   [0.5, 0.5, 0.5],
    #                   [0.5, 0.5, 0.5]])
    # self.bias_xh = np.array([0.1, 0.1, 0.1])

    # Init h0
    self.h = [np.zeros((self.units, ))]
    ht = self.h[0]

    for timestep in range(self.input_shape[0]):
      ux = self.U @ x_data[timestep]
      wh = self.W @ ht
      ht = ux + (wh + self.bias_xh)
      self.h.append(ht)
      ht = self.activation(ht)
    if (self.return_sequences == True):
      return self.h[1:]
    else:
      return self.h[-1]


  # Backward Propagation Methods
  def calcPrevDelta(self, neuron_input, delta, debug=False):
    pass


  def backprop(self, neuron_input, delta, lr=0.001, debug=False):
    pass


  def updateWeight(self, deltaWeight, deltaBias, debug=False):
    pass
