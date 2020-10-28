import numpy as np
from activation import *

class RNN(object):
  def __init__(self, units, hidden_size, input_shape=None, activation='tanh', return_sequences=False):
    self.units = units # output dim
    self.hidden_size = hidden_size
    self.input_shape = input_shape
    self.return_sequences = return_sequences
    self.activation_name = activation
    self.output_shape = None
    
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

    self.bias_xh = np.random.random((self.hidden_size,)) # bias for input to hidden
    self.bias_hy = np.random.random((self.units,)) # bias for hidden to output
    # 3 weights
    self.U = None
    self.V = None
    self.W = None

    if (self.input_shape != None):
      self.updateWBO()
  
  def updateInputShape(self, input_shape):
    self.input_shape = input_shape
    self.updateWBO()

  def updateWBO(self):
    sequence_length = self.input_shape[0]

    # Initialize 3 weight matrices (U, V, W)
    self.U = np.random.random((self.hidden_size, sequence_length))
    self.V = np.random.random((self.units, self.hidden_size))
    self.W = np.random.random((self.hidden_size, self.hidden_size))

    if (self.return_sequences == True):
      # output = returning full output sequence
      self.output_shape = (sequence_length, self.units)
    else:
      assert self.return_sequences == False, "Return Sequences value must be 'True' or 'False'."
      # output = returning last value of output sequence
      self.output_shape = (self.units)

  def getSaveData(self):
    return {
      "name": "RNN",
      "units": self.units,
      "return_sequences": self.return_sequences,
      "activation": self.activation_name,
      "data": {
        "bias_xh": self.bias_xh.tolist(),
        "bias_yh": self.bias_hy.tolist(),
        "U": self.U.tolist(),
        "V": self.V.tolist(),
        "W": self.W.tolist()
      }
    }

  def loadData(self, data):
    self.bias_xh = np.array(data['bias_xh'].copy())
    self.bias_hy = np.array(data['bias_hy'].copy())

    self.U = np.array(data['U'].copy())
    self.V = np.array(data['V'].copy())
    self.W = np.array(data['W'].copy())

  def forward(self, x_data):
    # TBD
    return x_data

  # Backward Propagation Methods
  def calcPrevDelta(self, neuron_input, delta, debug=False):
    pass

  def backprop(self, neuron_input, delta, lr=0.001, debug=False):
    pass

  def updateWeight(self, deltaWeight, deltaBias, debug=False):
    pass
