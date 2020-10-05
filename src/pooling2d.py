import numpy as np
from util import pooling2d

class Pooling2D(object):
  def __init__(self, pool_shape, stride, padding = 0, pool_mode = 'max'):
    self.pool_shape = pool_shape
    self.stride = stride
    self.padding = padding
    self.pool_mode = pool_mode
    self.activation = lambda x: x
    self.activation_deriv = lambda x: 1
    self.input_shape = None
    self.output_shape = None

    self.backward_delta = {
      'max': self.maximum_backward_delta,
      'avg': self.average_backward_delta
    }

    self.updateWBO()

  def updateInputShape(self, input_shape):
    self.input_shape = input_shape
    self.updateWBO()

  def updateWBO(self):
    if (self.input_shape != None):
      self.output_shape = (((self.input_shape[0] + 2*self.padding - self.pool_shape[0])) // self.stride + 1,
                           ((self.input_shape[1] + 2*self.padding - self.pool_shape[1])) // self.stride + 1,
                           (self.input_shape[-1]))

  def getSaveData(self):
    data = {
      'name': 'Pooling2D',
      'input_shape' : self.input_shape,
      'pool_shape': self.pool_shape,
      'stride': self.stride,
      'padding': self.padding,
      'pool_mode': self.pool_mode
      }

    return data

  def loadData(self, data):
    pass # TBD

  def forward(self, feature_maps):
    # assert self.input_shape == feature_maps.shape[1:]
    result = []
    for fmap in feature_maps:
      result.append(pooling2d(fmap, self.pool_shape, self.stride, self.padding, self.pool_mode))

    result = np.array(result)
    self.output_shape = result.shape
    return result

  def average_backward_delta(self, neuron_input, delta, current_element, dx):
    each_batch, each_row, each_col, each_channel = current_element
    
    temp_pool = neuron_input[
      each_batch,
      (each_row * self.stride):(each_row * self.stride + self.pool_shape[0]),
      (each_col * self.stride):(each_col * self.stride + self.pool_shape[1]),
      each_channel
    ]

    # average = delta divided by input shape (width and height)
    average_delta = delta[each_batch, each_row, each_col, each_channel] / temp_pool.shape[0] / temp_pool.shape[1]

    dx[
      each_batch,
      (each_row * self.stride):(each_row * self.stride + self.pool_shape[0]),
      (each_col * self.stride):(each_col * self.stride + self.pool_shape[1]),
      each_channel
    ] += np.ones((temp_pool.shape[0], temp_pool.shape[1])) * average_delta
    return dx

  def maximum_backward_delta(self, neuron_input, delta, current_element, dx):
    each_batch, each_row, each_col, each_channel = current_element

    temp_pool = neuron_input[
      each_batch,
      (each_row * self.stride):(each_row * self.stride + self.pool_shape[0]),
      (each_col * self.stride):(each_col * self.stride + self.pool_shape[1]),
      each_channel
    ]
    # Mask True if element in pool is the max of the pool, else False
    masking = (temp_pool == np.max(temp_pool))
    dx[
      each_batch,
      (each_row * self.stride):(each_row * self.stride + self.pool_shape[0]),
      (each_col * self.stride):(each_col * self.stride + self.pool_shape[1]),
      each_channel
    ] += masking * delta[each_batch, each_row, each_col, each_channel]

    return dx

  def calcPrevDelta(self, neuron_input, delta, debug=False):
    dx = np.zeros(neuron_input.shape)

    for each_batch in range(delta.shape[0]):
      for each_row in range(delta.shape[1]):
        for each_col in range(delta.shape[2]):
          for each_channel in range(delta.shape[3]):
            # store each range variable to a variable, passing it easier to backward delta function
            current_element = [each_batch, each_row, each_col, each_channel]
            dx = self.backward_delta[self.pool_mode](neuron_input, delta, current_element, dx)

    return dx

  def backprop(self, neuron_input, delta, debug=False):
    # no weight to update, only pass the error to previous layer
    pass

  def updateWeight(self, deltaWeight, debug=False):
    # no weight to update, only pass the error to previous layer
    pass

if __name__ == "__main__":
  np.random.seed(1)

  feature_maps = np.random.randint(1, 3, (2, 2, 2, 3))
  print("feature_maps shape:", feature_maps.shape)
  print("feature_maps:\n", feature_maps)

  pool = Pooling2D((2, 2), 2, pool_mode='avg')
  result = pool.forward(feature_maps)
  print("\n\nresult shape:", result.shape)
  print("result:\n", result)

  delta = np.random.random(result.shape)
  print("\n\ndelta shape:", delta.shape)
  print("delta:\n", delta)

  dx = pool.calcPrevDelta(feature_maps, delta)
  print("\n\ndx shape:", dx.shape)
  print("dx:\n", dx)
