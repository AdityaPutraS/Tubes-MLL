import numpy as np
from util import pooling2d

class Pooling2D(object):
  def __init__(self, pool_shape, stride, padding = None, pool_mode = 'max'):
    self.pool_shape = pool_shape
    self.stride = stride
    self.padding = padding
    self.pool_mode = pool_mode
    self.activation = lambda x: x
    self.activation_deriv = lambda x: 1
    self.output_shape = None

  def updateInputShape(self, input_shape):
    self.input_shape = input_shape
    self.updateWBO()

  def updateWBO(self):
    # WBO = weight, bias, output
    pass

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

  def calcPrevDelta(self, neuron_input, delta, debug=False):
		return delta

  def forward(self, feature_maps):
    result = []
    for fmap in feature_maps:
      result.append(pooling2d(fmap, self.pool_shape, self.stride, self.padding, self.pool_mode))

    result = np.array(result)
    self.output_shape = result.shape
    return result

  def backprop(self, neuron_input, delta, lr=0.001, debug=False):
		pass

  def updateWeight(self, deltaWeight, debug=False):
		pass

if __name__ == "__main__":
  test_fmap = np.array(
      [[1,2,3,4],
       [5,6,7,8],
       [9,10,11,12],
       [13,14,15,16]]
  )

  test_fmaps = np.array([test_fmap, test_fmap*2])

  pooling_layer = Pooling2D(pool_shape=(2,2), stride=2, pool_mode='max')
  print(pooling_layer.forward(test_fmaps, (2, 2), 2))