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

  def calcPrevDelta(self, neuron_input, delta, debug=False):
    return delta

  def forward(self, feature_maps):
    # assert self.input_shape == feature_maps.shape[1:]
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
  print("test 2D feature maps shape:", test_fmaps.shape)
  pooling_layer = Pooling2D(pool_shape=(2,2), stride=2, pool_mode='max')
  result = pooling_layer.forward(test_fmaps)
  print(result)
  print("result shape:", result.shape)

  """[[[ 6  8]
       [14 16]]

      [[12 16]
       [28 32]]]
     result shape: (2, 2, 2)"""

  test_3d_fmaps = np.array([
    [
      [
        [19, 66, 10],
        [80, 99, 10],
        [155, 148, 255]
      ], [
        [198, 254, 12],
        [88, 10, 12],
        [90, 19, 255]
      ], [
        [8, 1, 13],
        [113, 3, 13],
        [19, 1, 255]
      ]
    ], [
      [
        [1, 10, 14],
        [32, 31, 14],
        [10, 0, 123]
      ], [
        [9, 9, 15],
        [10, 15, 50],
        [99, 112, 0]
      ], [
        [10, 1, 12],
        [11, 11, 12],
        [19, 100, 111]
      ]
    ]
  ])

  print("\ntest 3D feature maps shape:", test_3d_fmaps.shape)
  pooling_3d_layer = Pooling2D(pool_shape=(2,2), stride=1, pool_mode='avg')
  result = pooling_3d_layer.forward(test_3d_fmaps)
  print(result)
  print("result shape:", result.shape)

  """[[[[ 96.25 107.25  11.  ]
        [103.25  69.   133.  ]]

        [[101.75  67.    12.5 ]
        [ 77.5    8.25 133.75]]]


      [[[ 13.    16.25  23.25]
        [ 37.75  39.5   46.75]]

        [[ 10.     9.    22.25]
        [ 34.75  59.5   43.25]]]]
      result shape: (2, 2, 2, 3)"""
