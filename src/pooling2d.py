import numpy as np

class Pooling2D(object):
  def __init__(self, pool_shape, stride, padding = None, pool_mode = 'max'):
    self.pool_shape = pool_shape
    self.stride = stride
    self.padding = padding
    self.pool_mode = pool_mode

  def forward(self, x_data):
    if self.padding != None:
      x = np.pad(x_data, self.padding, mode='constant')
    else:
      x = x_data

    output_shape = (((x.shape[0] - self.pool_shape[0]) // self.stride) + 1,
                    ((x.shape[1] - self.pool_shape[1]) // self.stride) + 1)
    
    pool_output = np.lib.stride_tricks.as_strided(
        x,
        shape = output_shape + self.pool_shape,
        strides = (self.stride * x.strides[0], self.stride * x.strides[1]) + x.strides
    )

    pool_output = pool_output.reshape(-1, *self.pool_shape)

    if self.pool_mode == 'max':
      return pool_output.max(axis=(1,2)).reshape(output_shape)
    elif self.pool_mode == 'avg':
      return pool_output.mean(axis=(1,2)).reshape(output_shape)
    
if __name__ == "__main__":
  test_mat = np.array(
      [[1,2,3,4],
       [5,6,7,8],
       [9,10,11,12],
       [13,14,15,16]]
  )

  pooling_layer = Pooling2D(pool_shape=(2,2), stride=2, pool_mode='max')
  print(pooling_layer.forward(test_mat, (2, 2), 2))