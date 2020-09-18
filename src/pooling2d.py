import numpy as np

def pooling_2d(x_data, pool_size, stride, padding = None, pool_mode = 'max'):
  if padding != None:
    x = np.pad(x_data, padding, mode='constant')
  else:
    x = x_data

  output_shape = (((x.shape[0] - pool_size[0]) // stride) + 1,
                  ((x.shape[1] - pool_size[1]) // stride) + 1)
  
  pool_output = np.lib.stride_tricks.as_strided(
      x,
      shape = output_shape + pool_size,
      strides = (stride * x.strides[0], stride * x.strides[1]) + x.strides
  )

  pool_output = pool_output.reshape(-1, *pool_size)

  if pool_mode == 'max':
    return pool_output.max(axis=(1,2)).reshape(output_shape)
  elif pool_mode == 'avg':
    return pool_output.mean(axis=(1,2)).reshape(output_shape)
    
if __name__ == "__main__":
  test_mat = np.array(
      [[1,2,3,4],
       [5,6,7,8],
       [9,10,11,12],
       [13,14,15,16]]
  )
  
  print(pooling_2d(test_mat, (2, 2), 2))