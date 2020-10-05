import numpy as np
import cv2
import os

# Utility

# mat = W * H * C
# kernel = num_filter * C_Kernel * W_Kernel * H_Kernel
# Output = (_ * _ * num_filter)
def conv2d(mat, kernel, pad, stride):
  padded_mat = np.pad(mat, pad)
  padded_mat_x, padded_mat_y, padded_mat_c = padded_mat.shape
  num_filter, kernel_c, kernel_x, kernel_y = kernel.shape
  output_shape = ((padded_mat_x - kernel_x) // stride + 1, (padded_mat_y - kernel_y) // stride + 1, num_filter)
  output = np.zeros(output_shape)
  for _filter in range(num_filter):
    for i in range(output_shape[0]):
      start_x = i*stride
      end_x = start_x + kernel_x
      for j in range(output_shape[1]):
        start_y = j*stride
        end_y = start_y + kernel_y 
        for chan in range(padded_mat_c):
          output[i, j, _filter] += np.tensordot(padded_mat[start_x:end_x, start_y:end_y, chan], kernel[_filter, min(chan, kernel_c-1), :, :]) 
  return output

def get_pooling_region(x, pool_shape, stride, output_shape):
  for i in range(output_shape[0]):
    for j in range(output_shape[1]):
      new_region = x[(i * stride):(i * stride + pool_shape[0]), (j * stride):(j * stride + pool_shape[1])]
      yield new_region, i, j

# Pooling function for 2d matrices
def one_channel_pooling(x_data, pool_shape, stride, padding, pool_mode = 'max'):
  # do this for each channel
  x = np.pad(x_data, padding , mode='constant')

  output_shape = (((x.shape[0] - pool_shape[0]) // stride) + 1,
                  ((x.shape[1] - pool_shape[1]) // stride) + 1)

  pool_output = np.zeros(output_shape)

  pooling_output_mode = {
    'max': np.amax,
    'avg': np.mean
  }

  for region, row, col in get_pooling_region(x, pool_shape, stride, output_shape):
    pool_output[row, col] = pooling_output_mode[pool_mode](region, axis=(0, 1))

  return pool_output

def pooling2d(x_data, pool_shape, stride, padding, pool_mode = 'max'):
  # pooling can be done on however many channel there is
  if (len(x_data.shape) == 2):
    # data consist of only single channel
    return one_channel_pooling(x_data, pool_shape, stride, padding, pool_mode)
  elif (len(x_data.shape) == 3):
    # data consist of n channels
    data = np.moveaxis(x_data, 2, 0) # change channels last to channels first formats

    pooling_output = []
    for data_channel in data:
      pooling_output.append(one_channel_pooling(data_channel, pool_shape, stride, padding, pool_mode))

    return np.moveaxis(np.array(pooling_output), 0, 2) # change channels first to channels last format

def readImage(path, image_size):
    result = []
    images = os.listdir(path)
    for image in images:
        result.append(cv2.resize(cv2.imread(path + '/' + image, 1), image_size))
    return np.array(result)
