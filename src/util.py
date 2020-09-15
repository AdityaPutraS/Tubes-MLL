import numpy as np
# Utility
def conv2d(mat, kernel, pad, stride):
    padded_mat = np.pad(mat, pad)
    padded_mat_x, padded_mat_y = padded_mat.shape
    kernel_x, kernel_y = kernel.shape
    output_shape = ((padded_mat_x - kernel_x) // stride + 1, (padded_mat_y - kernel_y) // stride + 1)
    output = np.zeros(output_shape)
    for i in range(0, output_shape[0]):
      start_x = i*stride
      end_x = start_x + kernel_x
      for j in range(0, output_shape[1]):
        start_y = j*stride
        end_y = start_y + kernel_y 
        output[i, j] = np.tensordot(padded_mat[start_x:end_x, start_y:end_y], kernel)
    return output  
