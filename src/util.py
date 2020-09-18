import numpy as np
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
    for filter in range(num_filter):
      for i in range(output_shape[0]):
        start_x = i*stride
        end_x = start_x + kernel_x
        for j in range(output_shape[1]):
          start_y = j*stride
          end_y = start_y + kernel_y 
          for chan in range(padded_mat_c):
            output[i, j, filter] += np.tensordot(padded_mat[start_x:end_x, start_y:end_y, chan], kernel[filter, min(chan, kernel_c-1), :, :]) 
    return output    
