# Activation Functions and their derivatives
import numpy as np

# Sigmoid function
def sigmoid(x):
  return 1/(1+np.exp(-x))

def sigmoid_deriv(x):
  return sigmoid(x) * (1 - sigmoid(x))

# ReLU function
def relu(x):
  return np.maximum(0, x)

def relu_deriv(x):
  return np.heaviside(x, 0)

# Leaky ReLU function
def leaky_relu(x):
  return np.maximum(0.01*x, x)

def leaky_relu_deriv(x):
  return np.where(x > 0, 1, 0.01)

# TanH function
def tanh(x):
  return np.tanh(x)

def tanh_deriv(x):
  return (1 - np.square(np.tanh(x)))

# Softmax function
def softmax(x):
	ex = np.exp(x - np.max(x))
	return ex / ex.sum()

def softmax_deriv(x):
  return x

def softmax_time_distributed(x):
  result = []
  for xx in x:
    ex = np.exp(xx - np.max(xx))
    result.append(ex/ex.sum())
  return result

def softmax_time_distributed_deriv(x):
  return x