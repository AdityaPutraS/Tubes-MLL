import numpy as np


# Activation Function & Turunannya
def sigmoid(x):
  return 1/(1+np.exp(-x))

def sigmoid_deriv(x):
  return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
  return np.maximum(0, x)

def relu_deriv(x):
  return np.heaviside(x, 0)

def leaky_relu(x):
  return np.maximum(0.01*x, x)

def leaky_relu_deriv(x):
  return np.where(x > 0, 1, 0.01)
