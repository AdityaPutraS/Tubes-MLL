import numpy as np
from activation import *

class Flatten:
	def __init__(self):
		self.input_shape = []
		self.output_shape = []
		self.activation_deriv = sigmoid_deriv

	def updateInputShape(self, input_shape):
		self.input_shape = input_shape
		output_x = 1
		for length in input_shape:
			output_x = output_x * length
		self.output_shape = (output_x, 1)
    
	def getSaveData(self):
		data = {'input_shape' : self.input_shape}
		return data

	def loadData(self, data):
		if('input_shape' not in data):
			raise KeyError("Data invalid")
		else:
			input_shape = data['input_shape']
			assert self.input_shape == input_shape

	def forward(self, mat):
		assert self.input_shape == np.shape(mat)
		flattened_matrix = np.ndarray.flatten(mat)
		return (flattened_matrix, flattened_matrix)

	def calcPrevDelta(self, neuron_input, delta, debug=False):
		return delta
	
	def backprop(self, neuron_input, delta, lr=0.001, debug=False):
		pass

	def updateWeight(self, deltaWeight, debug=False):
		pass