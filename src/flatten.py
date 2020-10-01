import numpy as np
from activation import *

class Flatten:
	def __init__(self):
		self.input_shape = None
		self.output_shape = None
		self.activation = lambda x: x
		self.activation_deriv = lambda x: 1

	def updateInputShape(self, input_shape):
		self.input_shape = input_shape
		output_x = 1
		for length in input_shape:
			output_x = output_x * length
		self.output_shape = (output_x, 1)
    
	def getSaveData(self):
		data = {
			'name': 'Flatten',
			'input_shape': self.input_shape
		}
		return data

	def loadData(self, data):
		if('input_shape' not in data):
			raise KeyError("Data invalid")
		else:
			input_shape = data['input_shape']
			assert self.input_shape == input_shape

	def forward(self, mat):
		assert self.input_shape == mat.shape[1:]
		flattened_matrix = []
		for each_data in mat:
			flattened_matrix.append(np.ndarray.flatten(each_data))
		return np.array(flattened_matrix)

	def calcPrevDelta(self, neuron_input, delta, debug=False):
		return delta.reshape([-1] + list(self.input_shape))
	
	def backprop(self, neuron_input, delta, lr=0.001, debug=False):
		return np.zeros(0)

	def updateWeight(self, deltaWeight, debug=False):
		# Tidak memiliki weight
		pass
