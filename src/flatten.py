import numpy as np

class Flatten:
	def __init__(self):
		self.input_shape = []
		self.output_shape = []
	
	def updateInputShape(self, input_shape):
		self.input_shape = input_shape
		output_x = 1
		for length in input_shape:
			output_x = output_x * length
		self.output_shape = (output_x, 1)

	def forward(self, mat):
		assert self.input_shape == np.shape(mat)
		flattened_matrix = np.ndarray.flatten(mat)
		return flattened_matrix
