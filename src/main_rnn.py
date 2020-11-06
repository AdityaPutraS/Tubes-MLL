from rnn import RNN
from dense import Dense
from sequential import Sequential
import numpy as np


def createModel(hidden_size, output_size):
	model = Sequential()
	model.add(RNN(hidden_size, return_sequences=True, weight_initializer='zeros', input_shape=(4,4)))
	model.add(Dense(output_size, time_distributed=True, activation='softmax'))
	return model

# def printValue(i):

model = createModel(3, 4)
x = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0]])
# print(model.layers[0].forward(x))
print(model.forward(x))
