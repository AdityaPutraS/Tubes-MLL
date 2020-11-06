from rnn import RNN
from dense import Dense
from sequential import Sequential
import numpy as np

def createModel(hidden_size, output_size, weight_initializer, timestep_count, feature):
	model = Sequential()
	model.add(RNN(hidden_size, return_sequences=True, weight_initializer=weight_initializer, input_shape=(timestep_count,feature)))
	model.add(Dense(output_size, time_distributed=True, weight_initializer=weight_initializer, activation='softmax'))
	return model

def printValue(model, output):
	h = model.layers[0].activation(model.layers[0].h[1:])
	print('====Hidden====')
	for i in range(len(h)):
		print('Timestep ke', i+1, h[i])
	print('====Output====')
	for i in range(len(output)):
		print('Timestep ke', i+1, output[i])

# Eksperimen 1
hidden_size = 3
output_size = 4
x = np.array([
		[1, 0, 0, 0],
		[0, 1, 0, 0],
		[0, 0, 1, 0]
	])
print('Eksperimen 1')
print('Hidden size: ', hidden_size)
print('Output size: ', output_size)
print('Sequence Length: ', x.shape[0])
print('Input size: ', x.shape)

model = createModel(hidden_size, output_size, 'zeros', x.shape[0], x.shape[1])
output = model.forward(x)
printValue(model, output)

# Eksperimen 2
hidden_size = 5
output_size = 5
x = np.array([
		[1, 0, 0],
		[0, 1, 0],
		[0, 0, 1],
		[0, 0, 1],
		[0, 0, 1]
	])
print('\n============================================')
print('Eksperimen 2')
print('Hidden size: ', hidden_size)
print('Output size: ', output_size)
print('Sequence Length: ', x.shape[0])
print('Input size: ', x.shape)
print('Input data:')
print(x)

model = createModel(hidden_size, output_size, 'ones', x.shape[0], x.shape[1])
output = model.forward(x)
printValue(model, output)

# Eksperimen 3
hidden_size = 4
output_size = 2
x = np.array([
		[1, 0, 0, 0.5],
		[0, 1, 0, 0.3],
		[0, 0, 1, 0.1],
		[0, 0, 1, 0.3],
		[0, 0, 1, 0.4]
	])
print('\n============================================')
print('Eksperimen 3')
print('Hidden size: ', hidden_size)
print('Output size: ', output_size)
print('Sequence Length: ', x.shape[0])
print('Input size: ', x.shape)
print('Input data:')
print(x)

model = createModel(hidden_size, output_size, 'random', x.shape[0], x.shape[1])
output = model.forward(x)
printValue(model, output)