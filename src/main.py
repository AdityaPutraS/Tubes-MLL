import numpy as numpy
from sequential import Sequential
from conv2d import Conv2D
from pooling2d import Pooling2D
from flatten import Flatten
from dense import Dense
from util import *

image_size = (100, 100)
np.random.seed(13517013)

# Load data
cats = readImage('../data/cats', image_size)
dogs = readImage('../data/dogs', image_size)
data = np.concatenate((cats, dogs))
np.random.shuffle(data)

# Rescale data
data = data/255

model = Sequential()
model.add(Conv2D(1, (2, 2), pad=1, stride=10, input_shape=(data.shape[1], data.shape[2], data.shape[3]), activation='relu'))
model.add(Pooling2D((2, 2), stride=1))
model.add(Flatten())
model.add(Dense(4, activation = 'relu'))
model.add(Dense(2, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

pred = model.forward(data)
print(pred)
print(pred.shape)