import numpy as np
import json
from dense import *
from flatten import *

class Sequential:
  # Representasi suatu model dimana perhitungan dilakukan sekuensial (layer per layer)
  def __init__(self):
    self.layers=[]
    self.output_shape = None

  def add(self, layer):
    # Tambah layer baru ke model
    if(len(self.layers) == 0 and layer.input_shape == None):
      raise ValueError("Input shape cannot be null")
    else:
      if(self.output_shape == None):
        self.output_shape = layer.output_shape
      else:
        layer.updateInputShape(self.output_shape)
        self.output_shape = layer.output_shape
      self.layers.append(layer)

  def forward(self, x):
    # Lakukan feed-forward
    temp = x.copy()
    for l in self.layers:
      temp = l.activation(l.forward(temp))
    return temp

  def calculateError(self, x, y):
    # Menghitung Mean Squared Error dari y dan hasil feed forward x
    temp = self.forward(x)
    return 0.5 * np.mean(np.square(y - temp))

  def saveModel(self, path):
    data = {'layers' : []}
    for layer in self.layers:
      data['layers'].append(layer.getSaveData())
    with open(path, 'w') as outfile:
      json.dump(data, outfile)

  def loadModel(self, path):
    with open(path) as json_file:
      data = json.load(json_file)
      for layer in data['layers']:
        if(layer['name'] == 'Dense'):
          self.add(Dense(layer['unit'], input_shape=layer['input_shape'], activation=layer['activation']))
          self.layers[-1].loadData(layer['data'])
        elif(layer['name'] == 'Flatten'):
          self.add(Flatten())
          self.layers[-1].loadData(layer['data'])
        else:
          raise TypeError("Unknown layer")

  def calcDelta(self, rawInput, yPred, yTarget, debug=False):
    # Menghitung delta output layer
    lastLayer = self.layers[-1]
    lastX = rawInput[-1]
    delta = lastLayer.activation_deriv(lastX) * (yTarget - yPred)
    delta = delta.ravel()
    listDelta = [delta]

    # Hitung delta setiap layer menggunakan layer setelahnya, mulai dari layer terakhir
    for i in range(len(self.layers)-1, -1, -1):
      prev_delta = self.layers[i].calcPrevDelta(rawInput[i], delta, debug=debug)
      listDelta.append(delta)
      delta = prev_delta

    return listDelta

  # Fit n epoch, n batch
  def fit(self, xData, yData, lr=0.001, epochs=1, batch_size=1, debug=False):
    listErr = []
    for e in range(epochs):
      self._fit_1_epoch(xData, yData, lr=lr, debug=debug, batch_size=batch_size)

      # Hitung error untuk epoch ini
      epochErr = self.calculateError(xData, yData)

      # Simpan error dalam list agar bisa di plot nantinya
      listErr.append(epochErr)

      # Print informasi epoch dan error
      # if (e%100==99):
      print('Epoch ', e+1, '= error : ', epochErr)
    return listErr

  # Fit 1 epoch, n batch
  def _fit_1_epoch(self, xData, yData, lr=0.001, batch_size=1, debug=False):
    # Training dilakukan dengan konsep mini_batch, update weight dilakukan setiap {batch_size}
    numBatch = int(np.ceil(xData.shape[0] / batch_size))

    # Untuk setiap data dalam minibatch, hitung deltaWeightnya
    deltaWeight = []
    for iter in range(numBatch):
      start = iter * batch_size
      end = start + batch_size

      # print(start, end)
      # for data in zip(xData[start:end], yData[start:end]):
        # print(data[0].shape)
        # x = np.array([data[0]])
        # y = np.array([data[1]])
      x = xData[start:end]
      y = yData[start:end]
      deltaWeight.append(self._fit_1_batch(x, y, lr=lr, debug=debug))
      # End for mini batch

    # Ide : Hitung jumlah delta weight tiap layer, terus update weight tiap layer dengan rata rata deltaWeight layer itu
    # Hitung total
    totalDeltaWeight = deltaWeight[0]
    for i in range(len(self.layers)):
      for j in range(1, len(deltaWeight)):
        totalDeltaWeight[i] += deltaWeight[j][i]

    # Update weight
    for idx, layer in enumerate(self.layers):
      layer.updateWeight(totalDeltaWeight[-idx-1] / (end-start))

  # Fit 1 batch
  def _fit_1_batch(self, xData, yData, lr=0.001, debug=False):
    # rawInput berisi input setiap layer secara raw(belum di aktivasi layer sebelumnya)
    rawInput = [xData]

    # Lakukan feed-forward untuk data {data}
    temp = xData.copy()
    for l in self.layers:
      out = l.forward(temp)
      temp = l.activation(out)
      rawInput.append(out.copy())

    # Hitung delta setiap layer
    listDelta = self.calcDelta(rawInput, temp, yData, debug=debug)

    # Lakukan backpropagation untuk setiap layer, mulai dari layer terakhir
    deltaWeight = []
    for i in range(len(self.layers)-1, 0, -1):
      l = self.layers[i]
      deltaWeight.append(l.backprop(rawInput[i], listDelta[-1-i], lr, debug=debug))
    deltaWeight.append(self.layers[0].backprop(rawInput[0], listDelta[-1], lr, debug=debug))

    return deltaWeight
