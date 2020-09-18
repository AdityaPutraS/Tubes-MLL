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
      temp = l.forward(temp)[1].copy()
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
        else:
          raise TypeError("Unknown layer")


  def fit(self, xData, yData, lr=0.001, epochs=1, debug=False, batch_size=1):
    # Training model dengan xData agar bisa menghasilkan yData
    # Training dilakukan dengan konsep mini_batch, update weight dilakukan setiap {batch_size}
    listErr = []
    for e in range(epochs):
      noData = 1
      numIter = int(np.ceil(xData.shape[0] / batch_size))
      # Mini batch
      for iter in range(numIter):
        deltaWeight = []
        # Hitung start dan end dari minibatch iterasi ke {iter}
        start = iter * batch_size
        end = start + batch_size
        # Untuk setiap data dalam minibatch, hitung deltaWeightnya
        for data in zip(xData[start:end], yData[start:end]):
          if(debug):
            print('Data ke-', noData)
            noData += 1
          x = np.array([data[0]])
          y = np.array([data[1]])
          temp = x.copy()

          # InputListNotActivated berisi sekumpulan output dari tiap layer (input dari layer selanjutnya) 
          #   tapi belum melewati fungsi aktivasi
          # InputListActivated berisi mirip seperti inputListNotActivated tapi sudah melewati aktivasi
          # InputListNotActivated berguna untuk menghitung deltaError tiap layer
          # InputListActivated berguna untuk melakukan backpropagation
          inputListNotActivated = [x]
          inputListActivated = [x]
          # Lakukan feed-forward untuk data {data}
          for l in self.layers:
            out = l.forward(temp)
            temp = out[1].copy()
            inputListNotActivated.append(out[0].copy())
            inputListActivated.append(out[1].copy())

          if(debug):
            print('InputList (~Activated) : ', inputListNotActivated)
            print('InputList (Activated) : ', inputListActivated)
          # Hitung delta output layer
          if(debug):
            print('activation_deriv(',inputListNotActivated[-1],') * (',y, ' - ', temp)
          delta = self.layers[-1].activation_deriv(inputListNotActivated[-1]) * (y - temp)
          delta = delta.ravel()
          listDelta = [delta]
          if(debug):
            print('deltaOutput : ', delta)
          # Hitung delta setiap layer menggunakan layer setelahnya, mulai dari layer terakhir
          for i in range(len(self.layers)-1, -1, -1):
            if(debug):
              print('Update weight layer ', i)
            prev_delta = self.layers[i].calcPrevDelta(inputListNotActivated[i], delta, debug=debug)
            listDelta.append(delta)
            delta = prev_delta
          # Lakukan backpropagation untuk setiap layer, mulai dari layer terakhir
          tempDeltaWeight = []
          for i in range(len(self.layers)-1, -1, -1):
            tempDeltaWeight.append(self.layers[i].backprop(inputListActivated[i], listDelta[-1-i], lr, debug=debug))
          if(debug):
            print('tempDeltaWeight : ')
            print(tempDeltaWeight)
          # Simpan deltaWeight ke dalam list terlebih dahulu
          deltaWeight.append(tempDeltaWeight)
          if(debug):
            print('List Delta :')
            print(listDelta)
        # End for mini batch

        # Ide : Hitung jumlah delta weight tiap layer, terus update weight tiap layer dengan rata rata deltaWeight layer itu
        if(debug):
          print(deltaWeight)
        # Hitung total
        totalDeltaWeight = deltaWeight[0]
        for i in range(len(self.layers)):
          for j in range(1, len(deltaWeight)):
            totalDeltaWeight[i] += deltaWeight[j][i]
        
        # Update weight
        for idx, layer in enumerate(self.layers):
          layer.updateWeight(totalDeltaWeight[-idx-1] / (end-start))
      # End for iterasi semua minibatch
      # Hitung error untuk epoch ini
      epochErr = self.calculateError(x, y)
      # Simpan error dalam list agar bisa di plot nantinya
      listErr.append(epochErr)
      if (e%100==99):
        print('Epoch ', e+1, '= error : ', epochErr)
    return listErr