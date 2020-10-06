# Main program for CNN
# IF4074 Pembelajaran Mesin Lanjut

import os
import cv2
import numpy as np

# Model Type
from sequential import Sequential

# Model Layers
from conv2d import Conv2D
from pooling2d import Pooling2D
from flatten import Flatten
from dense import Dense

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

if __name__ == "__main__":
    np.random.seed(13517013)
    image_size = (100, 100)

    def readImage(path):
        result = []
        images = os.listdir(path)
        for image in images:
            result.append(cv2.resize(cv2.imread(path + '/' + image, 1), image_size))
        return np.array(result)

    cats = readImage("../data/cats")
    dogs = readImage("../data/dogs")

    data = np.concatenate((cats, dogs))
    yData = np.array([0] * len(cats) + [1] * len(dogs))

    # scale data / 255
    data = data / 255

    x_train, x_test, y_train, y_test = train_test_split(data, yData, test_size=0.2, random_state=13517013)

    # ML Model
    model = Sequential()

    model.add(Conv2D(1, (2, 2), pad=1, stride=10, input_shape=(data.shape[1], data.shape[2], data.shape[3]), activation='relu'))
    model.add(Pooling2D((2, 2), stride=1))
    model.add(Flatten())
    model.add(Dense(4, activation='relu'))
    model.add(Dense(2, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    pred = model.forward(data)
    print(pred)

    # model.fit(x_train, y_train, epochs=10, lr=0.1, batch_size=5)
    # pred = model.forward(x_test)
    # print(classification_report(y_test, np.round(pred)))
