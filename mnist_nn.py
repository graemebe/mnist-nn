# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 15:26:53 2018

@author: graeme
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

print("Using tensorflow: " + tf.__version__)

from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

network.fit(train_images, train_labels, epochs=6, batch_size=128)

test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)

# Now test a prediction on a digit using training set

sample_digit_index = 10001
digit_img = train_images[sample_digit_index] # 1001 is a seven
print(digit_img.shape)
plt.imshow(digit_img.reshape((28,28)), cmap=plt.cm.binary)
plt.show()

sample_digit_label = np.argmax(train_labels[sample_digit_index], axis=0)
print("Sample digit is labelled as: " + str(sample_digit_label))

k = np.array(digit_img).reshape(1, 28*28)
prediction = np.argmax(network.predict(k), axis=1) 
print("Predicted digit: " + str(prediction))
if prediction == sample_digit_label:
    print("Correctly predicted digit!")
else:
    print("Error in prediction")


# Now test a prediction on a digit using test set

sample_digit_index = 4501
digit_img = test_images[sample_digit_index] # 1001 is a seven
print(digit_img.shape)
plt.imshow(digit_img.reshape((28,28)), cmap=plt.cm.binary)
plt.show()

sample_digit_label = np.argmax(test_labels[sample_digit_index], axis=0)
print("Sample digit is labelled as: " + str(sample_digit_label))

k = np.array(digit_img).reshape(1, 28*28)
prediction = np.argmax(network.predict(k), axis=1) 
print("Predicted digit: " + str(prediction))
if prediction == sample_digit_label:
    print("Correctly predicted digit!")
else:
    print("Error in prediction")






