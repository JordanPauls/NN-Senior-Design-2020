#!/usr/bin/env python
# coding: utf-8

# In[4]:


#LENET_SVD
from keras import backend
from keras import datasets
import keras
import numpy as np
from keras import models, layers
from keras.models import Model,Sequential, model_from_json
from keras.layers import Dense, Conv2D, AveragePooling2D, Flatten
from keras.datasets import mnist
from keras.utils import np_utils


# Load dataset as train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

img_rows, img_cols = x_train.shape[1:]

if backend.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_test = np.pad(x_test, ((0,0),(2,2),(2,2),(0,0)), 'constant') #pad the images
x_test = x_test.astype('float32') # Set numeric type to float32 from uint8
x_test /= 255 # Normalize value to [0, 1]
x_test = x_test.reshape(x_test.shape[0], 32,32,1) # Reshape the dataset into 4D array
y_test = np_utils.to_categorical(y_test, 10) # Transform lables to one-hot encoding


json_file = open('Lenet5_Files_Baseline/Lenet.json', 'r')
lenet_model_json = json_file.read()
json_file.close()
lenet_model = model_from_json(lenet_model_json)
lenet_model.load_weights("Lenet5_Files_Baseline/Lenet.h5")
lenet_model.compile(loss='categorical_crossentropy',optimizer='SGD',metrics=['accuracy'])

lenet_model.summary()

# Write the testing input and output variables
score = lenet_model.evaluate(x_test, y_test, verbose=0)
truncsvd_accuracy = score[1]
print('Accuracy: ', truncsvd_accuracy)

