#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Baseline Inference Network
import time
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

x_train = np.pad(x_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')
x_test = np.pad(x_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')

# Set numeric type to float32 from uint8
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalize value to [0, 1]
x_train /= 255
x_test /= 255

# Transform lables to one-hot encoding
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

# Reshape the dataset into 4D array
x_train = x_train.reshape(x_train.shape[0], 32,32,1)
x_test = x_test.reshape(x_test.shape[0], 32,32,1) 

json_file = open('Lenet.json', 'r')
lenet_model_json = json_file.read()
json_file.close()
lenet_model = model_from_json(lenet_model_json)
lenet_model.load_weights("Lenet.h5")
lenet_model.compile(loss='categorical_crossentropy',optimizer='SGD',metrics=['accuracy'])

# Write the testing input and output variables
tic = time.perf_counter()
score = lenet_model.evaluate(x_test, y_test, verbose=0)
print("Baseline Realization time", time.perf_counter()-tic)
truncsvd_accuracy = score[1]
print('Baseline Accuracy ', truncsvd_accuracy)
lenet_model.summary()

print('\n')
print('\n')
print('\n')
#LENET_SVD

import time
from keras import backend
from keras import datasets
import keras
import numpy as np
from keras import models, layers
from keras.models import Model,Sequential, model_from_json
from keras.layers import Dense, Conv2D, AveragePooling2D, Flatten
from keras.datasets import mnist
from keras.utils import np_utils


fc_id = 5 # FC Layer Number
rank = 8

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

x_train = np.pad(x_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')
x_test = np.pad(x_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')

# Set numeric type to float32 from uint8
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalize value to [0, 1]
x_train /= 255
x_test /= 255

# Transform lables to one-hot encoding
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

# Reshape the dataset into 4D array
x_train = x_train.reshape(x_train.shape[0], 32,32,1)
x_test = x_test.reshape(x_test.shape[0], 32,32,1) 

#Retrieve Model
json_file = open('Lenet.json', 'r')
lenet_model_json = json_file.read()
json_file.close()
lenet_model = model_from_json(lenet_model_json)
lenet_model.load_weights("Lenet.h5")
lenet_model.compile(loss='categorical_crossentropy',optimizer='SGD',metrics=['accuracy'])

# Loading weights of the model
keep = rank
fc1 = lenet_model.layers[fc_id].get_weights()

# Decomposition and Reconstruction
U, S, V = np.linalg.svd(fc1[0], full_matrices=False)
tU, tS, tV = U[:, 0:keep], S[0:keep], V[0:keep, :]
fc1_t = np.matmul(np.matmul(tU, np.diag(tS)), tV)

# Loading weights for new model
fc1[0] = fc1_t
lenet_model.layers[fc_id].set_weights(fc1)

# Write the testing input and output variables
tic = time.perf_counter()
score = lenet_model.evaluate(x_test, y_test, verbose=0)
print("SVD Realization time", time.perf_counter()-tic)
truncsvd_accuracy = score[1]
print('SVD Truncated Accuracy ', truncsvd_accuracy)
lenet_model.summary()


print('\n')
print('\n')
print('\n')

#Compressed Lenet SVD
import time
from keras import backend
from keras import datasets
import keras
import numpy as np
from keras import models, layers
from keras.models import Model,Sequential, model_from_json
from keras.layers import Dense, Conv2D, AveragePooling2D, Flatten
from keras.datasets import mnist
from keras.utils import np_utils

fc_id = 5 # FC Layer Number
rank = 8 # truncate value

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

x_train = np.pad(x_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')
x_test = np.pad(x_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')

# Set numeric type to float32 from uint8
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalize value to [0, 1]
x_train /= 255
x_test /= 255

# Transform lables to one-hot encoding
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

# Reshape the dataset into 4D array
x_train = x_train.reshape(x_train.shape[0], 32,32,1)
x_test = x_test.reshape(x_test.shape[0], 32,32,1) 

json_file = open('Lenet.json', 'r')
lenet_model_json = json_file.read()
json_file.close()
lenet_model = model_from_json(lenet_model_json)
lenet_model.load_weights("Lenet.h5")
lenet_model.compile(loss='categorical_crossentropy',optimizer='SGD',metrics=['accuracy'])

# Loading weights of the model
keep = rank
fc1 = lenet_model.layers[fc_id].get_weights()
fc2 = lenet_model.layers[fc_id+1].get_weights()
lenet_model.pop()
lenet_model.pop()
lenet_model.add(layers.Dense(rank))
lenet_model.add(layers.Dense(120, activation = 'tanh'))
lenet_model.add(layers.Dense(10, activation = 'softmax'))

# Decomposition and Reconstruction

OU, OS, OV = np.linalg.svd(fc1[0], full_matrices=False)

tU, tS, tV = OU[:, 0:keep], OS[0:keep], OV[0:keep, :]
fc1_t = np.matmul(np.matmul(tU, np.diag(tS)), tV)

OS = np.diag(OS)
TU, TS, TV =  OU[:, 0:keep], OS[0:keep, 0:keep], OV[0:keep,:]

svdlayer2 = np.matmul(TS, TV)
svdlayer1 = TU

lenet_model.layers[fc_id].set_weights([svdlayer1,np.zeros(rank)])
lenet_model.layers[fc_id+1].set_weights([svdlayer2,fc1[1]]) #pass both our weights matrix and original bias vector
lenet_model.layers[fc_id+2].set_weights(fc2)

tic = time.perf_counter()
score = lenet_model.evaluate(x_test, y_test, verbose=0)
print("Compressed SVD Realization time", time.perf_counter()-tic)
truncsvd_accuracy = score[1]
print('Compressed SVDTruncated Accuracy ', truncsvd_accuracy)

lenet_model.summary()

