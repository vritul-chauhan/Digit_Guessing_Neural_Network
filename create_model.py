#!/usr/bin/env python
# coding: utf-8

# # Importing modules

# In[1]:

import install_requirements

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import tensorflow as tf 
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.datasets import mnist


# ## Getting the dataset and diving into test and train

# In[2]:


(X_train, y_train), (X_test, y_test) = mnist.load_data()


# ### Size of train and test set

# In[3]:


print("X_train Shape", X_train.shape)
print("y_train Shape", y_train.shape)


# ## Preprocessing the input data , reshaping to (28,28,1)

# In[4]:


X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train/=255
X_test/=255

X_train.shape


# In[5]:


class_total = 10
Y_train = tf.keras.utils.to_categorical(y_train, class_total)
Y_test = tf.keras.utils.to_categorical(y_test, class_total)


# ## Preprocessing Training set

# In[7]:


train_gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
                         height_shift_range=0.08, zoom_range=0.08)
train_set = train_gen.flow(X_train, Y_train, batch_size=64)


# ## Preprocessing Test set

# In[8]:


test_gen = ImageDataGenerator()
test_set = test_gen.flow(X_test, Y_test, batch_size=64)


# # Building the CNN

# In[9]:


cnn = tf.keras.models.Sequential()


# ## Adding Hidden Layer 1

# In[10]:


cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size= 3, activation='relu', input_shape= [28,28,1]))


# ## MaxPooling Layers 1

# In[11]:


cnn.add(tf.keras.layers.MaxPool2D(pool_size= (2,2), strides = 2))


# ## Adding Hidden Layer 2 and MaxPooling layer 2

# In[12]:


cnn.add(tf.keras.layers.Conv2D(filters = 64, kernel_size= 3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size= (2,2), strides = 2))


# ## Flattening

# In[13]:


cnn.add(tf.keras.layers.Flatten())


# ## Fulling connected layer 1 with 120 nodes

# In[14]:


cnn.add(tf.keras.layers.Dense(units = 120, activation = 'relu'))


# ## Fully connected Layer 2 with 84 nodes

# In[15]:


cnn.add(tf.keras.layers.Dense(units = 84, activation = 'relu'))


# ## Output layer with 10 nodes

# In[16]:


cnn.add(tf.keras.layers.Dense(units = 10, activation = 'softmax'))


# In[17]:


cnn.summary()


# # Compiling the CNN

# In[18]:


cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# In[20]:


cnn.fit_generator(train_set, steps_per_epoch=60000/64, epochs=10, 
                    validation_data=test_set, validation_steps=10000/64)


# In[21]:


score = cnn.evaluate(X_test, Y_test)
print()
print('Test accuracy: ', score[1])

