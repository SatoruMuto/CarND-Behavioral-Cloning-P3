
# coding: utf-8

# In[1]:


import csv
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


lines = []
with open("driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)


# In[8]:


#if use all 3 camera images
images = []
measurements = []
for line in lines[3000:]:
# reduce sample to reduce calclation time... 
    for i in range(3):
        source_path = line[i]
        image = cv2.imread(source_path)
        images.append(image)
        if i == 0:
            measurement = float(line[3])
        elif i == 1:
            #left picture, right turn is positive steering angle so add 0.2 
            measurement = float(line[3]) + 0.2
        else:
            #right picture, right turn is negative steering angle so reduce 0.2 
            measurement = float(line[3]) - 0.2
            
        measurements.append(measurement)


# In[10]:


# Preproess, flip picture and steering angle to normalize clockwise and counter-clockwise data
augmented_images, augmented_measurements = [],[]
for image,measurement in zip(images,measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement * -1.0)


# In[11]:


X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)


# In[13]:


from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D


# In[16]:


# Nvidia end-to-end model

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape = (160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 6)

model.save('model_nv.h5')

