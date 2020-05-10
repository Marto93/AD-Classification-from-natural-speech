# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 18:13:20 2020

@author: Luca Martorano
"""

#%%

## Importing libraries

import numpy as np
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator

# Defining directories, hight/width of images and Sample (Train/Val) size

train_dir = 'C:/Users/marto/Desktop/Datasets/2. Dem@Care_Resized/Train'
val_dir = 'C:/Users/marto/Desktop/Datasets/2. Dem@Care_Resized/Val'

img_width, img_height, channels = 224, 224, 3
train_size, val_size = 436, 48

# Instantiate convolutional base

conv_base = VGG16(weights = 'imagenet',
                  include_top=False,
                  input_shape=(img_width, img_height, channels))

# Importing images with image data generator

datagen_train = ImageDataGenerator(rescale = 1./255)
batch_size = 4

############################ FEATURES EXTRACTION ##############################

# Extraction Training Features

def extract_features_train(directory, sample_count):
    features = np.zeros(shape=(sample_count, 7, 7, 512))  # Must be equal to the output of the convolutional base
    labels = np.zeros(shape=(sample_count))
    
    # Preprocess data
    generator_t = datagen_train.flow_from_directory(directory,
                                            target_size=(img_width,img_height),
                                            batch_size = batch_size,
                                            class_mode= 'binary')
    
    
    # Pass data through the convolutional base
    
    i = 0
    for inputs_batch, labels_batch in generator_t: 
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels

train_features, train_labels = extract_features_train(train_dir, train_size)

# Extraction Validation Features

datagen_val = ImageDataGenerator(rescale = 1./255)

def extract_features_val(directory, sample_count):
    features = np.zeros(shape=(sample_count, 7, 7, 512))  # Must be equal to the output of the convolutional base
    labels = np.zeros(shape=(sample_count))
    
    # Preprocess data
    generator_v = datagen_val.flow_from_directory(directory,
                                            target_size=(img_width,img_height),
                                            batch_size = batch_size,
                                            class_mode= 'binary')
    
    
    # Pass data through convolutional base
    i = 0
    for inputs_batch, labels_batch in generator_v: 
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels

val_features, val_labels = extract_features_val(val_dir, val_size)

#%%

################################## MODELING ###################################

# Importing deep learning libraries

import keras
from keras import models
from keras import layers
from keras import regularizers
from keras import optimizers
import keras_metrics as km
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold

# Setting K Fold Cross Validation

# fix random seed for reproducibility
seed = 42
epochs = 50

kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
cvscores = []

# Defining the model

for train, val in kfold.split(train_features, train_labels):
    
    #  Model Creation

    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(7,7,512)))
    model.add(layers.Dense(256, activation='relu', input_dim=(7*7*512),
              kernel_regularizer=keras.regularizers.l1(l1=0.1)))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(1, activation='sigmoid'))
    

    # Compiling the model

    model.compile(optimizer = Adam(lr= 1e-4),
              loss='binary_crossentropy',
              metrics=['acc', km.binary_precision(), km.binary_recall()])


    # Training the model
     
    #es = EarlyStopping(monitor='val_loss', mode='min')
    
    model.fit(train_features, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data = (val_features, val_labels),
              #callbacks = [es],
              verbose = 1)
              

    # Saving the model

    model.save('model_name.h5')
    
    # Evaluate the model
    scores = model.evaluate(val_features, val_labels, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))


#%%

############################### PLOTTING ######################################

# Plotting Accuracy

plt.plot(history.history["acc"])
plt.plot(history.history['val_acc'])
plt.title("Train/Val Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy","Validation Accuracy"])
plt.show()

# Plotting Loss Function

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Train/Val Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Loss","Validation Loss"])
plt.show()

#%%


