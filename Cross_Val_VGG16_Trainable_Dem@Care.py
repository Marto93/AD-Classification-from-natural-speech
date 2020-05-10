# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 11:31:57 2020

@author: Luca Martorano
"""

############################## IMPORTING LIBRARIES ############################
#%%

import os
import keras
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from keras.applications.vgg16 import VGG16 




os.chdir = 'C:/Users/marto/Desktop/Datasets'
train_dir = 'C:/Users/marto/Desktop/Datasets/2. Dem@Care_Resized/Train'
validation_dir = 'C:/Users/marto/Desktop/Datasets/2. Dem@Care_Resized/Val'

############################### IMPORTING DATA ################################

IMG_DIM = (224, 224)

## Importing Train Data (0)

train_files_0 = glob.glob(train_dir + '/0/*')
train_imgs_0 = [img_to_array(load_img(img, target_size=IMG_DIM)) for img in train_files_0]
train_imgs_0 = np.array(train_imgs_0)
train_labels_0 = np.zeros(len(train_files_0))

## Importing Train Data (1)

train_files_1 = glob.glob(train_dir + '/1/*')
train_imgs_1 = [img_to_array(load_img(img, target_size=IMG_DIM)) for img in train_files_1]
train_imgs_1 = np.array(train_imgs_1)
train_labels_1 = np.ones(len(train_files_1))

## Importing Validation Data (1)

val_files_0 = glob.glob(validation_dir + '/0/*')
val_imgs_0 = [img_to_array(load_img(img, target_size=IMG_DIM)) for img in val_files_0]
val_imgs_0 = np.array(val_imgs_0)
val_labels_0 = np.zeros(len(val_files_0))

## Importing Validation Data (1)


val_files_1 = glob.glob(validation_dir + '/1/*')
val_imgs_1 = [img_to_array(load_img(img, target_size=IMG_DIM)) for img in val_files_1]
val_imgs_1 = np.array(val_imgs_1)
val_labels_1 = np.ones(len(val_files_1))

## Merging train 0/1 and validation 0/1 data

train_data = np.concatenate((train_imgs_0, train_imgs_1), axis = 0)
train_labels = np.concatenate((train_labels_0, train_labels_1), axis = 0)
val_data = np.concatenate((val_imgs_0, val_imgs_1), axis = 0)
val_labels = np.concatenate((val_labels_0, val_labels_1), axis = 0)

#%%
############################## DATA GENERATOR #################################

input_shape = (224,224,3)
batch_size = 2

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow(train_data,
                                     train_labels)

val_generator = val_datagen.flow(val_data,
                                 val_labels)


#%%
############################ CONVOLUTIONAL BASE ###############################

vgg_model = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=input_shape)

output = vgg_model.layers[-1].output
output = keras.layers.Flatten()(output)
vgg_model = Model(vgg_model.input, output)

vgg_model.trainable = True
vgg_model.summary()

set_trainable = False
for layer in vgg_model.layers:
    if layer.name in ['block5_conv1','block5_conv2','block5_conv3' 'block4_conv1',
                      'block4_conv2','block4_conv3']:
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
        
layers = [(layer, layer.name, layer.trainable) for layer in vgg_model.layers]
pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable']) 

#%%
############################ FEATURES EXTRACTION ##############################


def get_bottleneck_features(model, input_imgs):
    features = model.predict(input_imgs, verbose=0)
    return features
    
train_feat_vgg = get_bottleneck_features(vgg_model, train_data)
val_feat_vgg = get_bottleneck_features(vgg_model, val_data)

print('Train Bottleneck Features:', train_feat_vgg.shape, 
      '\tValidation Bottleneck Features:', val_feat_vgg.shape)

#%%
########################### CLASSIFIER CREATION ###############################

import keras_metrics as km
from keras import models,layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras import optimizers
from sklearn.model_selection import StratifiedKFold

seed = 42
kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state = seed)
cv_scores = []

# Defining the Model

for train, val in kfold.split(train_feat_vgg, train_labels):
    
    model = models.Sequential()
    model.add(layers.Dense(256, activation='tanh', input_dim=(7*7*512)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    # Compile model
    model.compile(optimizer = optimizers.Adam(lr=1e-4),
                  loss='binary_crossentropy',
                  metrics=['acc', km.binary_precision(), km.binary_recall()])

    # Training the Model

    history = model.fit(train_feat_vgg, train_labels,
                        epochs=100,
                        batch_size=batch_size, 
                        validation_data=(val_feat_vgg, val_labels))

    # Evaluate the model
    scores = model.evaluate(val_feat_vgg, val_labels, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cv_scores.append(scores[1] * 100)

print("%.2f%% (+/- %.2f%%)" % (np.mean(cv_scores), np.std(cv_scores)))