# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 15:59:23 2020

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
from keras.applications.inception_v3 import InceptionV3

#%%
###############################################################################
#%%

os.chdir = 'C:/Users/marto/Desktop/Datasets'
train_dir = 'C:/Users/marto/Desktop/Datasets/3. Dem@Care_Resized_V3/Train'
validation_dir = 'C:/Users/marto/Desktop/Datasets/3. Dem@Care_Resized_V3/Val'

############################### IMPORTING DATA ################################

IMG_DIM = (299, 299)

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

input_shape = (299,299,3)
batch_size = 4

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow(train_data,
                                     train_labels)

val_generator = val_datagen.flow(val_data,
                                 val_labels)


#%%
############################ CONVOLUTIONAL BASE ###############################

incV3_model = InceptionV3(weights='imagenet',
                  include_top=False,
                  input_shape=input_shape)

output = incV3_model.layers[-1].output
output = keras.layers.Flatten()(output)
incV3_model = Model(incV3_model.input, output)

incV3_model.trainable = True
incV3_model.summary()
#%%
set_trainable = False
for layer in incV3_model.layers:
    if layer.name in ['conv2d_94', 'conv2d_93', 'conv2d_92', 'conv2d_91', 'conv2d_90',
                      'conv2d_89','conv2d_88','conv2d_87','conv2d_86','conv2d_85',
                      'conv2d_84','conv2d_83','conv2d_82','conv2d_81','conv2d_80',
                      'conv2d_79','conv2d_78','conv2d_77','conv2d_76','conv2d_75',
                      'conv2d_74','conv2d_73','conv2d_72','conv2d_71','conv2d_70',
                      'conv2d_69','conv2d_68','conv2d_67','conv2d_66','conv2d_65',
                      'conv2d_64','conv2d_63','conv2d_62','conv2d_61','conv2d_60',
                      'conv2d_59','conv2d_58','conv2d_57','conv2d_56','conv2d_55',
                      'conv2d_54','conv2d_53','conv2d_52','conv2d_51','conv2d_50',
                      'conv2d_49','conv2d_48','conv2d_47','conv2d_46','conv2d_45']:
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
        
layers = [(layer, layer.name, layer.trainable) for layer in incV3_model.layers]
pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable']) 

#%%
############################ FEATURES EXTRACTION ##############################


def get_bottleneck_features(model, input_imgs):
    features = model.predict(input_imgs, verbose=0)
    return features
    
train_feat_incV3 = get_bottleneck_features(incV3_model, train_data)
val_feat_incV3 = get_bottleneck_features(incV3_model, val_data)

print('Train Bottleneck Features:', train_feat_incV3.shape, 
      '\tValidation Bottleneck Features:', val_feat_incV3.shape)

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

for train, val in kfold.split(train_feat_incV3, train_labels):
    
    model = models.Sequential()
    model.add(layers.Dense(256, activation='tanh', input_dim=(8*8*2048)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    # Compile model
    model.compile(optimizer = optimizers.Adam(lr=1e-6),
                  loss='binary_crossentropy',
                  metrics=['acc', km.binary_precision(), km.binary_recall()])

    # Training the Model

    history = model.fit(train_feat_incV3, train_labels,
                        epochs=100,
                        batch_size=batch_size, 
                        validation_data=(val_feat_incV3, val_labels))

    # Evaluate the model
    scores = model.evaluate(val_feat_incV3, val_labels, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cv_scores.append(scores[1] * 100)

print("%.2f%% (+/- %.2f%%)" % (np.mean(cv_scores), np.std(cv_scores)))
