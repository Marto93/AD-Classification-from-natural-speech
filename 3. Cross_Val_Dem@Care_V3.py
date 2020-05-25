# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 15:44:59 2020

@author: Marto
"""

#%%
import os
import numpy as np
from keras.applications.inception_v3 import InceptionV3

main = os.chdir('C:/Users/marto/Desktop/Datasets')
train_dir = 'C:/Users/marto/Desktop/Datasets/3. Dem@Care_Resized_V3/Train'
validation_dir = 'C:/Users/marto/Desktop/Datasets/3. Dem@Care_Resized_V3/Val'

# Instantiate convolutional base

img_width, img_height, channels = 299, 299, 3

train_size, validation_size = 436, 48

conv_base = InceptionV3(weights= 'imagenet',
                  include_top=False,
                  input_shape=(img_width, img_height, channels))


# Estracting features from VGG16 architecture 

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 4

def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 8, 8, 2048))
    labels = np.zeros(shape=(sample_count))
    
    # Preprocess data
    generator = datagen.flow_from_directory(directory,
                                            target_size=(img_width,img_height),
                                            batch_size = batch_size,
                                            class_mode='binary')
                                            
    
    # Pass data through convolutional base
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels

train_features, train_labels = extract_features(train_dir, train_size)  
validation_features, validation_labels = extract_features(validation_dir, validation_size)
#%%

#%%

################################ MODELING #####################################
# Define model

from keras import models
from keras import layers
from keras import optimizers
import keras_metrics as km
from keras.optimizers import RMSprop, Adam
from sklearn.model_selection import StratifiedKFold

seed = 42
epochs = 100

kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state = seed)
cv_scores = []

# Defining the Model

for train, val in kfold.split(train_features, train_labels):
    
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(8,8,2048)))
    model.add(layers.Dense(256, activation='tanh', input_dim=(8*8*2048)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    # Compile model
    model.compile(optimizer = Adam(lr=1e-4),
                  loss='binary_crossentropy',
                  metrics=['acc', km.binary_precision(), km.binary_recall()])

    # Training the Model

    history = model.fit(train_features, train_labels,
                        epochs=epochs,
                        batch_size=batch_size, 
                        validation_data=(validation_features, validation_labels))

    # Evaluate the model
    scores = model.evaluate(validation_features, validation_labels, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cv_scores.append(scores[1] * 100)

print("%.2f%% (+/- %.2f%%)" % (np.mean(cv_scores), np.std(cv_scores)))

#%%
################################ PLOTTING #####################################

import matplotlib.pyplot as plt

# Plotting Accuracy

plt.plot(history.history["acc"])
plt.plot(history.history['val_acc'])
plt.title("Train/Val Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy","Validation Accuracy"])
plt.show()

# Plotting Loss

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Train/Val Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Loss","Validation Loss"])
plt.show()

