# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 12:03:20 2020

Train model according to J.N. Kather, et al. Nature Medicine.  https://doi.org/10.1038/s41591-019-0462-y (2019)
Other files in need: network.py

@author: zhang
"""
# ------------------------------------
import keras
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

import os
from keras.layers import Input
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from network import *

path = '/public/home/msi_nature'
os.chdir(path)
disease_type = 'DX' # DX, KR, STAD

################################################################
######################## loading data ##########################
################################################################
train_imagePath = path + '/data/{}/train'.format(disease_type)

### parameters of model
InitialLearnRate = 1e-5
L2Regularization = 1e-4
MiniBatchSize = 256
MaxEpoch = 100
hotLayers = 0
learningRateFactor = 2
PixlRangeShear = 5
img_height = 224
img_width = 224
channels = 3
num_classes = 2

train_datagen = ImageDataGenerator(rotation_range=360,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   width_shift_range=0.05,
                                   height_shift_range=0.05,
                                   validation_split=0.2,
                                   rescale=1 / 128.
                                   )

train_generator = train_datagen.flow_from_directory(directory = train_imagePath,
                                                    target_size=(img_width, img_height),
                                                    batch_size=MiniBatchSize,
                                                    class_mode='categorical',
                                                    subset='training'
                                                    )

val_generator = train_datagen.flow_from_directory(directory = train_imagePath,
                                                  target_size=(img_width, img_height),
                                                  batch_size=MiniBatchSize,
                                                  class_mode='categorical',
                                                  subset='validation'
                                                  )

print(train_generator.class_indices)
print(val_generator.class_indices)

###################################
########## train model ############
###################################
inputs = Input(shape=(img_height, img_width, channels))
resnet18 = ResNet(inputs, block = basic_block, layers=[2, 2, 2, 2], include_top=False)
model = build_model(resnet18, num_classes, InitialLearnRate)
#model.summary()
 
early_stopping = EarlyStopping(patience=MaxEpoch, verbose=1)
reduce_lr = ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1)

weight_path = path + '/weights'
os.chdir(weight_path)
json_string = model.to_json()
open('model_{}.json'.format(format(disease_type)), 'w').write(json_string)
checkpointer = ModelCheckpoint(filepath=weight_path +'/weights_{}.h5'.format(disease_type), verbose=1, 
                               monitor='val_loss', mode='auto', save_best_only=True)

N_epochs = MaxEpoch
batch_size = MiniBatchSize
model.fit_generator(train_generator,
                    verbose = 2,
                    epochs = N_epochs,
                    steps_per_epoch=train_generator.samples // MiniBatchSize,
                    validation_data = val_generator,
                    validation_steps = val_generator.samples // MiniBatchSize,
                    callbacks = [checkpointer, early_stopping, reduce_lr])
model.save_weights('last_weights_{}.h5'.format(disease_type), overwrite=True)
