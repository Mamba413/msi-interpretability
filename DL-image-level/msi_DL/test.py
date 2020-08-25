# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 14:31:51 2020

Test the DL network for MSI.

@author: zhang
"""
# ------------------------------------
import keras
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

import os
import re
import numpy as np
import pandas as pd
from keras.utils import np_utils
from sklearn import metrics
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
from math import sqrt
import matplotlib.pyplot as plt

def return_patient_ID(comp_ID):
    """
    function for getting patient ID
    input: 
        comp_ID - complete ID (filename)
    output:
        ID_match - patient ID
    """
    ID_match=re.search(r'(?<=TCGA-[A-Z0-9][A-Z0-9]-)[0-9A-Z]*',comp_ID,re.I)
    return ID_match.group() 

path = '/public/home/msi_nature'
os.chdir(path)
disease_type = 'DX' # DX, KR, STAD
################################################################
###################### loading model ###########################
################################################################
weight_path = path + '/weights'
os.chdir(weight_path)
model = model_from_json(open('model_{}.json'.format(disease_type)).read())
model.load_weights('weights_{}.h5'.format(disease_type))

################################################################
######################## loading data ##########################
################################################################
test_imagePath = path + '/data/{}/test'.format(disease_type)
save_path = path + '/result/{}'.format(disease_type)
if not os.path.isdir(save_path):
    os.makedirs(save_path)
os.chdir(save_path)

img_height = 224
img_width = 224
channels = 3
num_classes = 2

test_datagen = ImageDataGenerator(rescale = 1 / 128.)
test_generator = test_datagen.flow_from_directory(directory = test_imagePath,
                                                    target_size = (img_width, img_height),
                                                    batch_size = 1,
                                                    class_mode = 'categorical',
                                                    shuffle = False
                                                    )

print(test_generator.class_indices)
test_labels = test_generator.classes
test_labels = np_utils.to_categorical(test_labels, num_classes).astype(dtype = 'float16')
y_test_preds = model.predict_generator(test_generator, steps = test_generator.samples, verbose=1)

###################################################
############## patient ID result ##################
################################################### 
patient_name = test_generator.filenames
patient_ID = list(map(return_patient_ID, patient_name))
patient_ID_set = set(patient_ID)
print(len(patient_ID_set))

true_label = test_labels.copy()
y_pred = y_test_preds.copy()
patient_dict = {'ID':patient_ID, 'true_0': true_label[:, 0], 'true_1': true_label[:, 1],
                'pred_0': y_pred[:, 0], 'pred_1': y_pred[:, 1]}
patient_df = pd.DataFrame(patient_dict)
patient_mean = patient_df.groupby('ID').mean()
patient_mean = patient_mean.reset_index()
patient_true_label = np.asarray([patient_mean['true_0'], patient_mean['true_1']])
patient_true_label = np.transpose(patient_true_label, [1, 0])
patient_y_pred = np.asarray([patient_mean['pred_0'], patient_mean['pred_1']])
patient_y_pred = np.transpose(patient_y_pred, [1, 0])
patient_auc = metrics.roc_auc_score(patient_true_label, patient_y_pred, average='micro')
print('{} patient auc: '.format(disease_type), patient_auc)
interval = 1.96 * sqrt((patient_auc * (1 - patient_auc)) / len(patient_ID_set))
print('95% CI: [', patient_auc - interval, ', ', patient_auc + interval, ']')

patient_fpr, patient_tpr, patient_thresholds = metrics.roc_curve(patient_true_label.ravel(), patient_y_pred.ravel())

####
### plot roc
plt.figure()
lw = 2
plt.figure(figsize=(10,10))
plt.plot(patient_fpr, patient_tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.4f)' % patient_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('{} Patient MSI Roc'.format(disease_type))
plt.legend(loc="lower right")
plt.savefig('{}_patient_ROC.png'.format(disease_type))

patient_y_pred = np.argmax(patient_y_pred,axis=1)
patient_y_pred = np.asarray(patient_y_pred)
patient_true_label = np.argmax(patient_true_label,axis=1)
patient_true_label = np.asarray(patient_true_label)

patient_acc = metrics.accuracy_score(patient_true_label, patient_y_pred)
print('{} patient accuracy: '.format(disease_type), patient_acc)
patient_precision = metrics.precision_score(patient_true_label, patient_y_pred, average = None)
print('{} patient precision: '.format(disease_type), patient_precision)
patient_confusion = metrics.confusion_matrix(patient_true_label, patient_y_pred)
print('{} patient confusion: '.format(disease_type), patient_confusion)

####################################################
################### total ##########################
####################################################
auc = metrics.roc_auc_score(test_labels, y_test_preds, average='micro')
print('{} total auc: '.format(disease_type), auc)
interval = 1.96 * sqrt((auc * (1 - auc)) / len(patient_ID))
print('95% CI: [', auc - interval, ', ', auc + interval, ']')

fpr, tpr, thresholds = metrics.roc_curve(test_labels.ravel(), y_test_preds.ravel())
### plot roc
plt.figure()
lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.4f)' % auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('{} total MSI roc'.format(disease_type))
plt.legend(loc="lower right")
plt.savefig('{}_total_ROC.png'.format(disease_type))

y_test_preds = np.argmax(y_test_preds,axis=1)
y_test_preds = np.asarray(y_test_preds)
test_labels = np.argmax(test_labels,axis=1)
test_labels = np.asarray(test_labels)

acc = metrics.accuracy_score(test_labels, y_test_preds)
print('{} total accuracy: '.format(disease_type), acc)
precision = metrics.precision_score(test_labels, y_test_preds, average = None)
print('{} total precision: '.format(disease_type), precision)
confusion = metrics.confusion_matrix(test_labels, y_test_preds)
print('{} total confusion: '.format(disease_type), confusion)
