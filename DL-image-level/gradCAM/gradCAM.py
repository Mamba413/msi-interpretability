# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 17:07:28 2020

Grad CAM:
    reference: https://github.com/jacobgil/keras-grad-cam/blob
    other file in need: functions_for_gradCAM.py

output:
    disease_result.csv - including filename, origin label, predict label, and predict probs
    heatmap images
    guided heatmap images
    concatenate of origin images, heatmap images, and guided heatmap images

@author: Yuting Zhang
"""
# ------------------------------------
import keras
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

import os
import gc
import cv2
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
from functions_for_gradCAM import *

path = '/public/home/msi_nature'
os.chdir(path)
################################################################
######################## load model ############################
################################################################
weight_path = path + '/weights'
os.chdir(weight_path)

dataFrom = 'DX' # DX, KR, STAD

model = model_from_json(open('model_' + dataFrom + '.json').read())
model.load_weights('best_weights_' + dataFrom + '.h5')

#### gradCAM
register_gradient()
guided_model = modify_backprop(model, 'GuidedBackProp',
                               model_path = weight_path + '/model_' + dataFrom + '.json', 
                               weight_path = weight_path + '/best_weights_' + dataFrom + '.h5')
saliency_fn = compile_saliency_function(guided_model)

################################################################
########################## load data ###########################
################################################################
test_imagePath = path + '/data/{}/test'.format(dataFrom)
save_path = path + '/result/GradCAM/{}'.format(dataFrom)
save_path_combine = path + '/result/GradCAM/combine/{}'.format(dataFrom)

save_path_heatmap = save_path + '/heatmap'
if not os.path.isdir(save_path_heatmap):
    os.makedirs(save_path_heatmap)

save_path_guided = save_path + '/guided'
if not os.path.isdir(save_path_guided):
    os.makedirs(save_path_guided) 
    
save_path_heatmap_combine = save_path_combine + '/heatmap_combine'
if not os.path.isdir(save_path_heatmap_combine):
    os.makedirs(save_path_heatmap_combine)

save_path_guided_combine = save_path_combine + '/guided_combine'
if not os.path.isdir(save_path_guided_combine):
    os.makedirs(save_path_guided_combine)    
    
save_path_all = save_path_combine + '/heatmap_guided'
if not os.path.isdir(save_path_all):
    os.makedirs(save_path_all)    
    
img_height = 224
img_width = 224
channels = 3
num_classes = 2

test_datagen = ImageDataGenerator(rescale = 1 / 128.)
test_generator = test_datagen.flow_from_directory(directory = test_imagePath,
                                                    target_size = (img_width, img_height),
                                                    batch_size = 1,
                                                    class_mode = 'categorical',
                                                    shuffle = False)
test_generator.reset()
print(test_generator.class_indices)
class_name = list(test_generator.class_indices)
patient_name = test_generator.filenames
filenames = list(patient_name)
test_labels = test_generator.classes
y_test_preds = model.predict_generator(test_generator, steps = test_generator.samples, verbose=2)
y_test_probs = np.max(y_test_preds, axis = 1)
y_test_probs = list(y_test_probs)
y_test_preds = np.argmax(y_test_preds,axis=1)
y_test_preds = np.asarray(y_test_preds)

save_img_name = list(map(lambda x: x.split('/')[-1], filenames))
origin_labels = list(map(lambda x: 'MSI' if x == 0 else 'MSS', test_labels))
pred_labels = list(map(lambda x: 'MSI' if x == 0 else 'MSS', y_test_preds))
resultDF = np.array([save_img_name, origin_labels, pred_labels, y_test_probs])
resultDF = np.transpose(resultDF, (1, 0))
resultDF = pd.DataFrame(resultDF, columns = ['iamge_name', 'origin_label',
                                             'pred_label', 'pred_prob'])
resultDF.to_csv(save_path + '/{}_result.csv'.format(dataFrom), index = False)
################################################################
########################## Grad CAM ############################
################################################################
gradFun_List = []
for i in range(num_classes):
    gradFun_List.append(get_gradient_function(model, i, num_classes, layer_name = 'add_8'))

test_generator.reset()
# grad cam heatmap
empty_array = np.ones((224, 40, 3))
empty_array = np.asarray(empty_array * 255).astype(np.uint8)
count = 0
for x_batch, y_batch in test_generator:
    preprocessed_input = x_batch[0]
    preprocessed_input = np.expand_dims(preprocessed_input, axis=0)
    image_name = filenames[count]
    image_type = image_name.split('/')[-2]
    image_name = image_name.split('/')[-1].split('.')[0]
    predicted_class = y_test_preds[count]

    cam, heatmap = grad_cam(gradFun_List[int(predicted_class)], preprocessed_input, img_height)
    cv2.imwrite(save_path_heatmap + '/{}.jpg'.format(image_name), cam)
        
    img1 = x_batch[0]
    img1 = np.asarray(img1).astype(np.uint8)
    img2 = cam.copy()
    combine1 = np.concatenate((img1, empty_array, img2), axis = 1).astype(np.uint8)
    cv2.imwrite(save_path_heatmap_combine + '/{}.jpg'.format(image_name), combine1)
    
    saliency = saliency_fn([preprocessed_input, 0])
    gradcam = saliency[0] * heatmap[..., np.newaxis]
    gradcam = deprocess_image(gradcam)
    gradcam = cv2.resize(gradcam, (img_height, img_height))
    cv2.imwrite(save_path_guided + '/{}.jpg'.format(image_name), gradcam)
    
    img3 = gradcam.copy()
    combine2 = np.concatenate((img1, empty_array, img3), axis = 1).astype(np.uint8)
    cv2.imwrite(save_path_guided_combine + '/{}.jpg'.format(image_name), combine2)
    
    combine3 = np.concatenate((img1, empty_array, img2, empty_array, img3), axis = 1).astype(np.uint8)
    cv2.imwrite(save_path_all + '/{}.jpg'.format(image_name), combine3)
    
    count += 1
    if count % 100 == 0:
        print(str(count) + ' pic have been done...')
    if count >= len(filenames):
            break
        
    del cam, heatmap, saliency, gradcam
    gc.collect()
