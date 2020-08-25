# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 17:16:37 2020

functions for gradCAM.py
reference: https://github.com/jacobgil/keras-grad-cam/blob

@author: zhang
"""

# ------------------------------------
import keras
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

from keras.models import Model
from keras.models import model_from_json

################################################################
######################## grad cam ##############################
################################################################
from keras.layers.core import Lambda
from tensorflow.python.framework import ops
import keras.backend as K
import tensorflow as tf
import numpy as np
import keras
import cv2

def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape

def normalize(x):
    # utility function to normalized a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def register_gradient():
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                tf.cast(op.inputs[0] > 0., dtype)

def compile_saliency_function(model, activation_layer = 'add_8'):
    input_img = model.input
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    layer_output = layer_dict[activation_layer].output
    max_output = K.max(layer_output, axis = 3)
    saliency = K.gradients(K.sum(max_output), input_img)[0]
    return K.function([input_img, K.learning_phase()], [saliency])

def modify_backprop(model, name, model_path, weight_path):
    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': name}):
        
        # get layers that have an activation
        layer_dict = [layer for layer in model.layers[1:] if hasattr(layer, 'activation')]
        
        # replace relu activtion
        for layer in layer_dict:
            if layer.activation == keras.activations.relu:
                layer.activation = tf.nn.relu
                
        # re-instanciate a new model
        new_model = model_from_json(open(model_path).read())
        new_model.load_weights(weight_path)
        return new_model
    
def deprocess_image(x):
    '''
    Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    '''
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    
    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)
    
    # convert to RGB array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def get_gradient_function(input_model, category_index, nb_classes, layer_name):
    target_layer = lambda x: target_category_loss(x, category_index, nb_classes)
    
    x = input_model.layers[-1].output
    x = Lambda(target_layer, output_shape = target_category_loss_output_shape)(x)
    model = Model(input_model.layers[0].input, x)
    
    loss = K.sum(model.layers[-1].output)
    conv_output = [l for l in model.layers if l.name == layer_name][0].output
    
    grads = normalize(K.gradients(loss, conv_output)[0])    
    gradient_function = K.function([model.layers[0].input], [conv_output, grads])
    return gradient_function

def grad_cam(gradient_function, image, img_size):
    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]
    
    weights= np.mean(grads_val, axis = (0, 1))
    cam = np.ones(output.shape[0 : 2], dtype = np.float32)
    
    for i, w in enumerate(weights):
        cam += w * output[:, :, i]
        
    cam = cv2.resize(cam, (img_size, img_size))
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)
    
    # Return to BGR [0..255] from the preprocessed image
    image *= 128
    image = image[0, :]
    image -= np.min(image)
    image = np.minimum(image, 255)
    
    cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(image)
    cam = 255 * cam / np.max(cam)
    return np.uint8(cam), heatmap

