# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 14:09:14 2020

Build model according to J.N. Kather, et al. Nature Medicine.  https://doi.org/10.1038/s41591-019-0462-y (2019)
functions for gradCAM.py

@author: zhang
"""

from keras.layers import Conv2D, BatchNormalization, Activation
from keras.layers import MaxPool2D, GlobalAveragePooling2D, Dense, Dropout
from keras.layers import add
from keras.models import Model, Sequential
from keras.optimizers import Adam

########################################################################
########################## build ResNet18 ##############################
########################################################################
def basic_block(input_tensor, filters, stride=1, downsample=None):
    residual = input_tensor
    # 3x3
    x = Conv2D(filters, kernel_size=3, strides=stride, padding='same')(input_tensor)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    # 3x3
    x = Conv2D(filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    if downsample:  
        residual = downsample(input_tensor)
    x = add([x, residual])
    x = Activation('relu')(x)

    return x

def bottleneck(input_tensor, filters, stride=1, downsample=None):
    residual = input_tensor
    # 1x1
    x = Conv2D(filters, kernel_size=1, padding='same')(input_tensor)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    print(x.shape, residual.shape)
    # 3x3 core
    x = Conv2D(filters, kernel_size=3, strides=stride, padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    print(x.shape, residual.shape)
    # 1x1
    x = Conv2D(filters * 4, kernel_size=1, padding='same')(x)  # fms * 4
    x = BatchNormalization(axis=3)(x)
    if downsample:
        residual = downsample(input_tensor)
    print(x.shape, residual.shape)
    x = add([x, residual])
    x = Activation('relu')(x)

    return x

in_filters = 64

def make_layer(input_tensor, block, filters, blocks, stride=1):
    """
    :param input_tensor:
    :param block: basic_block, bottleneck
    :param filters: channels of output
    :param blocks: block repeat times
    :param stride: stage345, stride=2; stage2, stride = 1 (stage 1 with maxpool)
    :return: layer output
    """
    global in_filters
    downsample = None
    expansion = 4 if block.__name__ == 'bottleneck' else 1

    """
    whether residual need downsample 
      basic_block
        - stage2, stride=1, in_filters = filters * expansion, no downsample
      bottleneck
        - stage2，stride=1, in_filters != filters * expansion, use downsample to change channel
        - stride != 1, it's different according to different block (stage 3, 4, 5))
    """
    if stride != 1 or in_filters != filters * expansion:
        downsample = Sequential([
            Conv2D(filters * expansion, kernel_size=1, strides=stride, padding='same'),
            BatchNormalization(axis=3)
        ])
    in_filters = filters * expansion  # next layer input filters

    out = block(input_tensor, filters=filters, stride=stride, downsample=downsample)  # layer
    for i in range(1, blocks):
        out = block(out, filters=filters)  # repeat layer, no downsample, stride=1
    return out

in_filters = 64


def ResNet(input_tensor, block, layers, include_top=True, classes=1000):
    # stage1
    x = Conv2D(64, (7, 7), strides=(2, 2),  # 224 -> 112
               padding='same',
               kernel_initializer='he_normal',
               name='conv1')(input_tensor)
    x = BatchNormalization(axis=3, name='conv1_bn')(x)  # channel last
    x = Activation('relu', name='conv1_relu')(x)
    x = MaxPool2D((3, 3), strides=(2, 2), padding='same')(x)  # 112->56

    x = make_layer(x, block, filters=64, blocks=layers[0])  # stage 2
    x = make_layer(x, block, filters=128, blocks=layers[1], stride=2)  # stage 3
    x = make_layer(x, block, filters=256, blocks=layers[2], stride=2)  # stage 4
    x = make_layer(x, block, filters=512, blocks=layers[3], stride=2)  # stage 5

    # fc
    if include_top:
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(classes, activation='softmax', name='fc1000')(x)

    print(x.shape)

    model = Model(inputs=[input_tensor], outputs=[x])
    return model

######################################################
#################### DL for msi ######################
######################################################
def build_model(network, num_classes, InitialLearnRate):
    """
    build model according to J.N. Kather, et al. Nature Medicine. 
    https://doi.org/10.1038/s41591-019-0462-y (2019)
    """
    x = network.output
    x = GlobalAveragePooling2D(name = 'avg_pool')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation = 'softmax', name = 'fc')(x)
    model = Model(inputs = resnet18.input, outputs = predictions)
    model.compile(
        loss = 'categorical_crossentropy',
        optimizer = Adam(lr = InitialLearnRate),
        metrics=["accuracy"]
    )
    
    return model
