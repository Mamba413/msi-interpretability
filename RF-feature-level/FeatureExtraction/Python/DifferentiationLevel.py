# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 15:35:00 2019

@author: Lenovo
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 09:51:21 2019

@author: Lenovo
"""

import os
import cv2 as cv2
import matplotlib.pyplot as plt
import math
import numpy as np
from PIL import Image, ImageStat, ImageEnhance
import time
from skimage.measure import label
from skimage import morphology
from skimage import data, draw, color, transform, feature
import re
import pandas as pd
import Get_Patient_ID as GPD


# ----- A rough detection of the circle  structure in H image as differentiation level
# ----- The input is H section from ImageJ Color Convolution

def circleIdentify(path, filename, ii=20, plot=False, erosion_value=220):
    img = cv2.imread(path + '\\' + filename)
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # erosion and dilation
    kernel = np.ones((5, 5), np.uint8)
    kernel2 = np.ones((8, 8), np.uint8)
    dilation = cv2.dilate(img2, kernel)
    erosion = cv2.erode(dilation, kernel2)

    if len(np.argwhere(erosion > 250)) < 15000:
        if len(np.argwhere(erosion < 200)) < 10000:
            posi = np.argwhere(erosion > erosion_value)
            for i, j in posi:
                erosion[i, j] = 255
        else:
            posi = np.argwhere(erosion > erosion_value - 20)
            for i, j in posi:
                erosion[i, j] = 255
                # canny detection
    binaryimg = cv2.Canny(erosion, 30, 100)
    contours, hierarchy = cv2.findContours(binaryimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    x = []
    for i in range(len(contours)):
        x.append(len(contours[i]))
    # edge detection
    posi = [i for i in range(len(x)) if x[i] > 50]
    contours2 = [contours[i] for i in posi]
    canvas = np.zeros((224, 224), dtype=np.uint8)
    cv2.drawContours(canvas, contours2, -1, (255, 255, 255), 3)
    # dilation again
    kernel = np.ones((3, 3), np.uint8)
    erosion2 = cv2.dilate(canvas, kernel)
    # circle detection
    _, labels = cv2.connectedComponents(erosion2)
    num_area = labels.max()
    flag = 0
    img3 = img.copy()
    er = 1
    for i in range(1, num_area + 1):
        position_label = np.argwhere(labels == i)
        er = er + 1
        depend_area = np.zeros((224, 224), dtype=np.uint8)
        area = len(position_label)
        if area < 25000:
            for j, h in enumerate(position_label):
                depend_area[h[0]][h[1]] = 255
            contours_solo, hierarchy = cv2.findContours(depend_area, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            if area < 2000:
                circles1 = cv2.HoughCircles(depend_area, cv2.HOUGH_GRADIENT, 1,
                                            100, param1=10, param2=ii - 5, minRadius=20, maxRadius=80)
            else:
                circles1 = cv2.HoughCircles(depend_area, cv2.HOUGH_GRADIENT, 1,
                                            100, param1=10, param2=ii, minRadius=20, maxRadius=80)

            if circles1 is not None:
                circles = circles1[0, :, :]
                circles = np.uint16(np.around(circles))
                for d in circles[:]:
                    cv2.circle(img3, (d[0], d[1]), d[2], (255, 0, 0), 5)
                    cv2.circle(img3, (d[0], d[1]), 2, (255, 0, 255), 10)
                flag = flag + len(circles)
    if plot:
        if flag >= 1:
            cv2.imshow('binaryimg', binaryimg)
            cv2.imshow('original image', img)
            cv2.imshow('erosion image', erosion)
            cv2.imshow('circle detection', img3)
            cv2.waitKey(0)
        else:
            print('find no cicle like detection')
            cv2.imshow('binaryimg', binaryimg)
            cv2.imshow('original image', img)
            cv2.imshow('erosion', erosion)
            cv2.imshow('circle detection', img3)
            cv2.waitKey(0)
    return flag


# main
if __name__ == '__main__':
    # input path
    path1 = 'E:\\data\\mss_msi\\STAD_data\\STAD_HE\\STAD_test_mss'
    path2 = 'E:\\data\\mss_msi\\STAD_data\\STAD_HE\\STAD_train_mss'
    path3 = 'E:\\data\\mss_msi\\STAD_data\\STAD_HE\\STAD_test_msi'
    path4 = 'E:\\data\\mss_msi\\STAD_data\\STAD_HE\\STAD_train_msi'
    save_path = 'C:\\Users\\Lenovo\\Documents\\华南统计中心\\关于MSI的项目\\记录结果10_27\\'

    for t in range(0, 4):
        if t == 0:
            path = path1
        elif t == 1:
            path = path2
        elif t == 2:
            path = path3
        else:
            path = path4
        os.chdir(path)
        files = np.sort(os.listdir())
        if t == 0 or t == 2:
            name = ['spot_num', 'patient_ID']
            record_num = pd.DataFrame(columns=name)
            index = 0
            i00 = 1
            for filename in files:
                t1 = time.time()
                answer = circleIdentify(path, filename, ii=19)
                ID = GPD.return_patient_ID(filename)
                answer = [answer, ID]
                record_num.loc[index] = answer
                index = index + 1
                t2 = time.time()
                print('the', i00, 'th trial use', t2 - t1, 'second, and now the t is', t)
                i00 = i00 + 1
            if t == 0:
                record_num.to_csv(save_path + 'MSS_DIF.csv')
            elif t == 2:
                record_num.to_csv(save_path + 'MSI_DIF.csv')
