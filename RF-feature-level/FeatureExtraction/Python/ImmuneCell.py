# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 18:34:03 2019

@author: Lenovo
"""
import os
import cv2 as cv
import matplotlib.pyplot as plt
import math
import numpy as np
from PIL import Image, ImageStat, ImageEnhance
import time
from skimage.measure import label
from skimage import morphology
from multiprocessing import Pool
from scipy import stats
import re
import pandas as pd
import Get_Patient_ID as GPD


# A rough estimation of immunce cells' number
# Input from H section of ImageJ Color Deconvolution


def immune_cell_identi(path, filename, plot=False, convex=False, Low_thre=100, High_thre=170):
    img = cv.imread(path + '\\' + filename)
    # RGB to Gray
    GRAY = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    mask = cv.inRange(GRAY, Low_thre, High_thre)
    # through convex hull
    if convex:
        mask = morphology.convex_hull_object(mask, neighbors=4)

    _, label_c = cv.connectedComponents(mask, connectivity=4)
    label_c = np.array(label_c, dtype='bool')

    area1 = 17
    chull3 = morphology.remove_small_objects(label_c, area1, connectivity=1, in_place=False)
    mask2 = np.array(chull3, dtype='uint8')
    mask3 = np.where(mask2 == 0, mask2, 255)

    area2 = 4
    chull5 = morphology.remove_small_objects(label_c, area2, connectivity=1, in_place=False)

    mask4 = np.array(chull5, dtype='uint8')
    mask5 = np.where(mask4 == 0, mask4, 255)

    # mask6 is the region of interest, where the immune cells are located.
    mask6 = mask5 - mask3
    _, label_connect = cv.connectedComponents(mask6, connectivity=4)
    # the label_num is the number of connectivity domains
    label_num = label_connect.max()

    if plot:
        cv.imshow('original image', img)
        cv.imshow('grayscale image', mask)
        cv.imshow('final image', mask6)
        cv.waitKey(0)

    if convex:
        number = 0
        contours, hireachy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for i, contour in enumerate(contours):
            hull = cv.convexHull(contour)
            ishull = cv.isContourConvex(hull)
            cv.drawContours(img, list(hull), -1, (0, 0, 255), 2)
            if ishull:
                hull = cv.convexHull(contour, returnPoints=False)
                defects = cv.convexityDefects(contour, hull)
                if defects is not None:
                    for j in range(defects.shape[0]):
                        s, e, f, d = defects[j, 0]
                        start = tuple(contour[s][0])
                        end = tuple(contour[e][0])
                        far = tuple(contour[f][0])
                        cv.line(img, start, end, (0, 255, 0), 2)
                        cv.circle(img, far, 5, (0, 0, 255), -1)
                        print(j)
                    number = number + len(defects) - 1
                    label_num = label_num + number
    return label_num


if __name__ == '__main__':
    # Input
    path1 = 'I:\\image_process_msi\\KR_data\\KR_HE_record\\KR_test_mss'
    path2 = 'I:\\image_process_msi\\KR_data\\KR_HE_record\\KR_train_mss'
    path3 = 'I:\\image_process_msi\\KR_data\\KR_HE_record\\KR_test_msi'
    path4 = 'I:\\image_process_msi\\KR_data\\KR_HE_record\\KR_train_msi'
    save_path = 'C:\\Users\\Lenovo\\Documents\\华南统计中心\\关于MSI的项目\\记录结果10_20_KR_immune\\'

    for t in range(0, 4):
        # Set Paths
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
            name = ['immune_num', 'patient_ID']
            record_num = pd.DataFrame(columns=name)
            index = 0
            i00 = 1
            for filename in files:
                t1 = time.time()
                answer = immune_cell_identi(path, filename)
                ID = GPD.return_patient_ID(filename)
                answer = [answer, ID]
                record_num.loc[index] = answer
                index = index + 1
                t2 = time.time()
                print('the', i00, 'th trial use', t2 - t1, 'second, and now the t is', t)
                i00 = i00 + 1
            if t == 0:
                record_num.to_csv(save_path + 'MSS_immune.csv')
            elif t == 3:
                record_num.to_csv(save_path + 'MSI_immune.csv')
