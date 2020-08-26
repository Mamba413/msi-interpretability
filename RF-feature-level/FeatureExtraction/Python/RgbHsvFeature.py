# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 10:26:59 2019

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
from scipy import stats
import pandas as pd
import requests as rq
import re
from multiprocessing import Pool
from skimage.measure import label
from sklearn.mixture import GaussianMixture
from GMMFeature import White_balance_Brightness_Adjust

import Get_Patient_ID as GPD



def Get_Color_Feature(Balance_Img, Loc2 ):

    B, G, R = cv.split(Balance_Img)
    r2 = R.astype(int)
    g2 = G.astype(int)
    b2 = B.astype(int)
    
    # we also need to record HSV value
    HSV = cv.cvtColor(Balance_Img, cv.COLOR_BGR2HSV)
    H, S, V = cv.split(HSV)
    #  normalization
    other_chamber_r = [r2[i, j]/255 for i, j in Loc2]
    other_chamber_g = [g2[i, j]/255 for i, j in Loc2]
    other_chamber_b = [b2[i, j]/255 for i, j in Loc2]
    other_chamber_h = [H[i, j]/180 for i, j in Loc2]
    other_chamber_s = [S[i, j]/255 for i, j in Loc2]
    other_chamber_v = [V[i, j]/255 for i, j in Loc2]

    def return_statis(vec_interest):
        vec_mean = np.mean(vec_interest)
        vec_25 = np.percentile(vec_interest, 25)
        vec_median=np.percentile(vec_interest, 50)
        vec_75 = np.percentile(vec_interest, 75)
        vec_range = np.ptp(vec_interest)
        vec_var = np.var(vec_interest)
        vec_skew = stats.skew(vec_interest)
        vec_kur = stats.kurtosis(vec_interest)
        return vec_mean, vec_25, vec_median, vec_75, vec_range, vec_var, vec_skew, vec_kur

    r_vec = return_statis(other_chamber_r)
    g_vec = return_statis(other_chamber_g)
    b_vec = return_statis(other_chamber_b)
    
    h_vec = return_statis(other_chamber_h)
    s_vec = return_statis(other_chamber_s)
    v_vec = return_statis(other_chamber_v)
    
    return r_vec, g_vec, b_vec, h_vec, s_vec, v_vec
    
 

if __name__ == '__main__':
    # Input
    path1 = 'I:\\image_process_msi\\CRC_DX_TEST_MSS'
    path2 = 'I:\\image_process_msi\\CRC_DX_TRAIN_MSS'
    path3 = 'I:\\image_process_msi\\CRC_DX_TEST_MSIMUT'
    path4 = 'I:\\image_process_msi\\CRC_DX_TRAIN_MSIMUT'
    save_path = 'C:\\Users\\Lenovo\\Documents\\华南统计中心\\关于MSI的项目\\记录结果9_30\\'

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
            name = 'r_mean,r_25,r_median,r_75,r_range, r_var,r_skew,r_kur,\
            g_mean,g_25,g_median,g_75,g_range, g_var,g_skew,g_kur,\
            b_mean,b_25,b_median,b_75,b_range, b_var,b_skew,b_kur,\
            h_mean,h_25,h_median,h_75,h_range, h_var,h_skew,h_kur,\
            s_mean,s_25,s_median,s_75,s_range, s_var,s_skew,s_kur,\
            v_mean,v_25,v_median,v_75,v_range, v_var,v_skew,v_kur'

            name = name.replace('\n', '')
            name = name.split(',')
            name.append('patient_ID')
            record_RGBHSV = pd.DataFrame(columns=name)
            index = 0
        i00 = 1
        for filename in files:
            if os.path.splitext(filename)[1] == '.png':
                t1 = time.time()
                
                img, Loc = White_balance_Brightness_Adjust(path, filename , plot = False , save = False , save_path = None)
                result_RGBHSV = np.array(Get_Color_Feature( img, Loc)).reshape(-1)
                patient_id = GPD.return_patient_ID(filename)
                result_RGBHSV=list(result_RGBHSV)
                result_RGBHSV.append(patient_id)
                record_RGBHSV.loc[index] = result_RGBHSV

                index += 1
                
                t2 = time.time()
                print('the', i00, 'th trial use', t2-t1, 'second, and now the t is', t)
                i00 += 1
        if t == 1: 
            record_RGBHSV.to_csv(save_path + 'MSS_RGBHSV.csv')  
        elif t == 3:
            record_RGBHSV.to_csv(save_path + 'MSI_RGBHSV.csv')
