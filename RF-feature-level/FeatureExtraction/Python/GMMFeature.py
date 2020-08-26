# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 10:31:26 2019

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

import Get_Patient_ID as GPD


def White_balance_Brightness_Adjust(path, filename, plot=False, save=False, save_path=None):
    img = cv.imread(path + '\\' + filename)
    b, g, r = cv.split(img)
    LowWhite = np.array([180, 180, 180])
    HighWhite = np.array([255, 255, 255])
    mask = cv.inRange(img, LowWhite, HighWhite)

    mask2 = np.array(mask, dtype='bool')
    chull3 = morphology.remove_small_objects(mask2, 25, connectivity=1, in_place=False)
    chull4 = np.array(chull3, dtype='uint8')
    chull5 = np.where(chull4 == 0, chull4, 255)

    Loc1 = np.argwhere(chull5 == 255)
    Loc2 = np.argwhere(mask == 0)

    g1 = g.astype(int)
    r1 = r.astype(int)
    b1 = b.astype(int)

    # mean value of cavity
    white_chamber_r = [r1[i][j] for i, j in Loc1]
    white_chamber_g = [g1[i][j] for i, j in Loc1]
    white_chamber_b = [b1[i][j] for i, j in Loc1]

    if len(Loc1) != 0:
        r_mean = int(np.mean(white_chamber_r))
        g_mean = int(np.mean(white_chamber_g))
    else:
        r_mean = 1
        g_mean = 1
        b_mean = 1

    k = (r_mean + g_mean + b_mean) / 3
    kr = k / r_mean
    kg = k / g_mean
    kb = k / b_mean

    # white balance
    r = cv.addWeighted(src1=r, alpha=kr, src2=0, beta=0, gamma=0)
    g = cv.addWeighted(src1=g, alpha=kg, src2=0, beta=0, gamma=0)
    b = cv.addWeighted(src1=b, alpha=kb, src2=0, beta=0, gamma=0)
    balance_img1 = cv.merge([b, g, r])

    # brightness adjustment
    brightness = math.sqrt(k ** 2)
    b_avg = 240  # regulate the brightness in cavity to 240
    balance_img2 = Image.fromarray(np.uint8(balance_img1))
    balance_img3 = ImageEnhance.Brightness(balance_img2).enhance(b_avg / brightness)
    balance_img4 = np.uint8(balance_img3)
    # draw plot
    if plot:
        cv.imshow('original image', img)
        cv.imshow('balanced image', balance_img1)
        cv.imshow('balanced and brightness adjusted image ', balance_img4)
        cv.waitKey(0)

    # save adjusted image
    if save:
        balance_img3.save(save_path + '\\' + filename)

    return balance_img4, Loc2


# the GMM feature abstraction
# Balance_Img: the input comes from White_balance_Brightness_Adjust function

def Get_GMM_Feature(Balance_Img, Loc2):
    # initialiazation
    B, G, R = cv.split(Balance_Img)
    R = R.astype(int)
    G = G.astype(int)
    B = B.astype(int)
    BGR_record = [[B[i, j], G[i, j], R[i, j]] for i, j in Loc2]
    BGR_record = np.array(BGR_record)
    # transform into HSV, so that we can get the same statistic in HSV
    hsv_bal_img = cv.cvtColor(Balance_Img, cv.COLOR_BGR2HSV)
    H, S, V = cv.split(hsv_bal_img)
    # put that as input into GMM model
    gmm = GaussianMixture(n_components=3, random_state=1).fit(BGR_record)
    labels = gmm.predict(BGR_record)
    labels_0_posi = np.argwhere(labels == 0)
    labels_1_posi = np.argwhere(labels == 1)
    labels_2_posi = np.argwhere(labels == 2)
    # the counterpart position(in pair) of these sections
    Loc_label_0 = np.array([Loc2[i] for i in labels_0_posi])[:, 0, :]
    Loc_label_1 = np.array([Loc2[i] for i in labels_1_posi])[:, 0, :]
    Loc_label_2 = np.array([Loc2[i] for i in labels_2_posi])[:, 0, :]
    # the corresponding value in different section
    label_0_r = [R[i, j] / 255 for i, j in Loc_label_0]
    label_1_r = [R[i, j] / 255 for i, j in Loc_label_1]
    label_2_r = [R[i, j] / 255 for i, j in Loc_label_2]

    result1 = [np.mean(label_0_r), np.mean(label_1_r), np.mean(label_2_r)]
    sort_result = np.argsort(result1)
    Loc_all = [Loc_label_0, Loc_label_1, Loc_label_2]

    real_Loc1 = Loc_all[sort_result[2]]
    real_Loc2 = Loc_all[sort_result[1]]
    real_Loc3 = Loc_all[sort_result[0]]

    # RGB record
    label_0_r = [R[i, j] / 255 for i, j in real_Loc1]
    label_0_g = [G[i, j] / 255 for i, j in real_Loc1]
    label_0_b = [B[i, j] / 255 for i, j in real_Loc1]
    label_1_r = [R[i, j] / 255 for i, j in real_Loc2]
    label_1_g = [G[i, j] / 255 for i, j in real_Loc2]
    label_1_b = [B[i, j] / 255 for i, j in real_Loc2]
    label_2_r = [R[i, j] / 255 for i, j in real_Loc3]
    label_2_g = [G[i, j] / 255 for i, j in real_Loc3]
    label_2_b = [B[i, j] / 255 for i, j in real_Loc3]

    # HSV record
    label_0_h = [H[i, j] / 180 for i, j in real_Loc1]
    label_0_s = [S[i, j] / 255 for i, j in real_Loc1]
    label_0_v = [V[i, j] / 255 for i, j in real_Loc1]
    label_1_h = [H[i, j] / 180 for i, j in real_Loc2]
    label_1_s = [S[i, j] / 255 for i, j in real_Loc2]
    label_1_v = [V[i, j] / 255 for i, j in real_Loc2]
    label_2_h = [H[i, j] / 180 for i, j in real_Loc3]
    label_2_s = [S[i, j] / 255 for i, j in real_Loc3]
    label_2_v = [V[i, j] / 255 for i, j in real_Loc3]

    def statistic_want(input_vec, function=None):
        result = []
        for item in input_vec:
            result.append(function(item))
        return result

    RGB_mean_0_record = statistic_want([label_0_r, label_0_g, label_0_b], np.mean)
    RGB_var_0_record = statistic_want([label_0_r, label_0_g, label_0_b], np.var)
    RGB_skew_0_record = statistic_want([label_0_r, label_0_g, label_0_b], stats.skew)
    RGB_kur_0_record = statistic_want([label_0_r, label_0_g, label_0_b], stats.kurtosis)

    RGB_mean_1_record = statistic_want([label_1_r, label_1_g, label_1_b], np.mean)
    RGB_var_1_record = statistic_want([label_1_r, label_1_g, label_1_b], np.var)
    RGB_skew_1_record = statistic_want([label_1_r, label_1_g, label_1_b], stats.skew)
    RGB_kur_1_record = statistic_want([label_1_r, label_1_g, label_1_b], stats.kurtosis)

    RGB_mean_2_record = statistic_want([label_2_r, label_2_g, label_2_b], np.mean)
    RGB_var_2_record = statistic_want([label_2_r, label_2_g, label_2_b], np.var)
    RGB_skew_2_record = statistic_want([label_2_r, label_2_g, label_2_b], stats.skew)
    RGB_kur_2_record = statistic_want([label_2_r, label_2_g, label_2_b], stats.kurtosis)

    HSV_mean_0_record = statistic_want([label_0_h, label_0_s, label_0_v], np.mean)
    HSV_var_0_record = statistic_want([label_0_h, label_0_s, label_0_v], np.var)
    HSV_skew_0_record = statistic_want([label_0_h, label_0_s, label_0_v], stats.skew)
    HSV_kur_0_record = statistic_want([label_0_h, label_0_s, label_0_v], stats.kurtosis)

    HSV_mean_1_record = statistic_want([label_1_h, label_1_s, label_1_v], np.mean)
    HSV_var_1_record = statistic_want([label_1_h, label_1_s, label_1_v], np.var)
    HSV_skew_1_record = statistic_want([label_1_h, label_1_s, label_1_v], stats.skew)
    HSV_kur_1_record = statistic_want([label_1_h, label_1_s, label_1_v], stats.kurtosis)

    HSV_mean_2_record = statistic_want([label_2_h, label_2_s, label_2_v], np.mean)
    HSV_var_2_record = statistic_want([label_2_h, label_2_s, label_2_v], np.var)
    HSV_skew_2_record = statistic_want([label_2_h, label_2_s, label_2_v], stats.skew)
    HSV_kur_2_record = statistic_want([label_2_h, label_2_s, label_2_v], stats.kurtosis)

    return (np.array([RGB_mean_0_record, RGB_var_0_record, RGB_skew_0_record, RGB_kur_0_record,
                      RGB_mean_1_record, RGB_var_1_record, RGB_skew_1_record, RGB_kur_1_record,
                      RGB_mean_2_record, RGB_var_2_record, RGB_skew_2_record, RGB_kur_2_record,
                      HSV_mean_0_record, HSV_var_0_record, HSV_skew_0_record, HSV_kur_0_record,
                      HSV_mean_1_record, HSV_var_1_record, HSV_skew_1_record, HSV_kur_1_record,
                      HSV_mean_2_record, HSV_var_2_record, HSV_skew_2_record, HSV_kur_2_record]).flatten())


if __name__ == '__main__':
    # input
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
            name = '''r_mean_l0,g_mean_l0,b_mean_l0,r_var_l0,g_var_l0,b_var_l0,r_skew_l0,g_skew_l0,b_skew_l0,r_kur_l0,g_kur_l0,b_kur_l0,
            r_mean_l1,g_mean_l1,b_mean_l1,r_var_l1,g_var_l1,b_var_l1,r_skew_l1,g_skew_l1,b_skew_l1,r_kur_l1,g_kur_l1,b_kur_l1,
            r_mean_l2,g_mean_l2,b_mean_l2,r_var_l2,g_var_l2,b_var_l2,r_skew_l2,g_skew_l2,b_skew_l2,r_kur_l2,g_kur_l2,b_kur_l2,
            h_mean_l0,s_mean_l0,v_mean_l0,h_var_l0,s_var_l0,v_var_l0,h_skew_l0,s_skew_l0,v_skew_l0,h_kur_l0,s_kur_l0,v_kur_l0,
            h_mean_l1,s_mean_l1,v_mean_l1,h_var_l1,s_var_l1,v_var_l1,h_skew_l1,s_skew_l1,v_skew_l1,h_kur_l1,s_kur_l1,v_kur_l1,
            h_mean_l2,s_mean_l2,v_mean_l2,h_var_l2,s_var_l2,v_var_l2,h_skew_l2,s_skew_l2,v_skew_l2,h_kur_l2,s_kur_l2,v_kur_l2'''
            name = name.replace('\n', '')
            name = name.split(',')
            name.append('patient_ID')
            record_GMM = pd.DataFrame(columns=name)
            index = 0
        i00 = 1
        for filename in files:
            if os.path.splitext(filename)[1] == '.png':
                t1 = time.time()
                img, Loc = White_balance_Brightness_Adjust(path, filename, plot=False, save=False, save_path=None)
                result_GMM = Get_GMM_Feature(img, Loc)
                patient_id = GPD.return_patient_ID(filename)
                result_GMM = list(result_GMM)
                result_GMM.append(patient_id)
                record_GMM.loc[index] = result_GMM
                index = index + 1
                t2 = time.time()
                print('the', i00, 'th trial use', t2 - t1, 'second, and now the t is', t)
                i00 = i00 + 1
        if t == 1:
            record_GMM.to_csv(save_path + 'MSS_GMM.csv')
        elif t == 3:
            record_GMM.to_csv(save_path + 'MSI_GMM.csv')
