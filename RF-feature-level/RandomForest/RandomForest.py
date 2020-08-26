# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 16:01:04 2020

@author: lenovo
"""



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
import random
import sklearn.metrics as skmc
import numpy as np
import time as time
## read the feature from address
import matplotlib.pyplot as plt
from sklearn.externals import joblib

# dataset: DX/KR/STAD
# position: path for saving feature
# auc: whether return auc
# seed: seed for partition dataset

position = 'G:\\华统\\MSI和MSS\\数据汇总7\\KR\\'

def RandomForest_MSSMSI(position, dataset, auc=True, seed=12):
    
    if dataset == 'DX' or dataset == 'KR':
        left_thre = 16
        min_split = 17
    elif dataset == 'STAD':
        left_thre = 0
        min_split = 23

    # RGBHSV feature
    file1 = open(position + 'MSI_RGBHSV.csv')
    HsvrgbMSI = pd.read_csv(file1)
    file2 = open(position + 'MSS_RGBHSV.csv')
    HsvrgbMSS = pd.read_csv(file2)
    # Immune cell
    file3 = open(position + 'MSI_immune.csv')
    ImmuneMSI = pd.read_csv(file3)
    file4 = open(position + 'MSS_immune.csv')
    ImmuneMSS = pd.read_csv(file4)
    # differentiation feature
    file5 = open(position + 'MSI_DIF.csv')
    DifMSI = pd.read_csv(file5)
    file6 = open(position + 'MSS_DIF.csv')
    DifMSS = pd.read_csv(file6)
    posi1 = DifMSI[DifMSI.spot_num > 0].index.tolist()
    DifMSI.iloc[posi1, 1] = 1
    posi2 = DifMSS[DifMSS.spot_num > 0].index.tolist()
    DifMSS.iloc[posi2, 1] = 1
    # GMM feature
    file7 = open(position + 'MSI_GMM.csv')
    GmmMSI = pd.read_csv(file7)
    file8 = open(position + 'MSS_GMM.csv')
    GmmMSS = pd.read_csv(file8)
    # Texture feature
    file9 = open(position + 'MSI_Texture2.csv')
    TextureMSI = pd.read_csv(file9)   
    file10 = open(position + 'MSS_Texture2.csv')
    TextureMSS = pd.read_csv(file10)

    WholeMSI = pd.concat([HsvrgbMSI.iloc[:, 1:49],
                           pd.DataFrame(ImmuneMSI.loc[:,'immune_num']),
                           DifMSI.loc[:, 'spot_num'],
                           DifMSI.loc[:, 'patient_ID'],
                           GmmMSI[GmmMSI.columns.difference(['X', 'patient_ID'])],
                           TextureMSI
                           ],
                           axis=1)
    WholeMSS = pd.concat([HsvrgbMSS.iloc[:, 1:49],
                           pd.DataFrame(ImmuneMSS.loc[:, 'immune_num']),
                           DifMSS.loc[:, 'spot_num'],
                           DifMSS.loc[:, 'patient_ID'],
                           GmmMSS[GmmMSS.columns.difference(['X', 'patient_ID'])],
                           TextureMSS
                           ],
                           axis=1)
    # label
    labelMsi = pd.DataFrame([val for val in ['msi'] for i in range(len(WholeMSI))], columns=['label'])
    WholeMSI = pd.concat([WholeMSI, labelMsi], axis=1)
    labelMss = pd.DataFrame([val for val in ['mss'] for i in range(len(WholeMSS))] , columns=['label'])
    WholeMSS = pd.concat([WholeMSS, labelMss], axis=1)
   
    WholeData = pd.concat([WholeMSI, WholeMSS], axis=0, ignore_index=True)

    col_name = WholeData.columns.values.tolist()
    col_name2 = [name.strip()for name in col_name]
    WholeData.columns = col_name2
       
    # data_clean
    WholeData.drop(index=(WholeData.loc[(WholeData['g_var'] == 0)
                                     | (WholeData['r_var'] == 0)
                                     | (WholeData['b_var'] == 0)
                                     | (WholeData['h_var'] == 0)
                                     | (WholeData['s_var'] == 0)
                                     | (WholeData['v_var'] == 0)
                                     | (WholeData['immune_num'] <= left_thre)
                                     | (WholeData['immune_num'] >= 485)
                                     ].index), inplace=True)
        

    WholeData2 = WholeData.drop(['r_var', 'g_var', 'h_var', 'b_var', 's_var', 'v_var'] , axis=1)
    var_vec1 = ['r', 'g', 'b', 'h', 's', 'v']
    var_vec2 = ['mean', '25', 'median', '75']   
    for var1 in var_vec1:
        for var2 in var_vec2:
            var_comp = '_'.join([var1, var2])
            var_de = '_'.join([var1, 'var'])
            WholeData2[var_comp] = WholeData2[var_comp]/WholeData[var_de]

    # add two extra varivable   
    var1 = ["g_kur", "h_75"]
    var2 = ["immune_num", "h_skew" ]  
    add1 = pd.DataFrame(WholeData2[var1[0]] / WholeData2[var1[1]], columns=['one_div'])
    add2 = pd.DataFrame(WholeData2[var2[0]] / WholeData2[var2[1]], columns=['two_div'])
    WholeData3 = pd.concat([WholeData2, add1, add2], axis = 1)

    # we also leave the range variable
    var3 =['r_range','g_range','b_range','h_range','s_range','v_range']   
    WholeData3 = WholeData3.drop(var3, axis = 1)
       
    # begin to divide dataset into different parts
    WholeNameSet = list(WholeData3['patient_ID'].drop_duplicates())
    random.seed(a=seed)
    TrainName = random.sample( WholeNameSet, round(0.7*len(WholeNameSet)) )
    TestName = list(set(WholeNameSet) - set(TrainName))
    TrainData = WholeData3.loc[(WholeData3['patient_ID'].isin(TrainName))]
    TrainData = TrainData.reset_index(drop=True)
    TestData = WholeData3.loc[(WholeData3['patient_ID'].isin(TestName))]
    TestData = TestData.reset_index(drop=True)
    TrainDataPur = TrainData.drop(['patient_ID'], axis=1)
    TestDataPur = TestData.drop(['patient_ID'], axis=1)
    TrainDataX = TrainDataPur.drop(['label'], axis=1)
    TrainDataY = TrainDataPur['label']
    TestDataX = TestDataPur.drop(['label'], axis=1)
    TestDataY = TestDataPur['label']

    # Random forest
    Rfc = RandomForestClassifier(n_estimators=500, random_state=1, criterion='gini', n_jobs=-1, min_samples_split=min_split)
    Rfc.fit(TrainDataX, TrainDataY)
    result = Rfc.predict(TestDataX)
    skmc.confusion_matrix(result, TestDataY)
    # Score
    real = [1 if TestDataY.iloc[i] == 'mss' else 0 for i in range(len(TestDataY))]
    ClassificationResult = []
    RealPatientResult = []
    for j in range(len(TestName)):
        name = TestName[j]
        posi = TestData.loc[(TestData['patient_ID']==name)].index.tolist()
        PredictionResult = [result[i] for i in posi]
        length = PredictionResult.count('mss') / len(PredictionResult)
        ClassificationResult.append(length)
        RealPatientResult.append(real[posi[0]])
    AucValue = skmc.roc_auc_score(RealPatientResult, ClassificationResult)     
    VarName = TrainDataX.columns.values.tolist()
    if auc:
        return AucValue, Rfc, VarName, ClassificationResult, RealPatientResult
    else:
        return Rfc, VarName, ClassificationResult, RealPatientResult


if __name__ == '__main__':

    # Input
    position1 = 'G:\\华统\\MSI和MSS\\数据汇总7\\DX\\'
    position2 = 'G:\\华统\\MSI和MSS\\数据汇总7\\KR\\'
    position3 = 'G:\\华统\\MSI和MSS\\数据汇总7\\STAD\\'

    seed = 12

    aucDX, Rfc, VarName, ClassificationResult, RealPatientResult = \
        RandomForest_MSSMSI(position1, dataset='DX', auc=True, seed=seed)
    aucKR, Rfc, VarName, ClassificationResult, RealPatientResult = \
        RandomForest_MSSMSI(position2, dataset='KR', auc=True, seed=seed)
    aucSTAD, Rfc, VarName, ClassificationResult, RealPatientResult = \
        RandomForest_MSSMSI(position3, dataset='STAD', auc=True, seed=seed)
