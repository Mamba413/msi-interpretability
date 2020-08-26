# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 17:05:47 2020

@author: lenovo
"""

# 提出文件

from plotnine import *
import pandas as pd
from sklearn.externals import joblib
import numpy as np
import os
import re

path1 = 'G:\\华统\\MSI和MSS\\结果汇总7(跟随数据汇总)\\DX'
path2 = 'G:\\华统\\MSI和MSS\\结果汇总7(跟随数据汇总)\\KR'
path3 = 'G:\\华统\\MSI和MSS\\结果汇总7(跟随数据汇总)\\STAD'

RfcDX = joblib.load(os.path.join(path1, 'DXRfc.pkl'))
VarNameDx = joblib.load(os.path.join(path1, 'VarName.pkl'))
RfcKR = joblib.load(os.path.join(path2, 'KRRfc.pkl'))
VarNameKR = joblib.load(os.path.join(path2, 'VarName.pkl'))
RfcSTAD = joblib.load(os.path.join(path3, 'STADRfc.pkl'))
VarNameSTAD = joblib.load(os.path.join(path3, 'VarName.pkl'))


def BarPlot(Rfc, VarName, Dataset):
    pattern1 = re.compile(r'(\w)_([A-Za-z0-9]*)$')
    pattern2 = re.compile(r'(\w)_([A-Za-z0-9]*)_([A-Za-z0-9]*)$')
    pattern3 = re.compile(r'([A-Za-z]*)\.{1,2}([A-Za-z]*)$')
    pattern4 = re.compile(r'([A-Za-z]*)\.{1,2}([A-Za-z]*)\.([A-Za-z]*)$')
    pattern5 = re.compile(r'([A-Za-z]*)\.{1,2}([A-Za-z]*)\.([A-Za-z]*)\.(.*)$')
    pattern6 = re.compile(r'ROI\.\.1.00\.px\.per\.pixel\.\.(Hematoxylin|Eosin)\.\.(Haralick).*(F\d{1,2})\.$')

    ColName = []
    NewName = []
    for i, item in enumerate(VarName):
        res1 = pattern1.match(item)
        res2 = pattern2.match(item)
        res3 = pattern3.match(item)
        res4 = pattern4.match(item)
        res5 = pattern5.match(item)
        res6 = pattern6.match(item)

        if res1:
            NewName.append(res1.group(1).capitalize() + ' ' + res1.group(2).capitalize())
            ColName.append('GlobalColorFeature')
        elif res2:
            NewName.append(res2.group(1).capitalize() + res2.group(3).capitalize() + ' ' + res2.group(2).capitalize())
            ColName.append('PartialColorFeature')
        elif res3:
            NewName.append(res3.group(1)[0:3] + res3.group(2)[0:3])
            ColName.append('TextureFeature')
        elif res4:
            NewName.append(res4.group(1)[0:3] + res4.group(2).capitalize()[0:3]
                           + res4.group(3).capitalize()[0:3])
            ColName.append('TextureFeature')
        elif res5:
            NewName.append(res5.group(1)[0:3] + res5.group(2).capitalize()[0:3]
                           + res5.group(3).capitalize() + ' ' + res5.group(4).capitalize())
            ColName.append('TextureFeature')
        elif res6:
            NewName.append(res6.group(1).capitalize()[0:3] + res6.group(2).capitalize()[0:4]
                           + res6.group(3).capitalize())
            ColName.append('TextureFeature')
        elif item == 'immune_num':
            NewName.append('ImmuneNum')
            ColName.append('CellNum')
        elif item == 'count':
            NewName.append('TumorNum')
            ColName.append('CellNum')
        elif item == 'spot_num':
            NewName.append('DiffLevel')
            ColName.append('DiffLevel')
        else:
            NewName.append(item)
            ColName.append('GlobalColorFeature')

    # Importance Sort
    importance = Rfc.feature_importances_
    indices = np.argsort(importance)

    VarNameSort = np.array(NewName)[indices]
    ImportanceSort = importance[indices]
    ColNameSort = np.array(ColName)[indices]

    BarplotDf = pd.DataFrame({
        'name': VarNameSort,
        'importance': ImportanceSort,
        'color': ColNameSort
    })

    if Dataset == 'KR':
        for i in range(BarplotDf.shape[0]):
            BarplotDf.loc[i, 'name'] = BarplotDf.loc[i, 'name'] + 'z '
    if Dataset == 'STAD':
        for i in range(BarplotDf.shape[0]):
            BarplotDf.loc[i, 'name'] = ' z' + BarplotDf.loc[i, 'name']
    else:
        pass
    return BarplotDf


BarplotDfDx = BarPlot(RfcDX, VarNameDx, Dataset='DX')
BarplotDfKr = BarPlot(RfcKR, VarNameKR, Dataset='KR')
BarplotDfStad = BarPlot(RfcSTAD, VarNameSTAD, Dataset='STAD')

BarplotDfAll = pd.concat([BarplotDfDx, BarplotDfKr, BarplotDfStad], axis=0, ignore_index=True)
BarplotDfAll['name'] = pd.Categorical(BarplotDfAll['name'], categories=np.array(BarplotDfAll['name']), ordered=True)

# Label
label1 = ['DX' for j in range(137)]
label2 = ['KR' for j in range(182)]
label3 = ['STAD' for j in range(182)]
label = label1 + label2 + label3
BarplotDfAll['label'] = label

(
        ggplot(BarplotDfAll, aes(x='name', y='importance', fill='color'))
        + geom_bar(stat='identity')
        + coord_flip()
        + labs(x='variables', y='score', fill='feature class', nudge_y=1)
        + facet_wrap('~label', scales='free')
        + theme(legend_position='bottom', axis_text=element_text(size=4), axis_title=element_text(size=12),
                strip_text=element_text(size=15), panel_border=element_blank(),
                rect=element_rect(fill='white'), panel_spacing=0.4)
        + scale_fill_manual(values=['#fbb4ae', '#b3cde3', '#ccebc5', '#decbe4', '#fed9a6'])
)
