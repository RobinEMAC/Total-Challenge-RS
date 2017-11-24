#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 20:55:36 2017

@author: rstanek
"""

import csv
import numpy as np
noun_file = 'TrainSample.csv'
data = list(csv.reader(open(noun_file , 'rt', encoding="utf-8"), delimiter = ';'))
#print(data)
data2 = []
for i in range(1,len(data)):
    data2.append([float(x.replace(',','.')) for x in data[i] if (('/' not in x)and(x!=''))])
#print(data2)
    
data3 = []
for i in range(len(data2)):
    if len(data2[i])==44:
        data3.append(data2[i])

data4 = np.zeros(np.shape(data3))
for i in range(np.shape(data3)[0]):
    for j in range(np.shape(data3)[1]):
        data4[i,j]=data3[i][j]
print(data4)
#print(data3)

#corr=[]
#for i in range(len(data3)):
#    l=[]
#    for j in range(len(data3[i])):
#        l.append(np.corrcoef(data3[i][j],data3[i][i])[0,1])
#    corr.append(l)
    
corr = np.zeros((np.shape(data4)[1],np.shape(data4)[1]))
for i in range(np.shape(data4)[1]):
    for j in range(np.shape(data4)[1]):
        corr[i,j]=np.corrcoef(data4[:,i],data4[:,j])[0,1]
#print(corr)

import seaborn as sns
column_names = ['API','Surf_X','Surf_Y','Lateral_Length','Depth_TVD_PPLS','Erosion_PPLS','Pressure_PPLS','TOC_PPLS','Vcarb_PPLS','Vsand_PPLS','Vclay_PPLS','PR_PPLS','YM_PPLS','RHOB_PPLS','Res_PPLS','GR_PPLS','DT_PPLS','DTs_PPLS','Temperature','Temp_Anomaly','S3Tect_PPLS','S3_contrast_PPLS','Heat_Flow','Zone','Nbr_Stages','Frac_Gradient','Proppant_Designed','Proppant_in_Formation','Avg_Breakdown_Pressure','Avg_Treating_Pressure','Max_Treating_pressure','Min_Treating_Pressure','Avg_Rate_Slurry','Max_Rate_Slurry','Min_Rate_Slurry','ShutInPressure_Fil','ShutInPressure_Initial','ISIP','Shot_Density','Shot_Total','Proppant_per_ft','Stage_Spacing','GasCum360','OilCum360']

import pandas as pd

corr= pd.DataFrame(corr, columns=column_names, index=column_names)
# plot the heatmap
sns.heatmap(corr, 
        xticklabels=column_names,
        yticklabels=column_names)

cmap = cmap=sns.diverging_palette(5, 250, as_cmap=True)

def magnify():
    return [dict(selector="th",
                 props=[("font-size", "7pt")]),
            dict(selector="td",
                 props=[('padding', "0em 0em")]),
            dict(selector="th:hover",
                 props=[("font-size", "12pt")]),
            dict(selector="tr:hover td:hover",
                 props=[('max-width', '200px'),
                        ('font-size', '12pt')])]

corr.style.background_gradient(cmap, axis=1)\
    .set_properties(**{'max-width': '80px', 'font-size': '10pt'})\
    .set_caption("Hover to magify")\
    .set_precision(2)\
    .set_table_styles(magnify())
    
for i in range(np.shape(corr)[0]):
    for j in range(np.shape(corr)[0]):
        if np.abs(corr[i,j])<0.5:
            corr[i,j]=0


import csv
import numpy as np
noun_file = 'TrainSample.csv'
#Toutes les données avec la description
data = list(csv.reader(open(noun_file , 'rt', encoding="utf-8"), delimiter = ';'))

#Retirer la description
data2 = []
for i in range(1,len(data)):
    data2.append([float(x.replace(',','.')) for x in data[i] if (('/' not in x)and(x!=''))])

#Prendre toutes les données sans les lignes vides    
data3 = []
for i in range(len(data2)):
    if len(data2[i])==44:
        data3.append(data2[i])

#Mettre en tableau 2D
data4 = np.zeros(np.shape(data3))
for i in range(np.shape(data3)[0]):
    for j in range(np.shape(data3)[1]):
        data4[i,j]=data3[i][j]

#Mettre en DataFrame
import pandas as pd
column_names = ['API','Surf_X','Surf_Y','Lateral_Length','Depth_TVD_PPLS','Erosion_PPLS','Pressure_PPLS','TOC_PPLS','Vcarb_PPLS','Vsand_PPLS','Vclay_PPLS','PR_PPLS','YM_PPLS','RHOB_PPLS','Res_PPLS','GR_PPLS','DT_PPLS','DTs_PPLS','Temperature','Temp_Anomaly','S3Tect_PPLS','S3_contrast_PPLS','Heat_Flow','Zone','Nbr_Stages','Frac_Gradient','Proppant_Designed','Proppant_in_Formation','Avg_Breakdown_Pressure','Avg_Treating_Pressure','Max_Treating_pressure','Min_Treating_Pressure','Avg_Rate_Slurry','Max_Rate_Slurry','Min_Rate_Slurry','ShutInPressure_Fil','ShutInPressure_Initial','ISIP','Shot_Density','Shot_Total','Proppant_per_ft','Stage_Spacing','GasCum360','OilCum360']
df = pd.DataFrame(data4, columns=column_names)
print(df)

#corr=[]
#for i in range(len(data3)):
#    l=[]
#    for j in range(len(data3[i])):
#        l.append(np.corrcoef(data3[i][j],data3[i][i])[0,1])
#    corr.append(l)

#Matrice de corrélation entre les caractéristiques
corr = np.zeros((np.shape(data4)[1],np.shape(data4)[1]))
for i in range(np.shape(data4)[1]):
    for j in range(np.shape(data4)[1]):
        corr[i,j]=np.corrcoef(data4[:,i],data4[:,j])[0,1]

import seaborn as sns
corr_df= pd.DataFrame(corr, columns=column_names, index=column_names)

# plot the heatmap
sns.heatmap(corr_df, 
        xticklabels=column_names,
        yticklabels=column_names)

#Embellir la heatmap
cmap = sns.diverging_palette(5, 250, as_cmap=True)

def magnify():
    return [dict(selector="th",
                 props=[("font-size", "7pt")]),
            dict(selector="td",
                 props=[('padding', "0em 0em")]),
            dict(selector="th:hover",
                 props=[("font-size", "12pt")]),
            dict(selector="tr:hover td:hover",
                 props=[('max-width', '200px'),
                        ('font-size', '12pt')])]

corr_df.style.background_gradient(cmap, axis=1)\
    .set_properties(**{'max-width': '80px', 'font-size': '10pt'})\
    .set_caption("Hover to magify")\
    .set_precision(2)\
    .set_table_styles(magnify())

#Tri sur les différentes corrélation
for i in range(np.shape(corr)[0]):
    for j in range(np.shape(corr)[0]):
        if np.abs(corr[i,j])<0.5:
            corr[i,j]=0

corr_df= pd.DataFrame(corr, columns=column_names, index=column_names)

corr_df.style.background_gradient(cmap, axis=1)\
    .set_properties(**{'max-width': '80px', 'font-size': '10pt'})\
    .set_caption("Hover to magify")\
    .set_precision(2)\
    .set_table_styles(magnify())
    
#Tri sur les différentes corrélation
for i in range(np.shape(corr)[0]):
    for j in range(np.shape(corr)[0]):
        if np.abs(corr[i,j])<0.5:
            corr[i,j]=0

from sklearn import linear_model
