#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 11:51:30 2017

@author: rstanek
"""

#import libraries
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#load data set
noun_file = 'TrainSample.csv'
names = ['API','Surf_X','Surf_Y','Date_Drilling','Date_Completion','Date_Production','Lateral_Length','Depth_TVD_PPLS','Erosion_PPLS','Pressure_PPLS','TOC_PPLS','Vcarb_PPLS','Vsand_PPLS','Vclay_PPLS','PR_PPLS','YM_PPLS','RHOB_PPLS','Res_PPLS','GR_PPLS','DT_PPLS','DTs_PPLS','Temperature','Temp_Anomaly','S3Tect_PPLS','S3_contrast_PPLS','Heat_Flow','Zone','Nbr_Stages','Frac_Gradient','Proppant_Designed','Proppant_in_Formation','Avg_Breakdown_Pressure','Avg_Treating_Pressure','Max_Treating_pressure','Min_Treating_Pressure','Avg_Rate_Slurry','Max_Rate_Slurry','Min_Rate_Slurry','ShutInPressure_Fil','ShutInPressure_Initial','ISIP','Shot_Density','Shot_Total','Proppant_per_ft','Stage_Spacing','GasCum360','OilCum360']
dataset = pandas.read_csv('TrainSample.csv', sep=";", header = None , names = names, skiprows=1, decimal=",")
dataset.set_index('API')

#summarise the data set
print(dataset.describe())

#look at the class distribution
print(dataset.groupby('GasCum360').size())

#----------------------------------------------------
##Make some plots
## box and whisker plots
#dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
#plt.show()
#
##histograms
#dataset.hist()
#plt.show()
#
##some scatter plots (2 way interactions)
#scatter_matrix(dataset)
#plt.show()
#---------------------------------------------------
#--------------------------------------------------
#Define a Validation Set
# here we are training the model with 80% of the data
# and leaving 20% (30 observations) for validation
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
#--------------------------------------------------

#--------------------------------------------------
#Set up 10-fold cross validation
seed = 7
scoring = 'accuracy'
#-------------------------------------------------

#-------------------------------------------------
#Models to Evaluation
# Logistic Regression
# Linear Discriminant Analysis
# K-Nearest Neighbor
# Classification and Regression Trees
# Gaussian Naive Bayes
# Support Vector Machines

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)


# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))



import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from string import ascii_letters
import seaborn as sns

column_names = ['API','Surf_X','Surf_Y','Date_Drilling','Date_Completion','Date_Production','Lateral_Length','Depth_TVD_PPLS','Erosion_PPLS','Pressure_PPLS','TOC_PPLS','Vcarb_PPLS','Vsand_PPLS','Vclay_PPLS','PR_PPLS','YM_PPLS','RHOB_PPLS','Res_PPLS','GR_PPLS','DT_PPLS','DTs_PPLS','Temperature','Temp_Anomaly','S3Tect_PPLS','S3_contrast_PPLS','Heat_Flow','Zone','Nbr_Stages','Frac_Gradient','Proppant_Designed','Proppant_in_Formation','Avg_Breakdown_Pressure','Avg_Treating_Pressure','Max_Treating_pressure','Min_Treating_Pressure','Avg_Rate_Slurry','Max_Rate_Slurry','Min_Rate_Slurry','ShutInPressure_Fil','ShutInPressure_Initial','ISIP','Shot_Density','Shot_Total','Proppant_per_ft','Stage_Spacing','GasCum360','OilCum360']

#DataFrame de du document complet avec pour indice principal 'API'
df = pd.read_csv('TrainSample.csv', sep=";", header = None , names = column_names, skiprows=1, decimal=",")
df.set_index('API')


df=df.dropna(subset=['GasCum360'])
X=np.linspace(1,np.shape(Y)[0],num=np.shape(Y)[0])
X = X.reshape(len(df.index), 1)
Y = Y.reshape(len(df.index), 1)
regr = linear_model.LinearRegression()
regr.fit(X, Y)

#Affichage du scatterplot de 'ShutInPressure_Initial' en fonction de 'GasCum360'
plt.scatter(X, sorted(Y),  color='black')
plt.plot(X, regr.predict(X), color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()




from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4).fit(Y)
k=kmeans.predict(Y)
df2 = pd.DataFrame(k, columns=['Cluster'])
df3 = pd.concat([df,df2],axis=1, join_axes=[df.index])
df3 = df3.sort_values('GasCum360')


fig = plt.figure()
ax = fig.add_subplot(111)
Y=df3.GasCum360.values
X=np.linspace(1,np.shape(Y)[0],num=np.shape(Y)[0])
Cluster = df3.Cluster.values
Cluster = Cluster.reshape(len(df3.index), 1)
X = X.reshape(len(df3.index), 1)
Y = Y.reshape(len(df3.index), 1)
Cluster = df3.Cluster.values
scatter = ax.scatter(X,Y,c=Cluster)
plt.colorbar(scatter)

fig.show()