import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy.misc import comb
from sklearn import preprocessing
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn import cross_validation
#from sklearn.model_selection import KFold
from sklearn import grid_search
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn import svm
from sklearn.externals import joblib


data2 = pd.read_csv("C:\Users\Prerna Singh\Documents\machine\mnist_train.csv",header=None)
#print data2
#testData1=data2.copy()
print len(data2.columns)

Y = data2[0][:20000]
print len(Y)
#print Y
Ytest = Y
trainData=data2[:20000]
#print Ytest
trainData.drop(data2.columns[[0]], axis=1, inplace=True);
print len(trainData)
#data2=data2
#print data2
print len(data2)
print len(Ytest)
trainData.to_csv("C:\Users\Prerna Singh\Documents\machine\mnist_train_subset.csv", index=False, header=False)

trainData = np.array(trainData);
trainLabel = np.array(Y);
kf = cross_validation.KFold(len(trainData), n_folds=5, shuffle=True, random_state=4)
for train_index, test_index in kf:
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = trainData[train_index], trainData[test_index]
    y_train, y_test = trainLabel[train_index], trainLabel[test_index]


t=OneVsRestClassifier(LinearSVC(random_state=0))#.fit(trainData, Y)
#out=t.predict(data2)
c_range = np.logspace(0, 4, 10)
lrgs = grid_search.GridSearchCV(estimator=t, param_grid=dict(estimator__C=c_range), n_jobs=1)

[lrgs.fit(trainData[train_indices], trainLabel[train_indices]).score(trainData[test_indices],trainLabel[test_indices])
for train_indices, test_indices in kf]

print lrgs.best_estimator_


c_best = 2.7825594022071245

t=OneVsRestClassifier(LinearSVC(random_state=0,C=c_best))

t.fit(trainData, trainLabel)

joblib.dump(lrgs.best_estimator_, 'bestModel')

lrgs.best_estimator_.fit(trainData, trainLabel)

def loss(predictVal,ExpectVal):
	squared_loss=0;
	loss = (ExpectVal - predictVal)**2

	squared_loss = np.sum(loss)
	print squared_loss/(2*len(predictVal))

mean_squared_error(testLabels, out)**0.5



######################################################################RBF Multiclass ##############################################
svc = svm.SVC(kernel='rbf',random_state=0)

data = pd.read_csv("mnist_train_subset.csv",header=None)
print len(data)
labels = pd.read_csv("mnist_train_subset_labels.csv",header=None)
print len(labels)

trainData = np.array(data);
trainLabel = np.array(labels);
kf = cross_validation.KFold(len(trainData), n_folds=5, shuffle=True, random_state=4)
for train_index, test_index in kf:
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = trainData[train_index], trainData[test_index]
    y_train, y_test = trainLabel[train_index], trainLabel[test_index]

c_range = np.logspace(0, 4, 10)
gamma= np.logspace(0, -13, 4)
lrgs2 = grid_search.GridSearchCV(estimator=t, param_grid=dict(estimator__C=c_range, estimator__gamma=gamma), n_jobs=1)

[lrgs2.fit(trainData[train_indices], trainLabel[train_indices]).score(trainData[test_indices],trainLabel[test_indices])
for train_indices, test_indices in kf]


t=OneVsRestClassifier(svm.SVC(kernel='rbf',random_state=0,C=10.0,gamma=0.01))
t.fit(trainData, trainLabel)

joblib.dump(lrgs2.best_estimator_, 'bestModelRbf')

lrgs2.best_estimator_.fit(trainData, trainLabel)

print "RBF Model is successfully saved."