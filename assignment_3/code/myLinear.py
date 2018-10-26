import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy.misc import comb
import warnings
from sklearn.cross_validation import KFold
import os
from python import svmutil
from python import plotroc
from sklearn.metrics import roc_curve
from string import *
import sys, os.path
from sys import argv
from os import system


featureVect = []
labels = []

################### Creating a new file ################################################
def creatingNewFile(file, name, myclass):
	outputFile = open(name,"w")
	inputFile = open(file,"r")
	for line in in_file:
		spline = split(line)

		labels = []
		if spline[0].find(':') == -1:
			labels = split(spline[0],',')
			labels.sort()

		if (labels not in myclass):
			myclass.append(labels)

		if len(labels) == 0:
			outputFile.write("%s %s\n"%(myclass.index(labels), join(spline)))
		else:
			outputFile.write("%s %s\n"%(myclass.index(labels), join(spline[1:])))
	outputFile.close()
	inputFile.close()

def predict(model,x,y):
	print "-----------Starting Predicting using libSvm------------"
	predlabs, predacc, predvals = svmutil.svm_predict(y, x, model)
	print "accuracy: ",predlabs

def cross_validation():
	pathname = "C:\Users\Prerna Singh\Documents\libsvm-3.21/gnuplot-5.0.5/src/gnuplot"
	os.system("python tools/grid.py -v 5 -s 0 -t 0 mnist_train_binary.dat")

def trainingData():
	y, x = svmutil.svm_read_problem('mnist_train_binary.dat')
	featureVect = x
	labels = y
	prob  = svmutil.svm_problem(y, x)
	print "The value of C from cross validation is 0.03125"
	print "Training the model with the new C value"
	param = svmutil.svm_parameter('-t 0 -s 0 -c 0.03125')
	print param
	model = svmutil.svm_train(prob, param)
	print "Saving the model"
	svmutil.svm_save_model("model_linear.model",model)
	return model	

	
###################################### Multi Class ###############################################
def trainingMultiClass(x,y,prob,param):
	print "----------Starting training of the dataset for multiclass-----------------------"
	featureVect = x
	labels = y
	m = svmutil.svm_train(prob, param)
	svmutil.svm_save_model("multi.model",m)
	return m

def testMultiClass(model, x, y):
	predlabs, predacc, predvals = svmutil.svm_predict(y, x, model)
	return predlabs, predacc, predvals

def multi_preprocess(test,train):
	do_test = 1
	classList=[]
	build_new_file(train,"myTrain",classList)
	print "Number of training classes (sets of labels) is %s" % len(classList)
	sys.stdout.flush()
	out = open("trainClass","w")	
	for cl in classList:
		out.write("%s\n" % join(map(lambda(num):("%s"%num),cl),","))
	out.close()
	if (do_test == 1):
		build_new_file(test,"tmp_test",classList)

		
		

######################### Running Linear SVM  for binary ###############################################

m = train()
print "model is trained\n"

yOut, xOut = svmutil.svm_read_problem('mnist_test_binary.dat')
predict(m,xOut,yOut)
print "Prediction is complete\n"





