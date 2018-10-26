import numpy as np
np.random.seed(1337) 

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy.misc import comb
from sklearn import preprocessing
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')
from sklearn.neural_network import MLPClassifier
from nolearn.dbn import DBN
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
# import keras
# from keras.datasets import mnist
# from keras.models import Sequential
# from keras.layers.core import Dense, Dropout, Activation
# from keras.optimizers import SGD, Adam, RMSprop
# from keras.utils import np_utils
# from keras.layers import Input, Dense
# from keras.models import Model

#import auto_encoder
import pickle


def mnistNeural():
	print "Enter learning rate"
	learning_rate = input()
	mnist_data = pd.read_csv("mnist_train_bonus.csv",header=None)
	mnist_label=mnist_data[0]
	mnist_data.drop(mnist_data.columns[[0]], axis=1, inplace='True')

	mnist_test = pd.read_csv("mnist_test.csv",header=None)
	mnist_test_label=mnist_test[0]
	mnist_test.drop(mnist_test.columns[[0]], axis=1, inplace='True')

	X_train, X_test = mnist_data, mnist_test
	y_train, y_test = mnist_label, mnist_test_label

	mlp = MLPClassifier(hidden_layer_sizes=(500,250), max_iter=100, alpha=1e-4, solver='lbgfs',
					verbose=0, tol=1e-6, random_state=1,
					learning_rate_init=learning_rate, activation='tanh')

	mlp.fit(X_train, y_train)

	print("Test set score: %f" % mlp.score(X_test, y_test))
	predictions=mlp.predict(X_test)

	filename = 'ffnn_learning_rate.sav'
	pickle.dump(mlp, open(filename, 'wb'))

	p = confusion_matrix(y_test, predictions,labels=[0,1,2,3,4,5,6,7,8,9])
	fig = plt.figure()
	ax = fig.add_subplot(111)
	cax = ax.matshow(p)
	labels=[0,1,2,3,4,5,6,7,8,9]
	plt.title('Confusion matrix of the classifier')
	fig.colorbar(cax)
	ax.set_xticklabels([''] + labels)
	ax.set_yticklabels([''] + labels)
	plt.xlabel('Predicted')
	plt.ylabel('True')
	plt.show()

mnistNeural()