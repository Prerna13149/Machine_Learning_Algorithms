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
import math
from scipy import exp
from scipy.linalg import eigh
import csv
from scipy.spatial.distance import pdist, squareform

def Hbeta(D = np.array([]), beta = 1.0):
	# Compute P-row and corresponding perplexity
	P = np.exp(-D.copy() * beta);
	sumP = sum(P);
	H = np.log(sumP) + beta * np.sum(D * P) / sumP;
	P = P / sumP;
	return H, P;
def x2p(X = np.array([]), tol = 1e-5, perplexity = 30.0):
	# Initialize some variables
	print "Computing pairwise distances..."
	(n, d) = X.shape;
	sum_X = np.sum(np.square(X), 1);
	D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X);
	P = np.zeros((n, n));
	beta = np.ones((n, 1));
	logU = np.log(perplexity);

	# Loop over all datapoints
	for i in range(n):

		# Print progress
		if i % 500 == 0:
			print "Computing P-values for point ", i, " of ", n, "..."

		# Compute the Gaussian kernel and entropy for the current precision
		betamin = -np.inf;
		betamax =  np.inf;
		Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))];
		(H, thisP) = Hbeta(Di, beta[i]);

		# Evaluate whether the perplexity is within tolerance
		Hdiff = H - logU;
		tries = 0;
		while np.abs(Hdiff) > tol and tries < 50:

			# If not, increase or decrease precision
			if Hdiff > 0:
				betamin = beta[i].copy();
				if betamax == np.inf or betamax == -np.inf:
					beta[i] = beta[i] * 2;
				else:
					beta[i] = (beta[i] + betamax) / 2;
			else:
				betamax = beta[i].copy();
				if betamin == np.inf or betamin == -np.inf:
					beta[i] = beta[i] / 2;
				else:
					beta[i] = (beta[i] + betamin) / 2;

			# Recompute the values
			(H, thisP) = Hbeta(Di, beta[i]);
			Hdiff = H - logU;
			tries = tries + 1;

		# Set the final row of P
		P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP;

	# Return final P-matrix
	print "Mean value of sigma: ", np.mean(np.sqrt(1 / beta));
	return P;

def pca(X = np.array([]), no_dims = 50):
	print "Preprocessing the data using PCA..."
	(n, d) = X.shape;
	X = X - np.tile(np.mean(X, 0), (n, 1));
	(l, M) = np.linalg.eig(np.dot(X.T, X));
	Y = np.dot(X, M[:,0:no_dims]);
	return Y;

def tsne(X = np.array([]), no_dims = 2, initial_dims = 784, perplexity = 30.0):

	# Check inputs
	if isinstance(no_dims, float):
		print "Error: array X should have type float.";
		return -1;
	if round(no_dims) != no_dims:
		print "Error: number of dimensions should be an integer.";
		return -1;

	# Initialize variables
	X = pca(X, initial_dims).real;
	(n, d) = X.shape;
	max_iter = 1000;
	initial_momentum = 0.5;
	final_momentum = 0.8;
	eta = 500;
	min_gain = 0.01;
	Y = np.random.randn(n, no_dims);
	dY = np.zeros((n, no_dims));
	iY = np.zeros((n, no_dims));
	gains = np.ones((n, no_dims));

	# Compute P-values
	P = x2p(X, 1e-5, perplexity);
	P = P + np.transpose(P);
	P = P / np.sum(P);
	P = P * 4;									# early exaggeration
	P = np.maximum(P, 1e-12);

	# Run iterations
	for iter in range(max_iter):

		# Compute pairwise affinities
		sum_Y = np.sum(np.square(Y), 1);
		num = 1 / (1 + np.add(np.add(-2 * np.dot(Y, Y.T), sum_Y).T, sum_Y));
		num[range(n), range(n)] = 0;
		Q = num / np.sum(num);
		Q = np.maximum(Q, 1e-12);

		# Compute gradient
		PQ = P - Q;
		for i in range(n):
			dY[i,:] = np.sum(np.tile(PQ[:,i] * num[:,i], (no_dims, 1)).T * (Y[i,:] - Y), 0);

		# Perform the update
		if iter < 20:
			momentum = initial_momentum
		else:
			momentum = final_momentum
		gains = (gains + 0.2) * ((dY > 0) != (iY > 0)) + (gains * 0.8) * ((dY > 0) == (iY > 0));
		gains[gains < min_gain] = min_gain;
		iY = momentum * iY - eta * (gains * dY);
		Y = Y + iY;
		Y = Y - np.tile(np.mean(Y, 0), (n, 1));

		# Compute current value of cost function
		if (iter + 1) % 10 == 0:
			C = np.sum(P * np.log(P / Q));
			print "Iteration ", (iter + 1), ": error is ", C

		# Stop lying about P-values
		if iter == 100:
			P = P / 4;

	# Return solution
	return Y;


def normalize(data):
    filter_df = data.copy()
    std_scale = preprocessing.MinMaxScaler().fit(data)
    df_std = std_scale.transform(data)
    k = 0
    print df_std
    for i in filter_df.columns:
        for j in range(len(data)):
            filter_df[i].values[j] = df_std[j][k]
        k += 1
    return filter_df

def euclid_distance(a, b):
    #b=np.linalg.norm(a-b)
    return pow(pow(a[0]-b[0],2) + pow(a[1]-b[1],2),0.5)
    
def distance(xList, point):
    distList=[];
    for i in range(len(xList)):
        p=euclid_distance(xList[i], point)
        distList.append(p)
    #print distList
    finalList=np.sort(distList)
    l=[];
    for i in range(4):
        itemindex = np.where(distList==finalList[i])
        #print itemindex
        l.append(itemindex)
    
    #print "sorted list is-------------"
    #print finalList[:8]
    #k =  np.count_nonzero(finalList==finalList[2])
    #print k
    #print l;
    return l

def knn(train, point,labels):
    new_list = []
    l=distance(train, point)
    #print l
    for i in l:
        for j in range(len(i[0])):
            new_list.append(i[0][j])
    #print new_list
    newIdx=new_list[1:4]
    #print newIdx
    newY=[]
    for i in newIdx:
        newY.append(labels[i])
    print newY
    return newY

def make_tuples(data):
    tuples = []
    for i in range(len(data)):
        tuple_2 = []
        for j in data.columns:
            tuple_2.append(data[j][i])
        tuples.append(tuple_2)
    return tuples


    
def exp_matrix(kernel,gamma,var):
    new_kernel = kernel
    for i in range(len(kernel)):
        for j in range(len(kernel[0])):
            new_kernel[i][j] = exp(-gamma*kernel[i][j]/2*var)
    return new_kernel


# In[276]:

def find_kernel(X, gamma,components):
    variance = np.var(X)
    variance = np.mean(variance)
    #print len(X.columns)
    euclid_distance = pdist(X, 'sqeuclidean')
    mat_distance = squareform(euclid_distance)
    kernel = exp_matrix(mat_distance,gamma,variance)
    #kernel = math.exp(-gamma * mat_distance)
    N = kernel.shape[0]
    one_n = np.ones((N,N)) / N
    kernel = kernel - one_n.dot(kernel) - kernel.dot(one_n) + one_n.dot(kernel).dot(one_n)
    
    eigvals, eigvecs = eigh(kernel)

    # Obtaining the i eigenvectors that corresponds to the i highest eigenvalues.
    X_pc = np.column_stack((eigvecs[:,-i] for i in range(1,components+1)))

    return X_pc
    
    




def knnForAll(data):
    #labels=data[0]
    #data2=normalize(newTrain)
    #data2.drop(data2[[0]], inplace=True, axis=1)
    #train = make_tuples(data2)
    #train = pca(data2,784);
    train = find_kernel(data,1,2)
    x=[i[0] for i in train]
    y=[i[1] for i in train]
    final=[];
    for i in range(len(x)):
        point=[];
        point.append(x[i]);
        point.append(y[i]);
        newY=[]
        #newLabels=[];
        newY=knn(train, point, labels)
        final.append(newY)
    return final

    
    
#newTrain=pd.read_csv("C:\Users\Prerna Singh\Documents\machine\mnist_train_bonus.csv",header=None)
newtrain = pd.read_csv("mnist_train_bonus.csv",header=None)
#newTrain = newTrain.iloc[np.random.permutation(len(newTrain))]
labels = newtrain[0]

newtrain.drop(newtrain.columns[0],axis=1)

newtrain = normalize(newtrain)



kernel = find_kernel(np.array(newtrain),1,2)

print kernel

labels=newtrain[0]
print kernel.shape

test = pd.read_csv("mnist_test_bonus.csv",header=None)

labels_test = test[0]

test_2 = test.drop(test.columns[0],axis=1)

def knnForAllOne(data,labels):
    
    
    train = find_kernel(data,1,2);
    #train = kernel
    x=[i[0] for i in train]
    y=[i[1] for i in train]
    final=[];
    for i in range(len(x)):
        point=[];
        point.append(x[i]);
        point.append(y[i]);
        newY=[]
        #newLabels=[];
        newY=knn(train, point, labels)
        final.append(newY)
    return final

finalOut=knnForAllOne(test_2,labels_test)


#finalOut = knnForAll(newTrain)
#outLabels = newtrain[0]
#df = pd.DataFrame(finalOut, columns=['dist1','dist2', 'dist3'])
#df.to_csv("OutputKnnTrain.csv")

##############finding maximum frequency############################# 
outLabels=[]
for i in finalOut:
    counts = np.bincount(i)
    t = np.argmax(counts)
    outLabels.append(t)
print outLabels

print len(outLabels)
df1 = pd.DataFrame(outLabels, columns=['labels'])
df1.to_csv("OutputKnnLabelsTrain.csv")


orgLabel=np.array(labels_test)
#k1=newTrain[0]
#orgLabel=np.array(k1)

def accuracy(orgLab, predLab):
    sumL=0;
    for i in range(len(orgLab)):
        if(orgLab[i]==predLab[i]):
            sumL=sumL+1
    print sumL
    p=(float)(sumL)/(len(orgLab))
    print p
    return p

acc=accuracy(orgLabel,outLabels)

#print outLabels

print acc