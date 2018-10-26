from sklearn import preprocessing
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')
from nolearn.dbn import DBN
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.layers import Input, Dense
from keras.models import Model


batchSize = 128
nclasses = 10
nepoch = 20

encoding_dimension = 128
def trainTest(x_train_final, y_train, x_test_final, y_test):
	print "Enter learning rate"
	learning_rate = input()
	model = Sequential()

	model.add(Dense(100, input_shape=(784,)))
	model.add(Activation('relu'))

	model.add(Dense(50))
	model.add(Activation('sigmoid'))

	model.add(Dense(10))
	model.add(Activation('softmax'))

	sgd = SGD(lr=learning_rate)
	model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
	# convert class vectors to binary class matrices
	Y_train = np_utils.to_categorical(y_train, nclasses)
	Y_test = np_utils.to_categorical(y_test, nclasses)

	build = model.fit(x_train_final, Y_train,
	                    batch_size=batchSize, nepoch=50,
	                    verbose=0, validation_data=(x_test_final, Y_test))

	score = model.evaluate(x_test_final, Y_test, verbose=0)
	print('Test accuracy:', score[1])

def encoder():
	(X_train, y_train), (X_test, y_test) = mnist.load_data()

	X_train = X_train.reshape(60000, 784)
	X_test = X_test.reshape(10000, 784)
	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	X_train /= 255
	X_test /= 255
	print(X_train.shape[0], 'size of train samples')
	print(X_test.shape[0], 'size of test samples')
	input_img = Input(shape=(784,))
	encoded = Dense(encoding_dimension, activation='relu')(input_img)
	encoded_2 = Dense(100, activation='relu')(encoded)
	decoded = Dense(100, activation='relu')(encoded_2)
	decoded_2 = Dense(784, activation='sigmoid')(decoded)
	autoencoder = Model(input=input_img, output=decoded_2)	
	autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

	################training the data #############################################################
	print "--------------Training------------------"
	autoencoder.fit(X_train, X_train,
	                nepoch=50,
	                batch_size=256,
	                shuffle=True,
	                verbose=0,
	                validation_data=(X_test, X_test))


	x_train_final = autoencoder.predict(X_train)
	x_test_final = autoencoder.predict(X_test)
	print "------------Prediction done-------------"
	print "----Plotting----"
	plt.figure(figsize=(20, 4))
	for i in range(10):
	    # display original
	    ax = plt.subplot(2, n, i + 1)
	    plt.imshow(X_train[i].reshape(28, 28))
	    plt.gray()
	    ax.get_xaxis().set_visible(False)
	    ax.get_yaxis().set_visible(False)

	    # display reconstruction
	    ax = plt.subplot(2, n, i + 1 + n)
	    plt.imshow(x_train_final[i].reshape(28, 28))
	    plt.gray()
	    ax.get_xaxis().set_visible(False)
	    ax.get_yaxis().set_visible(False)
	plt.show()
	trainTest(x_train_final, y_train, x_test_final, y_test)



	


encoder()