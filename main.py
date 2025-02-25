import os    
os.environ['THEANO_FLAGS'] = "device=gpu0,floatX=float32" #IF NVIDIA GPU IN USE
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['GOTO_NUM_THREADS'] = '8'
os.environ['OMP_NUM_THREADS'] = '8'
#os.environ['THEANO_FLAGS'] = 'device=cpu,blas.ldflags=-lblas -lgfortran'

import theano
import time
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D 
from keras.layers.pooling import MaxPooling1D
from keras.layers import Dense
from sklearn import svm
from sklearn import datasets
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.cross_validation import train_test_split
import numpy
from sklearn import preprocessing

start_time = time.time() #Take time how low whole program run takes

#-- Classification problem
noutputs = 2

#-- input dimensions
nfeats=5
width = 100
ninputs = nfeats*width

#-- number of hidden units (for MLP only):
nhiddens = ninputs / 2

#-- hidden units, filter sizes (for ConvNet only):
nstates = (50,625,125)
filtsize = 10
poolsize = 5
padding = numpy.floor(filtsize/2)

models = ['convnet','linear','mlp','own'] #Define all models which are available
current_model = 'own'

data_path = "data" # This folder holds the csv files

# load csv files. We use numpy.loadtxt. Delimiter is ","
# and the text-only header row will be skipped.

print("Loading data...")
x_train_raw = numpy.loadtxt(data_path + os.sep + "x_train.csv", delimiter = ",", skiprows = 1)
x_test_raw  = numpy.loadtxt(data_path + os.sep + "x_test.csv", delimiter = ",", skiprows = 1)    
y_train_raw = numpy.loadtxt(data_path + os.sep + "y_train.csv", delimiter = ",", skiprows = 1)

print("All files loaded. Preprocessing...")

# remove the first column(Id)
x_train_raw = x_train_raw[:,1:] 
x_test_raw  = x_test_raw[:,1:]   
y_train_raw = y_train_raw[:,1:] 
	
print("len(x_train_raw):"+str(len(x_train_raw)))
# Every 100 rows correspond to one gene.
# Extract all 100-row-blocks into a list using numpy.split.
num_genes_train = x_train_raw.shape[0] / 100#
num_genes_test  = x_test_raw.shape[0] / 100

print("Train / test data has %d / %d genes." % \
		(num_genes_train, num_genes_test))

x_train_raw = numpy.split(x_train_raw, num_genes_train)
x_test_raw  = numpy.split(x_test_raw, num_genes_test)

# Reshape by raveling each 100x5 array into a 500-length vector
x_train_raw = [g.ravel() for g in x_train_raw] #numpy.reshape(x_train_raw,(15485,100,5))#
x_test_raw  = [g.ravel() for g in x_test_raw]

#print("mean before:"+str(numpy.mean(x_train_raw)))
#min_max_scaler = preprocessing.MinMaxScaler()
#x_train_raw = preprocessing.scale(x_train_raw) 	
#print("mean after:"+str(numpy.mean(x_train_raw)))



# convert data from list to array
x_train_raw = numpy.array(x_train_raw)
y_train_raw = numpy.array(y_train_raw)
x_test_raw  = numpy.array(x_test_raw)
y_train_raw = numpy.ravel(y_train_raw)



print("x_train_raw:"+str(x_train_raw.shape))

x_train, x_test, y_train, y_test = train_test_split(x_train_raw, y_train_raw, test_size=0.3)
print("x_train:"+str(x_train.shape[1:]))

X = x_train
Y = y_train
model = None
# create model
print("Creating model:"+current_model)

if current_model == 'own':
	#Copied some random place and modified a bit.
	model = Sequential()
	model.add(Dense(len(x_train), input_dim=500, init='uniform', activation='relu'))#input_dim=13936, init='uniform', activation='relu'))
	model.add(Dense(500, init='uniform', activation='relu'))
	model.add(Dense(100, init='uniform', activation='relu'))
	model.add(Dense(5, init='uniform', activation='relu'))
	model.add(Dense(1, init='uniform', activation='sigmoid'))

elif model == 'linear':
   #-- Simple linear model
	model = Sequential()
	model.add(Dense(len(x_train), input_dim=500, init='uniform', activation='relu'))#input_dim=13936, init='uniform', activation='relu'))
	model.add(Dense(1, init='uniform', activation='sigmoid'))

elif current_model == 'mlp': #Does not work yet, needs lua to python conversion
	
	print("Not verified yet, some random ideas only")
	#-- Simple 2-layer neural network, with tanh hidden units
	model = Sequential()
	model.add(Dense(len(x_train), input_dim=ninputs))
	model.add(Dense(nhiddens, activation='tanh')) #nhiddens = ninputs/2
	model.add(Dense(1, init='uniform', activation='sigmoid')) #Sigmoid forces end result 0-1.0 range

elif current_model == 'convnet': #Does not work yet, needs lua to python conversion
	print("Not implemented yet, this might require data in x,y,z format, now we have 1500*500 => 1500*100*5")
	exit()
	#https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py
	#look this one
	
	#--a typical modern convolution network (conv+relu+pool)
	model = Sequential()

	#-- stage 1 : filter bank -> squashing -> Max pooling
	model.add(Convolution2D(nfeats, nstates[1], filtsize, border_mode='same', input_shape=x_train.shape[:1]))
	model.add(ReLU())
	model.add(MaxPooling1D(pool_size=poolsize, stride=None, border_mode='valid')) #poolsize=5

	#-- stage 2 : standard 2-layer neural network
	model.add(View(np.ceil((width-filtsize)/poolsize)*nstates[1]))
	model.add(Dropout(0.5))
	model.add(Linear(np.ceil((width-filtsize)/poolsize)*nstates[1], nstates[2]))
	model.add(ReLU())
	model.add(Linear(nstates[2], nstates[3]))
	model.add(ReLU())
	model.add(Linear(nstates[3], noutputs))

else:
	error('unknown -model')
	exit()
	
	
# Compile model
print("Compile Model")
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
print("Model Fit")
model.fit(X, Y, nb_epoch=10, batch_size=len(x_train))
# evaluate the model
print("Model evaluate")
scores = model.evaluate(X, Y)
print(scores)
print("Predicting now")
#Predict by using trained model
predictions = model.predict(x_test)
#Round result to 1 or 0, because our y_test data is 0 or 1. Final version this must be remove.
print("Rounding")
rounded = [numpy.round(x) for x in predictions]

#Show final scores
print("Final Results")
score = accuracy_score(rounded, y_test)
print("Score:"+str(score))
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


print("Calculation take %s seconds" % (time.time() - start_time))


exit()
predictions = model.predict(x_test) #some wierd bug, Garbage collector not destroying object if this here

