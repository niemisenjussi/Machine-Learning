import os    
#os.environ['THEANO_FLAGS'] = "device=gpu0,floatX=float32" #IF NVIDIA GPU IN USE
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['GOTO_NUM_THREADS'] = '8'
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['THEANO_FLAGS'] = 'device=cpu,blas.ldflags=-lblas -lgfortran'

import theano
from keras.models import Sequential
from keras.layers import Dense
from sklearn import svm
from sklearn import datasets
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.cross_validation import train_test_split
import numpy
from sklearn import preprocessing

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

# Every 100 rows correspond to one gene.
# Extract all 100-row-blocks into a list using numpy.split.
num_genes_train = x_train_raw.shape[0] / 100
num_genes_test  = x_test_raw.shape[0] / 100

print("Train / test data has %d / %d genes." % \
	 (num_genes_train, num_genes_test))
x_train_raw = numpy.split(x_train_raw, num_genes_train)
x_test_raw  = numpy.split(x_test_raw, num_genes_test)

# Reshape by raveling each 100x5 array into a 500-length vector
x_train_raw = [g.ravel() for g in x_train_raw]
x_test_raw  = [g.ravel() for g in x_test_raw]

# convert data from list to array
x_train_raw = numpy.array(x_train_raw)
y_train_raw = numpy.array(y_train_raw)
x_test_raw  = numpy.array(x_test_raw)
y_train_raw = numpy.ravel(y_train_raw)

print("mean before:"+str(numpy.mean(x_train_raw)))
min_max_scaler = preprocessing.MinMaxScaler()
x_train_raw = preprocessing.scale(x_train_raw) 	
print("mean after:"+str(numpy.mean(x_train_raw)))

x_train, x_test, y_train, y_test = train_test_split(x_train_raw, y_train_raw, test_size=0.3, random_state=0)

X = x_train
Y = y_train

# create model
print("Creating model")
model = Sequential()
model.add(Dense(100, input_dim=500, init='uniform', activation='relu'))
model.add(Dense(5, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))
# Compile model
print("Compile Model")
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
print("Model Fit")
model.fit(X, Y, nb_epoch=500, batch_size=10)
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

exit()
predictions = model.predict(x_test) #some wierd bug, Garbage collector not destroying object if this here

