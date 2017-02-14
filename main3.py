import os    
os.environ['THEANO_FLAGS'] = "device=opencl0:0,floatX=float32" #IF NVIDIA GPU IN USE
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['GOTO_NUM_THREADS'] = '8'
os.environ['OMP_NUM_THREADS'] = '8'
#os.environ['THEANO_FLAGS'] = 'device=cpu,blas.ldflags=-lblas -lgfortran'

import theano
import time
from keras.models import Sequential
from keras.layers.convolutional import Convolution1D 
from keras.layers.convolutional import Convolution2D 
from keras.layers.pooling import MaxPooling1D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Reshape
from keras.layers.core import Flatten
#from keras.layers.core import Linear
from keras.layers import Dense
from sklearn import svm
from keras.optimizers import Adam
from keras.optimizers import SGD
from sklearn import datasets
from sklearn import tree
from keras.models import load_model
from keras.utils import np_utils
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.cross_validation import train_test_split
import numpy
from sklearn import preprocessing


start_time = time.time() #Take time how low whole program run takes

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
x_train_raw = numpy.array(x_train_raw)


x_train_raw = [g.ravel() for g in x_train_raw] #numpy.reshape(x_train_raw,(15485,100,5))#
x_test_raw  = [g.ravel() for g in x_test_raw]
print("x_train_raw shape:"+str(numpy.array(x_train_raw).shape))
print("x_test_raw shape:"+str(numpy.array(x_train_raw).shape))

#REALLY dirty method to rearrange array from 100*5 => 1*5*100
#FIXME
def process_data(dataset):
	x_convnet = []
	i = 0
	temparr1 = []
	temparr2 = []
	temparr3 = []
	temparr4 = []
	temparr5 = []

	j = 0
	for row in dataset:
		for elem in row:
			if j==0:
				temparr1.append(elem)
			elif j==1:
				temparr2.append(elem)
			elif j==2:
				temparr3.append(elem)
			elif j==3:
				temparr4.append(elem)
			elif j==4:
				temparr5.append(elem)
			j += 1
			if j == 5:
				j = 0
			i += 1
			if i == 500:
				masterarr = []
				masterarr.append(temparr1)
				masterarr.append(temparr2)
				masterarr.append(temparr3)
				masterarr.append(temparr4)
				masterarr.append(temparr5)
				masterarr = preprocessing.scale(masterarr)
				x_convnet.append(masterarr)
		
				temparr1 = []
				temparr2 = []
				temparr3 = []
				temparr4 = []
				temparr5 = []
				masterarr = []
				i=0
	x_convnet = numpy.array(x_convnet)
	return x_convnet.reshape(x_convnet.shape[0],1,5,100)
	

x_train_3d = process_data(x_train_raw)
x_test_3d = process_data(x_test_raw)


# convert data from list to array
x_train_raw = numpy.array(x_train_raw)
y_train_raw = numpy.array(y_train_raw)
x_test_raw  = numpy.array(x_test_raw)
y_train_raw = numpy.ravel(y_train_raw)


print("x_train_raw:"+str(x_train_raw.shape))
x_train3d, x_test3d, y_train3d, y_test3d = train_test_split(x_train_3d, y_train_raw, test_size=0.3)

print("y_train3d:"+str(y_train3d.shape))
print("x_train3d:"+str(x_train3d.shape))




runmode = 'TESTING' #'TESTING' #or REAL #use REAL if you want to submitable data to kaggle

#Best paramters in comments 			#86,2% so far
nfeats =      [1500,100]				#1500,100
filtters =    [[5,4],[1,5]]	   	#5,4  1,5
pooling =     [[1,3],[1,2]]			#1,3  1,2
denselayers = [500,5,1]		      #500  5,  1
dropouts =    [0.1,0.1]			   	#0.1  0.1


print("model parameters:")
print("nfeats:"+str(nfeats))
print("filtters:"+str(filtters))
print("pooling:"+str(pooling))
print("denselayers:"+str(denselayers))
print("dropouts:"+str(dropouts))


model = Sequential()
print("convnet shape:"+str(x_train3d.shape[1:]))
model.add(Convolution2D( nfeats[0], filtters[0][0], filtters[0][1], border_mode='same', input_shape=(1,5,100), activation='relu'))
model.add(MaxPooling2D( pool_size=(pooling[0][0],  pooling[0][1]), dim_ordering="th"))

model.add(Convolution2D( nfeats[1], filtters[1][0], filtters[1][1])) #100 was ok
model.add(MaxPooling2D( pool_size=(pooling[1][0], pooling[1][1]), dim_ordering="th"))

model.add(Dropout(dropouts[0]))

model.add(Flatten())
model.add(Dense(denselayers[0], activation='relu'))
model.add(Dropout(dropouts[0]))
model.add(Dense(denselayers[1], activation='relu')) #5 was ok
#model.add(Dense(1, activation='relu'))
model.add(Dense(1, activation='relu'))
print("output shape:"+str(model.output_shape))
print(model.summary())


# returns a compiled model
# identical to the previous one
#model = load_model('my_model.h5')	

	
# Compile model
print("Compile Model")
#optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, clipnorm=1.)
optimizer = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error',  optimizer=optimizer, metrics=['accuracy'])
# loss:   #mean_squared_error binary_crossentropy


print("Model Fit, outer")
if runmode == 'TESTING':
	model.fit(x_train3d, y_train3d, nb_epoch=1000, validation_data=(x_test3d, y_test3d), batch_size=500, shuffle=True,show_accuracy=True, verbose=1)
	print("Model evaluate, outer")
	scores = model.evaluate(x_test3d, y_test3d)
	print(scores)
	predictions = model.predict(x_test3d)
	
	rounded = [numpy.round(x) for x in predictions] #Round result to 1 or 0, because our y_test data is 0 or 1.
	score = accuracy_score(rounded, y_test3d)
	print("Score inner:"+str(score))
	print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
	
elif runmode == 'REAL':
	print("Running REAL MODE, cannot use validation accuracy")
	model.fit(x_train_3d, y_train_raw, nb_epoch=10, batch_size=500, shuffle=True,show_accuracy=True, verbose=1)

	print("Predicting real results")
	predictions = model.predict(x_test_3d)

	f = open('REAL_result.csv', 'w')
	f.write("GeneId,Prediction\n");
	for i in range(0,len(predictions)):
		f.write(str(i+1)+","+str(predictions[i][0])+"\n")
	f.close()

		
print("Calculation take %s seconds" % (time.time() - start_time))


exit()
predictions = model.predict(x_test) #some wierd bug, Garbage collector not destroying object if this here
