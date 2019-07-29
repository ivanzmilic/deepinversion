import numpy as np
np.random.seed(123)
import numpy

import matplotlib
matplotlib.use('Agg')

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Conv1D, MaxPooling1D
from keras.utils import np_utils
from keras.models import load_model

#import pyana
from astropy.io import fits
import sys

import matplotlib.pyplot as plt 

#filenames to read
train_set_spectra_f = sys.argv[1] #data for training
train_set_nodes_f   = sys.argv[2] #parameters for training
to_fit_spectra_f = sys.argv[3] #data for interpretation

already_trained = int(sys.argv[5])

#read the data itself. Note for the notation nodes == parameters (that we want to infer)
train_set_spectra = fits.open(train_set_spectra_f)[0].data
train_set_nodes = fits.open(train_set_nodes_f)[0].data
to_fit_spectra = fits.open(to_fit_spectra_f)[0].data


#the spectra to be fit needs to be normalized (components 1-3 need to come to physical units)
#to_fit_spectra = to_fit_spectra[:,:,:,:]
#to_fit_spectra[:,:,0,:] *= 3.275E14/30.6
#for s in range(1,4):
#	to_fit_spectra[:,:,s,:] *= to_fit_spectra[:,:,0,:]

#we are only inferring from 0 and 3 components:
to_fit_spectra = np.delete(to_fit_spectra,numpy.s_[1:3],axis=2)

#same thing for the train set specta:
train_set_spectra = np.delete(train_set_spectra,numpy.s_[1:3],axis=2)

#we are only using a certain number of parameters:
train_set_nodes = np.transpose(train_set_nodes,(2,1,0))
N_NODES_TO_USE = 13
N_B_NODES = 3
train_set_nodes = train_set_nodes[:,:,:N_NODES_TO_USE]


#normalize the spectra first. This brings 1st component on a scale 0,1 and the second 
#component is just normalized w.r.t (since mean of the second compoonent is zero already)
spectra_min = np.min(train_set_spectra[:,:,0,:])
spectra_max = np.max(train_set_spectra[:,:,0,:])
spectra_range = spectra_max-spectra_min
train_set_spectra[:,:,0,:] -=spectra_min
train_set_spectra[:,:,:,:] /= spectra_range
to_fit_spectra[:,:,0,:] -=spectra_min
to_fit_spectra[:,:,:,:] /= spectra_range

#implement spectral mask:
if (int(sys.argv[8])<=0):
	print 'No mask provided'
else:
	mask = np.loadtxt(sys.argv[9],unpack=True,skiprows=1)
	train_set_spectra *=mask
	to_fit_spectra *= mask

#to_fit_spectra = to_fit_spectra[:,:,:,:907]
print to_fit_spectra.shape

# nodes describing magnetic field need to projected to the line of sight and the 
# inclination of the magnetic field transformed to cosine value
# @bakara:certain parameters need to be transformed in this manner, otherwise things are all over
# the place
if (N_B_NODES):
	train_set_nodes[:,:,-1] = np.cos(train_set_nodes[:,:,-1])
	for n in range(-N_B_NODES,-1):
		train_set_nodes[:,:,n] *= train_set_nodes[:,:,-1]


# now normalize the parameters themselves:
nodes_min = np.zeros(N_NODES_TO_USE)
nodes_max = np.zeros(N_NODES_TO_USE)
nodes_norm = np.zeros(N_NODES_TO_USE)
for n in range(0,N_NODES_TO_USE):
	nodes_min[n] = np.min(train_set_nodes[:,:,n])
	nodes_max[n] = np.max(train_set_nodes[:,:,n])
	nodes_norm[n] = nodes_max[n]-nodes_min[n]
	train_set_nodes[:,:,n] -= nodes_min[n]
	train_set_nodes[:,:,n] /= nodes_norm[n]

N_STOKES = 2

#flatten
NX_train = train_set_spectra.shape[0]
NY_train = train_set_spectra.shape[1]
NL = train_set_spectra.shape[3] #must be same for both train and fit data
NX_data = to_fit_spectra.shape[0]
NY_data = to_fit_spectra.shape[1]
train_set_spectra = np.transpose(train_set_spectra,(0,1,3,2))
train_set_spectra = train_set_spectra.reshape(NX_train*NY_train,NL,N_STOKES)
train_set_nodes = train_set_nodes[:,:,:N_NODES_TO_USE]
train_set_nodes = train_set_nodes.reshape(NX_train*NY_train,N_NODES_TO_USE)

# seems like 64 is optimal for some reason but bigger might be better if we leave it to train longer?
N_BATCH = 64

model=Sequential()

already_trained = int(sys.argv[5])

if (already_trained):
	print 'Loading trained model from a file ...'
	model = load_model(sys.argv[4])

#convolution layers - depend on sampling. For critically sampled data.
# and I mean critically sampled w.r.t to important scale (doppler width)
# should be something like 7 -> 5 - > 3 if the sampling is more or less similar to Doppler with
#History of usage: 
# IR 1.5 micron lines 7-5-3 works.
# Na D from SST: 21-11-5 (I think)
# SOLIS training on a grid data set. Will try with 7-5-3

else: 
	model=Sequential()

	# add some convolutional layers
	model.add(Conv1D(64,(7,),activation='relu',input_shape=(NL,N_STOKES)))
	model.add(MaxPooling1D(pool_size=(2,)))
	model.add(Conv1D(64,(5,),activation='relu',))
	model.add(MaxPooling1D(pool_size=(2,)))
	model.add(Conv1D(64,(3,),activation='relu',))
	model.add(MaxPooling1D(pool_size=(2,)))
	model.add(Dropout(0.5))
	model.add(Flatten())

	# at the beginning it seems like two Dense layers are enough (honestly eeven one might be enough)
	model.add(Dense((NL-200)*N_STOKES,activation='sigmoid'))
	model.add(Dense(30,activation='sigmoid'))
	#model.add(Dense(20,activation='sigmoid'))
	model.add(Dense(N_NODES_TO_USE,activation='linear'))

	#compile, train
	#@bakara : 200 epochs gives sort of convergence. It will take some time to train. 30mins or so on
	#my machine
	model.compile(loss='mean_squared_error',optimizer='adam')
	model.fit(train_set_spectra,train_set_nodes,batch_size=N_BATCH,epochs=100,verbose=1,validation_split=0.2)
	model.save(sys.argv[4])

#prepare other data for the interpretation
to_fit_spectra = np.transpose(to_fit_spectra,(0,1,3,2))
print NX_data, NY_data
to_fit_spectra = to_fit_spectra.reshape(NX_data*NY_data,NL,N_STOKES)

#put inferred parameters here:
to_fit_nodes = np.zeros([NX_data*NY_data,N_NODES_TO_USE])
to_fit_nodes = model.predict(to_fit_spectra)
to_fit_nodes = to_fit_nodes.reshape(NX_data,NY_data,N_NODES_TO_USE)

#transpose because of the way pyana in c++ reads this
to_fit_nodes = np.transpose(to_fit_nodes,(2,0,1))

#now look at the parameter values used for training, and first put them back to the original form
train_set_nodes = train_set_nodes.reshape(NX_train,NY_train,N_NODES_TO_USE)
train_set_nodes = np.transpose(train_set_nodes,(2,0,1))

for n in range(0,N_NODES_TO_USE):
	to_fit_nodes[n,:,:] *= nodes_norm[n]
	to_fit_nodes[n,:,:] += nodes_min[n]
	train_set_nodes[n,:,:] *= nodes_norm[n]
	train_set_nodes[n,:,:] += nodes_min[n]

#this is comparison which only makes sense because the dataset to interpret actually has a training set
#as its subset. So, basically I am trying out, on my own to see how well the results agree. 


test = int(sys.argv[7])
if (test):
	for i in range (0,N_NODES_TO_USE):
		plt.clf()
		plt.cla()
		plt.subplot(211)
		m = np.mean(train_set_nodes[i])
		s = np.std(train_set_nodes[i])
		#print m,s
		plt.imshow(train_set_nodes[i],vmin=m-3*s,vmax=m+3*s,cmap='hot')
		plt.colorbar()	
		plt.subplot(212)
		plt.imshow(to_fit_nodes[i],vmin=m-3*s,vmax=m+3*s,cmap='hot')	
		plt.colorbar()	
		plt.tight_layout()
		plt.savefig('simple_compare'+str(i)+'.png',bbox_inches='tight')

		print np.std((train_set_nodes[i]-to_fit_nodes[i]))/np.fabs(np.amax(train_set_nodes[i]))

else:
	for i in range (0,N_NODES_TO_USE):
		plt.clf()
		plt.cla()
		m = np.mean(to_fit_nodes[i])
		s = np.std(to_fit_nodes[i])
		#print m,s
		plt.imshow(to_fit_nodes[i],vmin=m-3*s,vmax=m+3*s,cmap='hot')	
		plt.colorbar()	
		plt.tight_layout()
		plt.savefig('inferred_nodes_'+str(i)+'.png',bbox_inches='tight')


#revert the infered parameters back to original form (magnetic field; inclination instead of 
#los magnetic field; cos of inclination)

to_fit_nodes[-1,:,:] = np.clip(to_fit_nodes[-1],-0.99,0.99)

to_fit_nodes[-N_B_NODES:-1] /= to_fit_nodes[-1]
to_fit_nodes[-1] = np.arccos(to_fit_nodes[-1])

to_fit_nodes = np.transpose(to_fit_nodes,(0,2,1))
fits_hdu_output = fits.PrimaryHDU(to_fit_nodes)
fits_hdu_output.writeto(sys.argv[6],overwrite=True)

