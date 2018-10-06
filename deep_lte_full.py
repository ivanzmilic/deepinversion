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

from astropy.io import fits 
import sys

import matplotlib.pyplot as plt 

#filenames to read
train_set_spectra_f = sys.argv[1] #data for training
train_set_atmos_f   = sys.argv[2] #parameters for training
to_fit_spectra_f = sys.argv[3] #data for interpretation

#read the data itself
temp = fits.open(train_set_spectra_f)
train_set_spectra = temp[0].data
temp = fits.open(train_set_atmos_f)
train_set_atmos = temp[0].data
temp = fits.open(to_fit_spectra_f)
to_fit_spectra = temp[0].data

#the spectra needs to be normalized (components 1-3 need to come to physical units)
for s in range(1,4):
	to_fit_spectra[:,:,s,:] *= to_fit_spectra[:,:,0,:]
	train_set_spectra[:,:,s,:] *= train_set_spectra[:,:,0,:]

#we are only inferring from 0 and 3 components:
to_fit_spectra = np.delete(to_fit_spectra,numpy.s_[1:3],axis=2)

#same thing for the train set specta:
train_set_spectra = np.delete(train_set_spectra,numpy.s_[1:3],axis=2)


#normalize the spectra first. This brings 1st component on a scale 0,1 and the second 
#component is just normalized w.r.t (since mean of the second compoonent is zero already)
#this is mostly due to physical reasons, but perhaps I should just normalize to 
# mean 0 and unit deviation? 
spectra_min = np.min(train_set_spectra[:,:,:,:])
spectra_max = np.max(train_set_spectra[:,:,:,:])
spectra_range = spectra_max-spectra_min
train_set_spectra[:,:,0,:] -=spectra_min
train_set_spectra[:,:,:,:] /= spectra_range
to_fit_spectra[:,:,0,:] -=spectra_min
to_fit_spectra[:,:,:,:] /= spectra_range

#tauscale is same for the all pixels, let's have it handy for later
tau = np.copy(train_set_atmos[0,0,0,:])

# magnetic field need to projected to the line of sight and the 
# inclination of the magnetic field transformed to cosine value
# certain parameters need to be transformed in this manner, otherwise things are all over
# the place (or maybe not, never actually tried, this was me talking out of my a**)
train_set_atmos[:,:,10,:] = np.cos(train_set_atmos[:,:,10,:])
train_set_atmos[:,:,7,:] *= train_set_atmos[:,:,10,:]

#we use following parameters: Temperature, Magnetic field, Los velocity magnetic field inclination
#I lied! I did not try to infer density/pressure here :(
params = [2,7,9,10]
NP = len(params)
train_set_atmos = train_set_atmos[:,:,[2,7,9,10]]

NZ = train_set_atmos.shape[-1]

# now normalize the parameters themselves:
N_P_TO_USE = NP*NZ
param_min = np.zeros(N_P_TO_USE)
param_max = np.zeros(N_P_TO_USE)
param_norm = np.zeros(N_P_TO_USE)

N_STOKES = 2

#flatten
NX_train = train_set_spectra.shape[0]
NY_train = train_set_spectra.shape[1]
NL = train_set_spectra.shape[3] #must be same for both train and fit data
NX_data = to_fit_spectra.shape[0]
NY_data = to_fit_spectra.shape[1]
train_set_spectra = np.transpose(train_set_spectra,(0,1,3,2))
train_set_spectra = train_set_spectra.reshape(NX_train*NY_train,NL,N_STOKES)
train_set_atmos = train_set_atmos.reshape(NX_train*NY_train,N_P_TO_USE)

#normalize the atmosphere:
for n in range(0,N_P_TO_USE):
	param_min[n] = np.min(train_set_atmos[:,n])
	param_max[n] = np.max(train_set_atmos[:,n])
	param_norm[n] = param_max[n]-param_min[n]
	train_set_atmos[:,n] -= param_min[n]
	train_set_atmos[:,n] /= param_norm[n]


# seems like 64 is optimal for some reason but bigger might be better if we leave it to train longer?
N_BATCH = 64

model=Sequential()

already_trained = int(sys.argv[5])

if (already_trained):
	print 'Loading trained model from a file ...'
	model = load_model(sys.argv[4])

#convolution layers - depend on sampling. For critically sampled data.
# and I mean critically sampled w.r.t to important scale (doppler width)
# I found that something like 7->5->3 works the best 

else: 
	model=Sequential()

	# add some convolutional layers
	model.add(Conv1D(N_BATCH,(9,),activation='relu',input_shape=(NL,N_STOKES)))
	model.add(MaxPooling1D(pool_size=(2,)))
	model.add(Conv1D(N_BATCH,(5,),activation='relu',))
	model.add(MaxPooling1D(pool_size=(2,)))
	model.add(Conv1D(N_BATCH,(3,),activation='relu',))
	model.add(MaxPooling1D(pool_size=(2,)))
	model.add(Flatten())

	# at the beginning it seems like two Dense layers are enough (honestly eeven one might be enough)
	model.add(Dense(NL*2,activation='relu'))
	model.add(Dense(N_P_TO_USE,activation='linear'))

	#compile, train
	#100-200 epochs gives sort of convergence. It will take some time to train. 30mins or so on
	#my machine
	model.compile(loss='mean_squared_error',optimizer='adam')
	model.fit(train_set_spectra,train_set_atmos,batch_size=N_BATCH,epochs=30,verbose=1,validation_split=0.2)

	model.save(sys.argv[4])

#prepare other data for the interpretation
to_fit_spectra = np.transpose(to_fit_spectra,(0,1,3,2))
print NX_data, NY_data
to_fit_spectra = to_fit_spectra.reshape(NX_data*NY_data,NL,N_STOKES)

#put inferred parameters here:
to_fit_atmos = np.zeros([NX_data*NY_data,N_P_TO_USE])
to_fit_atmos = model.predict(to_fit_spectra)
to_fit_atmos = to_fit_atmos.reshape(NX_data,NY_data,N_P_TO_USE)

for n in range(0,N_P_TO_USE):
	to_fit_atmos[:,:,n] *= param_norm[n]
	to_fit_atmos[:,:,n] += param_min[n]
	
to_fit_atmos = to_fit_atmos.reshape(NX_data,NY_data,NP,NZ)
to_fit_atmos = np.transpose(to_fit_atmos,(0,1,3,2))

small = np.where(to_fit_atmos[:,:,:,-1] < -0.9999)
to_fit_atmos[small,-1] = 0.9999
big = np.where(to_fit_atmos[:,:,:,-1] > 0.9999)
to_fit_atmos[big,-1] = 0.9999

to_fit_atmos = np.transpose(to_fit_atmos,(0,1,3,2))

to_fit_atmos[:,:,1,:] /= to_fit_atmos[:,:,-1,:]
to_fit_atmos[:,:,-1,:] = np.arccos(to_fit_atmos[:,:,-1,:])

atmos_full = np.zeros([NX_data,NY_data,12,NZ])
atmos_full[:,:,0,:] = np.copy(tau)
atmos_full[:,:,2,:] = np.copy(to_fit_atmos[:,:,0,:])
atmos_full[:,:,7,:] = np.copy(to_fit_atmos[:,:,1,:])
atmos_full[:,:,9,:] = np.copy(to_fit_atmos[:,:,2,:])
atmos_full[:,:,10,:] = np.copy(to_fit_atmos[:,:,3,:])

hdu = fits.primaryHDU(atmos_full)
hdu.writeto(sys.argv[6])

