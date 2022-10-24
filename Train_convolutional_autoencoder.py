 # -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 13:21:46 2022

@author: Amin Boumerdassi
This script contains an autoencoder for anomaly detection of noise glitches.
"""
from tensorflow import keras
import numpy as np
from tensorflow.keras.layers import Input, Conv1D, Conv1DTranspose, Dropout, BatchNormalization, Dense, MaxPooling1D, SpatialDropout1D, GaussianNoise
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse, mae
from tensorflow.keras.optimizers import Adam, Adamax, Adadelta
from tensorflow.keras.regularizers import L1L2, L1, L2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle

#The glitch types are:
#["Air_Compressor","Blip_","Chirp","Extremely_Loud","Helix","Koi_Fish",
#"Light_Modulation","Low_Frequency_Burst","Low_Frequency_Lines","Paired_Doves","No_Glitch",
#"Power_Line","Repeating_Blips","Scattered_Light","Scratchy","Tomte","Violin_Mode","Whistle","None_of_the_Above"]

#Choose a confidence level
conf= "95_100pc"#{lower}_{higher}per cent
save_dir= "glitch_data_and_label_conf_"+conf

'''#Load dictionary for easy filtering of glitches
with open("/home/amin.boumerdassi/MLy_glitch_classification/Training_data/glitches_by_mly/"+save_dir+"/mly_glitch_mapping.pkl", 'rb') as f:
    classes_dict = pickle.load(f)#format: {"glitch_type": int}'''

#Choose a glitch type
glitch_type= "Blip"#for blip use "Blip"

model_name= "conv_{:}_AE_conf_{:}".format(glitch_type,conf)

#Load glitch data
dsampled_glitch_data= np.load("/home/amin.boumerdassi/MLy_glitch_classification/Training_data/glitches_by_mly/"+save_dir+"/mly_"+glitch_type+"_data.npy")

'''#Load and decode labels for filtering data by glitch type
ylabel= np.load("/home/amin.boumerdassi/MLy_glitch_classification/Training_data/glitches_by_mly/"+save_dir+"/mly_glitch_labels_encoded.npy")
ylabel= np.argmax(ylabel,axis=1)

#To filter by glitch type
dsampled_glitch_data= dsampled_glitch_data[ylabel==classes_dict[glitch_type]]
ylabel= ylabel[ylabel==classes_dict[glitch_type]]'''

'''#Normalise the timeseries to have max value of 1
for glitch_idx in range(len(dsampled_glitch_data)):
    dsampled_glitch_data[glitch_idx]=dsampled_glitch_data[glitch_idx]/np.amax(dsampled_glitch_data[glitch_idx])'''
    
#Then apply test-train split
X_train, X_test, _, _ = train_test_split(dsampled_glitch_data, ylabel, test_size=0.33,random_state=1)

'''#Roll the glitches, then crop them to reduce the noise
fs=4096#hz
glitch_dur=1/10#s
window_dur=1/8#s
midpoint_ele= len(X_train[0])//2
new_dims= int(np.round_(fs*window_dur))

for glitch_index in range(X_train.shape[0]):
    X_train[glitch_index]=np.roll(X_train[glitch_index], np.random.randint(low=-fs*.5*(window_dur-glitch_dur),high=+fs*.5*(window_dur-glitch_dur)))

for glitch_index in range(X_test.shape[0]):
    X_test[glitch_index]=np.roll(X_test[glitch_index], np.random.randint(low=-fs*.5*(window_dur-glitch_dur),high=+fs*.5*(window_dur-glitch_dur)))
    
X_train=X_train[:,int(midpoint_ele-.5*new_dims):int(midpoint_ele+.5*new_dims),:]
X_test=X_test[:,int(midpoint_ele-.5*new_dims):int(midpoint_ele+.5*new_dims),:]'''
    

#Initialise the regularisation
regularise= L1(0)#L1L2(l1=1e-5, l2=1e-5)

#Create the convolutional autoencoder class
#batch normalisation may speed up the training
autoencoder= keras.Sequential()#nodes: 32->8->32, maybe 256->128->256
#The encoder
autoencoder.add(Input(shape=(X_train.shape[1],X_train.shape[2])))
#The encoder
autoencoder.add(GaussianNoise(0.1,seed=1))
autoencoder.add(Conv1D(64, kernel_size=8, activation='sigmoid', strides=1, padding="same"))
autoencoder.add(MaxPooling1D(pool_size=2))
autoencoder.add(BatchNormalization(momentum=0.9))
#autoencoder.add(SpatialDropout1D(0.05))
autoencoder.add(Conv1D(64, kernel_size=8, activation='sigmoid', strides=1, padding="same"))
autoencoder.add(MaxPooling1D(pool_size=2))
autoencoder.add(BatchNormalization(momentum=0.9))
#autoencoder.add(SpatialDropout1D(0.05))
autoencoder.add(Conv1D(64, kernel_size=8, activation='sigmoid', strides=1, padding="same"))
autoencoder.add(MaxPooling1D(pool_size=2))
autoencoder.add(BatchNormalization(momentum=0.9))
#autoencoder.add(SpatialDropout1D(0.05))
autoencoder.add(Conv1D(64, kernel_size=8, activation='sigmoid', strides=1, padding="same"))
autoencoder.add(MaxPooling1D(pool_size=2))
autoencoder.add(BatchNormalization(momentum=0.9))
#autoencoder.add(SpatialDropout1D(0.05))
autoencoder.add(Conv1D(64, kernel_size=8, activation='sigmoid', strides=1, padding="same"))
autoencoder.add(MaxPooling1D(pool_size=2))
autoencoder.add(BatchNormalization(momentum=0.9))
#autoencoder.add(SpatialDropout1D(0.05))
autoencoder.add(Conv1D(64, kernel_size=8, activation='sigmoid', strides=1, padding="same"))
autoencoder.add(MaxPooling1D(pool_size=2))
autoencoder.add(BatchNormalization(momentum=0.9))
#autoencoder.add(SpatialDropout1D(0.05))
autoencoder.add(Conv1D(64, kernel_size=8, activation='sigmoid', strides=1, padding="same"))
autoencoder.add(MaxPooling1D(pool_size=2))
autoencoder.add(BatchNormalization(momentum=0.9))
#autoencoder.add(SpatialDropout1D(0.05))
autoencoder.add(Conv1D(64, kernel_size=8, activation='sigmoid', strides=1, padding="same"))
autoencoder.add(MaxPooling1D(pool_size=2))
#autoencoder.add(SpatialDropout1D(0.05))
autoencoder.add(Conv1D(64, kernel_size=8, activation='linear', strides=1, padding="same"))
autoencoder.add(MaxPooling1D(pool_size=2))

#The decoder
autoencoder.add(Conv1DTranspose(64, kernel_size=8, strides=2, activation='sigmoid', padding="same"))
autoencoder.add(BatchNormalization(momentum=0.9))
#autoencoder.add(SpatialDropout1D(0.05))
autoencoder.add(Conv1DTranspose(64, kernel_size=8, strides=2, activation='sigmoid', padding="same"))
autoencoder.add(BatchNormalization(momentum=0.9))
#autoencoder.add(SpatialDropout1D(0.05))
autoencoder.add(Conv1DTranspose(64, kernel_size=8, strides=2, activation='sigmoid', padding="same"))
autoencoder.add(BatchNormalization(momentum=0.9))
#autoencoder.add(SpatialDropout1D(0.05))
autoencoder.add(Conv1DTranspose(64, kernel_size=8, strides=2, activation='sigmoid', padding="same"))
autoencoder.add(BatchNormalization(momentum=0.9))
#autoencoder.add(SpatialDropout1D(0.05))
autoencoder.add(Conv1DTranspose(64, kernel_size=8, strides=2, activation='sigmoid', padding="same"))
autoencoder.add(BatchNormalization(momentum=0.9))
#autoencoder.add(SpatialDropout1D(0.05))
autoencoder.add(Conv1DTranspose(64, kernel_size=8, strides=2, activation='sigmoid', padding="same"))
autoencoder.add(BatchNormalization(momentum=0.9))
#autoencoder.add(SpatialDropout1D(0.05))
autoencoder.add(Conv1DTranspose(64, kernel_size=8, strides=2, activation='sigmoid', padding="same"))
autoencoder.add(BatchNormalization(momentum=0.9))
#autoencoder.add(SpatialDropout1D(0.05))
autoencoder.add(Conv1DTranspose(64, kernel_size=8, strides=2, activation='sigmoid', padding="same"))
autoencoder.add(BatchNormalization(momentum=0.9))
#autoencoder.add(SpatialDropout1D(0.05))
autoencoder.add(Conv1DTranspose(64, kernel_size=8, strides=2, activation='sigmoid', padding="same"))
autoencoder.add(Conv1DTranspose(1, kernel_size=8, strides=1, activation='linear', padding="same"))

#Experiment with the learning rate
optimiser= Adam(learning_rate=.0005)#default=0.01, using 0.0001 for blips w/ mae loss

#Compile the autoencoder
autoencoder.summary()

autoencoder.compile(optimizer=optimiser, loss="mse")

#Add early stopping
early_stopping = EarlyStopping(monitor='loss', patience=500)

#Train using X and Y data both as timeseries
history= autoencoder.fit(X_train, X_train,
                epochs=1000, 
                batch_size=64,
                shuffle=True,
                validation_data=(X_test, X_test), callbacks=[early_stopping])#originally batch size 64

#Save the trained autoencoder
autoencoder.save(model_name)

#Save plot of the loss distribution of training data
train_reconstructions = autoencoder.predict(X_train)
train_loss = np.mean(np.abs(train_reconstructions - X_train), axis=1)
#for setting x-limits on histogram
train_upper_lim= np.quantile(train_loss,.99)
train_lower_lim= np.quantile(train_loss,.01)

plt.hist(train_loss, bins=100, density=True)#, range=(train_lower_lim,train_upper_lim))
plt.xlabel("Train loss")
plt.ylabel("Probability Density")
plt.title("Training: {:} losses".format(glitch_type))
plt.figtext(0,0,"Model: {:}".format(model_name), ha="left")
plt.xlim(train_lower_lim,train_upper_lim)
plt.savefig("{:}/{:}_train_loss.png".format(model_name,glitch_type),bbox_inches='tight')

#Save plot of the loss distribution of validation data
test_reconstructions = autoencoder.predict(X_test)
test_loss = np.mean(np.abs(test_reconstructions - X_test), axis=1)
#for setting x-limits on histogram
test_upper_lim= np.quantile(test_loss,.99)
test_lower_lim= np.quantile(test_loss,.01)

plt.figure()
plt.hist(test_loss, bins=100, density=True)#, range=(test_lower_lim,test_upper_lim))
plt.xlabel("Test loss")
plt.ylabel("Probability Density")
plt.title("Testing: {:} losses".format(glitch_type))
plt.figtext(0,0,"Model: {:}".format(model_name), ha="left")
plt.xlim(test_lower_lim,test_upper_lim)
plt.savefig("{:}/{:}_test_loss.png".format(model_name,glitch_type),bbox_inches='tight')

#Set up subplots of training reconstructions
plt.figure()
subplot_dims= [2,3]
rng= np.random.RandomState(1)
glitch_indices= rng.randint(0, high=len(X_train), size=subplot_dims[0]*subplot_dims[1])
plot_indices= np.arange(1,len(glitch_indices)+1)
#Annotate subplots
plt.subplots(subplot_dims[0],subplot_dims[1],sharex=True)
plt.subplot(subplot_dims[0], subplot_dims[1], 1)
plt.ylabel("Whitened Strain")
plt.title("{:} training reconstructions".format(glitch_type))
plt.subplot(subplot_dims[0], subplot_dims[1], subplot_dims[0]*subplot_dims[1]-subplot_dims[1]+1)
plt.ylabel("Whitened Strain")
plt.xlabel("Time (s)")
#Plot the subplots
for glitch_index, plot_index in zip(glitch_indices,plot_indices):
    plt.subplot(subplot_dims[0], subplot_dims[1], plot_index)
    plt.plot(np.arange(0,X_train.shape[1]/4096,1/4096),X_train[glitch_index],"red", label="Original waveform",alpha=1) 
    plt.plot(np.arange(0,X_train.shape[1]/4096,1/4096),X_train[glitch_index]-train_reconstructions[glitch_index], "k", label="Reconstruction error")
    plt.plot(np.arange(0,X_train.shape[1]/4096,1/4096),train_reconstructions[glitch_index],"green", label="Reconstructed waveform",alpha=1)
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
plt.savefig("{:}/{:}_training_reconstructions.png".format(model_name,glitch_type),bbox_inches='tight')

#Save subplots of testing reconstructions
plt.figure()
subplot_dims= [2,3]
glitch_indices= rng.randint(0, high=len(X_test), size=subplot_dims[0]*subplot_dims[1])
plot_indices= np.arange(1,len(glitch_indices)+1)
#Annotate subplots
plt.subplots(subplot_dims[0],subplot_dims[1],sharex=True)
plt.subplot(subplot_dims[0], subplot_dims[1], 1)
plt.title("{:} testing reconstructions".format(glitch_type))
plt.ylabel("Whitened Strain")
plt.subplot(subplot_dims[0], subplot_dims[1], subplot_dims[0]*subplot_dims[1]-subplot_dims[1]+1)
plt.ylabel("Whitened Strain")
plt.xlabel("Time (s)")
for glitch_index, plot_index in zip(glitch_indices,plot_indices):
    plt.subplot(subplot_dims[0], subplot_dims[1], plot_index)
    plt.plot(np.arange(0,X_train.shape[1]/4096,1/4096),X_test[glitch_index],"red", label="Original waveform",alpha=1)
    plt.plot(np.arange(0,X_train.shape[1]/4096,1/4096),X_test[glitch_index]-test_reconstructions[glitch_index], "k", label="Reconstruction error")
    plt.plot(np.arange(0,X_train.shape[1]/4096,1/4096),test_reconstructions[glitch_index],"green", label="Reconstructed waveform",alpha=1)# 
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
plt.savefig("{:}/{:}_test_reconstructions.png".format(model_name,glitch_type),bbox_inches='tight')

#Save plot of the training history
plt.figure()
plt.plot(history.history['loss'], label="Training loss")
plt.plot(history.history['val_loss'], label="Validation loss")
plt.title('Training: {:} loss history'.format(glitch_type))
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.figtext(0,0,"Model: {:}".format(model_name), ha="left")
plt.legend()
plt.savefig("{:}/{:}_loss_history.png".format(model_name,glitch_type),bbox_inches='tight')

'''#From this distribution, decide on the threshold loss (e.g. mean + 1 std)
threshold = np.mean(train_loss) + np.std(train_loss)
with open("{:}/threshold_loss.txt".format(model_name),"w",newline="") as file:
    file.write(str(threshold))'''

#Save the training and testing loss for plotting against the test losses
np.save("{:}/training_losses".format(model_name),train_loss)
np.save("{:}/testing_losses".format(model_name),test_loss)
