from tensorflow.keras.models import load_model
import sys
sys.path.append("/home/amin.boumerdassi/mly_local/")
from mly.datatools import DataPod, DataSet
import matplotlib.pyplot as plt
import numpy as np

#The glitch types are:
#["Air_Compressor","Blip_","Chirp","Extremely_Loud","Helix","Koi_Fish",
#"Light_Modulation","Low_Frequency_Burst","Low_Frequency_Lines","Paired_Doves","No_Glitch",
#"Power_Line","Repeating_Blips","Scattered_Light","Scratchy","Tomte","Violin_Mode","Whistle","None_of_the_Above"]

#Choose a confidence level
conf= "95_100pc"#{lower}_{higher}per cent
save_dir= "glitch_data_and_label_conf_"+conf

#Choose a glitch autoencoder and load it
glitch_type= "Blip"
model_name= "conv_{:}_AE_conf_{:}".format(glitch_type,conf)
autoencoder= load_model(model_name)

#Load testing losses and store the upper/lower lims to cut out long tails
train_loss= np.load(model_name+"/testing_losses.npy")
upper_lim= np.quantile(train_loss,.99)#needs to be chosen by eye
lower_lim= np.quantile(train_loss,.01)#needs to be chosen by eye

#Load the injections for testing on the autoencoder - iterate by injection type
SNRs= [10,15,30,40,60]
injection_types= ["CSGs","Ringdowns","WNBs","CBCs","Gaussian"]
rng= np.random.RandomState(1)

for SNR in SNRs:
    #Prepare plotting variables and subplots
    subplot_dims= [len(injection_types),3]
    plot_indices= np.arange(1,subplot_dims[1]+1)#for subplots
    plt.figure(SNR+1)
    plt.subplots(subplot_dims[0], subplot_dims[1],sharex=True)
    
    #Plot training loss prob. density, but first compute limits of plotting
    plt.figure(SNR)
    upper_lim= np.quantile(train_loss,.99)
    lower_lim= np.quantile(train_loss,.01)
    plt.hist(train_loss, bins=100, label=glitch_type, density=True,range=(lower_lim,upper_lim))
    #Testing and plotting injection histograms
    for injection_type in injection_types:
        #Check if we're testing Gaussian
        if injection_type=="Gaussian":
            injection_dir= "/home/amin.boumerdassi/MLy_glitch_classification/H_fs_4096/noise/Gaussian_optimal_correlation_/OptimalNoise_0_10000.pkl"
        else:
            injection_dir= "/home/amin.boumerdassi/MLy_glitch_classification/H_fs_4096/burst/{:}_optimal_correlation_/BurstWithOptimalNoise_{:}_10000.pkl".format(injection_type,SNR)    
        #Test and plot injections losses
        plt.figure(SNR)
        injection_dataSet= DataSet.load(injection_dir)
        injection_strain= injection_dataSet.exportData("strain", shape=(None,4096,1))#1: one single detector stream
        reconstructions = autoencoder.predict(injection_strain)
        test_loss = np.mean(np.abs(reconstructions - injection_strain), axis=1)
        plt.hist(test_loss, bins=100, label=injection_type, alpha=.5, density=True)#, range=(lower_lim,upper_lim))
        #To update our xlim on the graph; xlim to be applied at the end
        if upper_lim< np.quantile(test_loss,.99):
            upper_lim= np.quantile(test_loss,.99)
        else:
            pass
        if lower_lim > np.quantile(test_loss,.99):
            lower_lim= np.quantile(test_loss,.99)
        else:
            pass
        #Plot reconstructed waveforms and reconstruction error
        plt.figure(SNR+1)
        plt.subplot(subplot_dims[0], subplot_dims[1], plot_indices[0])
        plt.ylabel(injection_type, rotation=0)        
        inj_indices= rng.randint(0, high=len(injection_strain), size=subplot_dims[1])
        for plot_no, inj_no in zip(plot_indices, inj_indices):
            plt.subplot(subplot_dims[0], subplot_dims[1], plot_no)
            plt.plot(np.arange(0,1,1/4096), injection_strain[inj_no],"red", label="Original injection", alpha=1)
            plt.plot(np.arange(0,1,1/4096), injection_strain[inj_no]-reconstructions[inj_no],"k", label="Reconstruction error".format(injection_type,inj_no), alpha=1)
            plt.plot(np.arange(0,1,1/4096), reconstructions[inj_no],"green", label="Reconstructed injection", alpha=1)

        plot_indices+=3

    #Annotate and save reconstructions
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.subplot(subplot_dims[0], subplot_dims[1], 1)
    plt.title("Injection reconstructions @ SNR {:}".format(SNR))
    plt.subplot(subplot_dims[0], subplot_dims[1], subplot_dims[0]*subplot_dims[1]-subplot_dims[1]+1)
    plt.xlabel("Time (s)")
    plt.savefig("{:}/injection_SNR_{:}_reconstructions.png".format(model_name,SNR),bbox_inches='tight')
    
    #Annotate and save injection loss histograms
    plt.figure(SNR)
    plt.title("Testing: injections@SNR{:} loss".format(SNR))
    plt.xlabel("Test loss")
    plt.ylabel("Probability Density")
    plt.xlim(lower_lim,upper_lim)
    plt.legend()
    plt.figtext(0,0,"Model: {:}".format(model_name), ha="left")
    plt.savefig("{:}/injections_SNR{:}_test_loss.png".format(model_name,SNR),bbox_inches='tight')
