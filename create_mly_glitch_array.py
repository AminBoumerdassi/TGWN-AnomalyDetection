# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 20:54:39 2022

@author: Amin Boumerdassi

This script will load the glitch directory list to locate all glitch files
and append them to an MLy dataset. These are saved as a numpy 
file.
"""
from numpy import save
import pickle
import sys
sys.path.append("/home/amin.boumerdassi/mly_local/")
from mly.datatools import DataPod, DataSet
import glob
#from sklearn.preprocessing import LabelEncoder
#from tensorflow.keras.utils import to_categorical

#The glitch types are:
glitch_types=["Air_Compressor","Blip_","Chirp","Extremely_Loud","Helix","Koi_Fish",
"Light_Modulation","Low_Frequency_Burst","Low_Frequency_Lines","Paired_Doves","No_Glitch",
"Power_Line","Repeating_Blips","Scattered_Light","Scratchy","Tomte","Violin_Mode","Whistle","None_of_the_Above"]

#Choose a confidence level
conf= "95_100pc"#{lower}_{higher}per cent
save_dir= "glitch_data_and_label_conf_"+conf

for glitch_type in glitch_types:
    #Define glitch directories
    glitch_directory1 = r'o1_glitch_pkl_conf_{:}/{:}*.pkl'.format(conf,glitch_type)
    glitch_directory2 = r'o2_glitch_pkl_conf_{:}/{:}*.pkl'.format(conf,glitch_type)
    #glitch_directory3 = r'o3/text_files/'
    
    #Locate all relevant glitch files
    glitch_dir= glob.glob(glitch_directory1)+glob.glob(glitch_directory2)#+glob.glob(glitch_directory3+'*.txt')
    #Save glitch type's file directory list
    #with open("glitch_data_and_label_conf_"+conf+"/mly_"+glitch_type+"_dir_list.pkl", "wb") as f:
    #    pickle.dump(glitch_files, f)
    
    #Load glitch dir list
    #with open(save_dir+"/mly_"+glitch_type+"_dir_list.pkl", "rb") as file:
    #    glitch_dir= pickle.load(file)
        
    #Prepare the dataset and its labels as lists
    pod_list= []
    #glitch_labels= []
    
    #Load each glitch pkl file and append the datapod and label to lists
    for directory in glitch_dir:
        with open(directory, "rb") as file:
            datapod= pickle.load(file)#loads a datapod
            pod_list.append(datapod)
            
    #Create the dataset, and export its glitch data & labels
    dataset= DataSet(pod_list)
    glitch_data= dataset.exportData("strain", shape=(None, 4096, 1))#to ensure dims= (no. of glitches, fs, no. of detectors)
    
    #Remove extra underscore from glitch type
    if glitch_type[-1]=="_":
        glitch_type= glitch_type[:-1]
    
    #If you need to generate labels:
    #Initalise the label encoder, and encode glitch labels
    #glitch_labels= dataset.exportLabels("Glitch Type")
    #label_encoder= LabelEncoder()
    #ylabel= to_categorical(label_encoder.fit_transform(glitch_labels))
    #Store the label mappings as a dictionary
    #label_dict_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))) 
    #save(save_dir+"/mly_"+glitch_type+"_labels_encoded.npy", ylabel)

    #Save glitch data, encoded labels, and dict mapping of labels
    save(save_dir+"/mly_"+glitch_type+"_data.npy", glitch_data)
    #with open(save_dir+"/mly_glitch_mapping.pkl", "wb") as f:
    #    pickle.dump(label_dict_mapping, f)

'''#Load glitch dir list
with open(save_dir+"/mly_glitch_dir_list.pkl", "rb") as file:
    glitch_dir= pickle.load(file)

#Prepare the dataset and its labels as lists
pod_list= []
#glitch_labels= []

#Load each glitch pkl file and append the datapod and label to lists
for directory in glitch_dir:
    with open(directory, "rb") as file:
        datapod= pickle.load(file)#loads a datapod
        pod_list.append(datapod)
        #glitch_labels.append(datapod.labels["Glitch Type"])
        
#Create the dataset, and export its glitch data & labels
dataset= DataSet(pod_list)
glitch_data= dataset.exportData("strain", shape=(None, 4096, 1))#to ensure dims= (no. of glitches, fs, no. of detectors) 
glitch_labels= dataset.exportLabels("Glitch Type")

#Initalise the label encoder, and encode glitch labels
label_encoder= LabelEncoder()
ylabel= to_categorical(label_encoder.fit_transform(glitch_labels))

#Store the label mappings as a dictionary
label_dict_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))) 
    
#Save glitch data, encoded labels, and dict mapping of labels
save(save_dir+"/mly_glitch_data.npy", glitch_data)
save(save_dir+"/mly_glitch_labels_encoded.npy", ylabel)
with open(save_dir+"/mly_glitch_mapping.pkl", "wb") as f:
    pickle.dump(label_dict_mapping, f)'''

