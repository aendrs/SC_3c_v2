# -*- coding: utf-8 -*-
"""
MEGA script for complete evaluation of models in the linux-PC

"""

import os
os.chdir('C:\\Users\\andre\\Documents\\Torino\\colliculus')
from shutil import copyfile
from sys import platform

from keras.models import  Model
from keras.layers import (
        Input, Lambda, Dense, Conv2D, BatchNormalization, Activation, 
        concatenate, MaxPooling2D, AveragePooling2D, Flatten, Dropout, GlobalAveragePooling2D, SpatialDropout2D)
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import CSVLogger
from keras import backend as K

from PMK_customlayer import P_CustomLayer, M_CustomLayer, K_CustomLayer
from fractalnet import fractal_net
import attention_mechanism
import vis_funcs

from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import skimage
from skimage.transform import resize
import numpy as np
import time
import sys

import pickle
from sklearn import metrics
import seaborn as sns
import pandas as pd
import umap

#%%  ========================================================================
# Variables and flags to set up before running

#  Global variables
num_classes=3

# 1) KDEF_bubbles_averaging
KDEF_bubbles_averaging=False
    
# 2) KDEF_saliency_average
KDEF_saliency_average =False

# 3) KDEF_freq_model_analysis
KDEF_freq_model_analysis=False

# 4) KDEF noisy images analysis
KDEF_noise_model_analysis=True
KDEF_noise_model_SPECKLE=False


# 5) KDEF partial occlusion procedure
KDEF_partial_occlusion=False



# 5) evaluate_customPMK_models_2021.py (this one is huge and does a lot)
general_evaluation_bubbles=False
KDEF_section4=False
V6_samples=False


#%% ======= LOAD MODEL ======================================================


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

model=[] #this is just to avoid the stupid warnings in Spyder IDE
script='Main_Model.py' #'Colliculus_FractalAttention_20.py'
#pathtomodel=folder where the model was saved
pathtomodel='C:\\Users\\andre\\Documents\\Torino\\colliculus\\trained_models\\2021-11-18'
# Load the model using the unreliable method (in lieu of the proper one)
# due to the incapacity of keras to save/load our model we have to first 
# reconstruct the model from the original script. So go to the relevant script

K.set_learning_phase(False)

# This is a hacky way to load the model structure from another script, 
# (due to the fact that we have package problems, which may be resolved with later updates)
# pay attention to the beginning and end lines!!!!
begline=54 
endline=234 
txt = ''
with open(script,'r+',encoding='utf-8') as sc:
    for i, line in enumerate(sc):
        if i >= begline and i<= endline:
            txt = txt + line
exec(txt)
# Load Weights
model.load_weights(os.path.join(pathtomodel,"model.h5")) #model.h5 are the WEIGHTS



#%% ======== 1 ========== 1 ============== 1 ================================
# 1) KDEF_bubbles_averaging
#   =========================================================================


if KDEF_bubbles_averaging:
    
    if num_classes==5:
        if platform=='win32':
            data_path='C:\\Users\\andre\\Documents\\Torino\\colliculus\\datasets\\KDEF_frontal'
        else:
            data_path= '/home/andres/Documents/colliculus/datasets/KDEF_frontal'   #KDEF dataset
        labels_dict={'afraid':0, 'angry': 1, 'happy':2, 'neutral':3, 'sad':4} 
        AKDEF_labels={'afraid':'AF', 'angry': 'AN', 'happy':'HA', 'neutral':'NE', 'sad':'SA'}
    elif num_classes==3:
        if platform=='win32':
            data_path='C:\\Users\\andre\\Documents\\Torino\\colliculus\\datasets\\KDEF_frontal_3class'
        else:
            data_path= '/home/andres/Documents/colliculus/datasets/KDEF_frontal_3class'   #KDEF dataset
        labels_dict={'happy':0, 'neutral':1, 'sad':2} 
        AKDEF_labels={'happy':'HA', 'neutral':'NE', 'sad':'SA'}
    else: 
        raise Exception("num_classes should be either 3 or 5" )
        
    if platform=='win32': 
        AKDEF_path='C:\\Users\\andre\\Documents\\Torino\\colliculus\\datasets\\AKDEF_square'
    else:
        AKDEF_path='/home/andres/Documents/colliculus/datasets/AKDEF_square'
        
    savedirectory='KDEF_avg_bubbles'
    
    
    
    sigma_vec=[7.68,8.96,10.24] #[6.4, 7.68, 8.96, 10.24]# [6.4, 7.68] #, 8.96, 10.24] 11.52
    bubbles_iterations=500
    
    #%% =========  CALCULATE BUBBLES for averaging
    
    start=time.time()
    
    bubblesdict={'subj':[]}
    for sigma in sigma_vec:
        for gen in ['F','M']:
            for cat in list(labels_dict.keys()):
                sigmapointless=str(sigma).replace('.','') #remove point from sigma 
                bubblesdict['bubbleplaneN_SS_'+gen+'_'+str(cat)+'_s'+sigmapointless]=[]  # Norm(scaled/scaled)
                bubblesdict['bubbleplaneSS_'+gen+'_'+str(cat)+'_s'+sigmapointless]=[]  # scaled/scaled         
                bubblesdict['bubbleplaneNS_'+gen+'_'+str(cat)+'_s'+sigmapointless]=[]  # Norm(scaled/scaled)                
                bubblesdict['correctplane_'+gen+'_'+str(cat)+'_s'+sigmapointless]=[]
                bubblesdict['totalplane_'+gen+'_'+str(cat)+'_s'+sigmapointless]=[]
                bubblesdict['accvec_'+gen+'_'+str(cat)+'_s'+sigmapointless]=[]
    
    for sigma in sigma_vec:    
        sigmapointless=str(sigma).replace('.','') #remove point from sigma
        for root, dirs, files in os.walk(data_path, topdown=True):
            for file in files:      
                print(file)
                       
                category=root.split(os.sep)[-1] #extract CATEGORY string from the root directory, could be done otherwise with the dict or a list
                gender=file[1]
                
                bubblesdict['subj'].append([category,file])
                #print('working on image: {}  {}'.format(category, file))
                    
                # load base image============================================================== 
                baseimage = skimage.io.imread(root + os.sep +  file)
                baseimage=resize(baseimage, (128,128),preserve_range=True, anti_aliasing=True)
                inputimg=baseimage.astype(np.float32)/255.
                inputimg=inputimg[np.newaxis,...]# Make it a batch of one.
             
                predictions=model.predict(inputimg) #get PREDICTIONS
                label_index = np.argmax(predictions) #useful for later, the label of the max category    
                correct_label=labels_dict.get(category) #get the numeric label        
        
                # finally, apply the  bubbles technique version 3 (look function definitions)      
    
                bubbles_number=10
                maxbubbles=40
                bubbles_sigma=sigma #10.24 #6.4
                correct_plane, total_plane, bubbles_vec, acc_vec = vis_funcs.bubbles_technique_3(iterations=bubbles_iterations, 
                                                                                     model=model, 
                                                                                     imagelist=inputimg, 
                                                                                     bubbles=bubbles_number, 
                                                                                     sigma=bubbles_sigma, 
                                                                                     img_label_idx=correct_label,
                                                                                     acc_baseline=predictions[0,correct_label],
                                                                                     maxbubbles=maxbubbles,
                                                                                     usecomplementmask=True,
                                                                                     offsetpx=10)
                
                
                # convert total_plane and correct_plane from lists to normalized images
                correctplane_img=vis_funcs.bubbles_correct_plane(correct_plane, normalizeimg=False)
                totalplane_img=vis_funcs.bubbles_total_plane(total_plane,normalizeimg=False)
                
                correctplaneS=vis_funcs.scale(correctplane_img) #S stands for scaled
                totalplaneS=vis_funcs.scale(totalplane_img)
                
                correctplaneN=vis_funcs.normalize(correctplane_img) #N stnds for Normalized
                totalplaneN=vis_funcs.normalize(totalplane_img)
                
                bubbleplaneSS=vis_funcs.scale(correctplaneS/totalplaneS) #scaled/scaled
                bubbleplaneNS=vis_funcs.scale(correctplaneN/totalplaneS) #normalized/scaled
                
                # RO(img)  Remove Outliers
                correctplaneRO=vis_funcs.bubbles_remove_img_outliers(correctplane_img, std_limit_down=3, std_limit_up=4)
                totalplaneRO=vis_funcs.bubbles_remove_img_outliers(totalplane_img, std_limit_down=3, std_limit_up=4)
                
                # RO(Norm(img))
                correctplaneRON=vis_funcs.bubbles_remove_img_outliers(correctplaneN, std_limit_down=3, std_limit_up=4)
                totalplaneRON=vis_funcs.bubbles_remove_img_outliers(totalplaneN, std_limit_down=3, std_limit_up=4)
                
                # Norm(RO(img))
                correctplaneNRO=vis_funcs.normalize(correctplaneRO)
                totalplaneNRO=vis_funcs.normalize(totalplaneRO)
                                
                correctplaneSRO=vis_funcs.scale(correctplaneRO)
                totalplaneSRO=vis_funcs.scale(totalplaneRO)                
                bubbleplaneNS_RO=vis_funcs.scale(correctplaneNRO/totalplaneSRO)
                bubbleplaneSS_RO=vis_funcs.scale(correctplaneSRO/totalplaneSRO)             
                bubbleplaneNN=vis_funcs.normalize(correctplaneRON/(totalplaneRON+1e-5))                                
                bubbleplaneN_SS=vis_funcs.normalize(bubbleplaneSS)               
                                
             
                
                bubblesdict['bubbleplaneN_SS_'+gender+'_'+str(category)+'_s'+sigmapointless].append(bubbleplaneN_SS)
                bubblesdict['bubbleplaneSS_'+gender+'_'+str(category)+'_s'+sigmapointless].append(bubbleplaneSS)     
                bubblesdict['bubbleplaneNS_'+gender+'_'+str(category)+'_s'+sigmapointless].append(bubbleplaneNS)              
                bubblesdict['correctplane_'+gender+'_'+str(category)+'_s'+sigmapointless].append(correctplane_img)
                bubblesdict['totalplane_'+gender+'_'+str(category)+'_s'+sigmapointless].append(totalplane_img)

                
                # acc_vec contains all the classification results across trials, we are only interested in the vector of the correct label
                real_acc_vec=[]
                for acc in acc_vec:
                    real_acc_vec.append(acc[0,correct_label])
                    
                bubblesdict['accvec_'+gender+'_'+str(category)+'_s'+sigmapointless].append(real_acc_vec) #append real_acc_vec list to the dictionary entry
                
                #====================================================================
                
                finalsavedirectory=os.path.join(pathtomodel,'KDEF_all_bubbles','sigma'+sigmapointless,category)        #check if savedirectory exists otherwise create it
                if not os.path.exists(finalsavedirectory):
                    os.makedirs(finalsavedirectory)
                
                fig, axes = plt.subplots(2,2, figsize=(12, 11))       
                axes[0,0].axis('off')
                im1=axes[0,0].imshow(vis_funcs.normalize(baseimage)) 
                axes[1,0].axis('off')
                im2=axes[1,0].imshow(vis_funcs.normalize(baseimage))   
                im2=axes[1,0].imshow(vis_funcs.normalize(bubbleplaneSS),alpha=0.55,cmap='jet')   
                axes[1,0].set_title('scaled(Correct_Plane)/scaled(Total_Plane) ',fontsize=12)
                fig.colorbar(im2,ax=axes[1,0],ticks=[0.1,0.3,0.5,0.7,0.9], orientation ='vertical') 
                axes[1,1].axis('off')
                im2=axes[1,1].imshow(vis_funcs.normalize(baseimage))   
                im2=axes[1,1].imshow(bubbleplaneNS,alpha=0.55,cmap='jet')   
                axes[1,1].set_title('norm(Correct_Plane)/scaled(Total_Plane) \n No Outliers',fontsize=12)
                fig.colorbar(im2,ax=axes[1,1],ticks=[0.1,0.3,0.5,0.7,0.9], orientation ='vertical')               
                
                axes[0,1].set_title('Image: {} {}'.format(category,file), fontsize=14)
                axes[0,1].imshow(vis_funcs.normalize(baseimage)*0.2+0.8)
                axes[0,1].text(10, 65, 
                    '''
                    category: {}\n
                    Acc baseline: {:.2f}\n
                    Iterations: {}\n
                    sigma: {}\n
                    Mean bubbles: {:.2f}\n
                    '''.format(category,predictions[0,correct_label], bubbles_iterations, 
                        sigma, np.mean(bubbles_vec)),
                      ha="left", va="center",color='k' , fontsize=14)
                plt.suptitle('Bubbles technique (Version 3) \n Image: {} {}'.format(category,file), fontsize=15)  
                plt.savefig(os.path.join(finalsavedirectory,'{}_{}_full_BUBBLES.png'.format(category,file.split('.')[0])),dpi=100)
                plt.close()     
        
                #====================================================================
                
                
    #%% Save megadictionary
        
    
    with open(pathtomodel+os.sep+'bubblesdict.pickle', 'wb') as f:
        pickle.dump(bubblesdict, f)        
                
    #with open('train.pickle', 'rb') as f:
    #    bubblesdictionary = pickle.load(f)
    #%%  Now that the megadictionary has been constructed, plot the --AVERAGE-- images
    
    # norm(sum_all([scaled/scaled]))  
    for sigma in sigma_vec:              
        sigmapointless=str(sigma).replace('.','') #remove point from sigma
        finalsavedirectory=os.path.join(pathtomodel,savedirectory,'sigma'+sigmapointless+'_SS')       #check if savedirectory exists otherwise create it
        if not os.path.exists(finalsavedirectory):
            os.makedirs(finalsavedirectory)
        for gender in ['F','M']:
            for category in list(labels_dict.keys()):
                bubblesplane_total=np.zeros((128,128),np.float32) 
                for plane in bubblesdict['bubbleplaneSS_'+gender+'_'+str(category)+'_s'+sigmapointless]:
                    bubblesplane_total=bubblesplane_total+plane
                bubblesplane_total=vis_funcs.normalize(bubblesplane_total)
            
                AKDEFfile=gender+AKDEF_labels.get(category)+'S.JPG' #get the AKDEF file to plot over it
                AKDEFimg=skimage.io.imread(AKDEF_path + os.sep +  AKDEFfile)
                AKDEFimg=resize(AKDEFimg, (128,128),preserve_range=False, anti_aliasing=True)
             
                fig, axes = plt.subplots(1,2, figsize=(13, 7))       
                axes[0].axis('off')
                im1=axes[0].imshow(AKDEFimg,cmap='gray')   
                axes[0].set_title('AKDEF image. Gender={}, Emotion={}'.format(gender,category),fontsize=12)
                axes[1].axis('off')
                im2=axes[1].imshow(AKDEFimg,cmap='gray')  
                im2=axes[1].imshow(bubblesplane_total,alpha=0.55,cmap='jet')   
                axes[1].set_title('Average scaled_correct/scaled_total Plane ',fontsize=12)
                plt.suptitle('Average Bubbles over KDEF dataset. \n Average scaled_correct/scaled_total Plane \n b_sigma={}, iter={}'.format(sigma, bubbles_iterations),fontsize=14)
                plt.savefig(finalsavedirectory+os.sep+'avgBubbles_KDEF_{}_{}_SS.png'.format(gender,category),dpi=100)
                plt.close()
    
    
    
    # norm(sum_all([norm(scaled/scaled)]))                 
    for sigma in sigma_vec:              
        sigmapointless=str(sigma).replace('.','') #remove point from sigma
        finalsavedirectory=os.path.join(pathtomodel,savedirectory,'sigma'+sigmapointless+'_NSS')        #check if savedirectory exists otherwise create it
        if not os.path.exists(finalsavedirectory):
            os.makedirs(finalsavedirectory)
        for gender in ['F','M']:
            for category in list(labels_dict.keys()):
                bubblesplane_total=np.zeros((128,128),np.float32) 
                for plane in bubblesdict['bubbleplaneN_SS_'+gender+'_'+str(category)+'_s'+sigmapointless]:
                    bubblesplane_total=bubblesplane_total+plane
                bubblesplane_total=vis_funcs.normalize(bubblesplane_total)
            
                AKDEFfile=gender+AKDEF_labels.get(category)+'S.JPG'
                AKDEFimg=skimage.io.imread(AKDEF_path + os.sep +  AKDEFfile)
                AKDEFimg=resize(AKDEFimg, (128,128),preserve_range=False, anti_aliasing=True)
             
                fig, axes = plt.subplots(1,2, figsize=(13, 7))       
                axes[0].axis('off')
                im1=axes[0].imshow(AKDEFimg,cmap='gray')   
                axes[0].set_title('AKDEF image. Gender={}, Emotion={}'.format(gender,category),fontsize=12)
                axes[1].axis('off')
                im2=axes[1].imshow(AKDEFimg,cmap='gray')  
                im2=axes[1].imshow(bubblesplane_total,alpha=0.55,cmap='jet')   
                axes[1].set_title('Average norm_scaled_Correct/norm_scaled_Total',fontsize=12)
                plt.suptitle('Average Bubbles over KDEF dataset.\n Average norm_scaled_Correct/norm_scaled_Total \n b_sigma={}, iter={}'.format(sigma, bubbles_iterations),fontsize=14)
                plt.savefig(finalsavedirectory+os.sep+'avgBubbles_KDEF_{}_{}_NSS.png'.format(gender,category),dpi=100)
                plt.close()
    


    # norm(sum_all([norm/scaled]))            
    for sigma in sigma_vec:              
        sigmapointless=str(sigma).replace('.','') #remove point from sigma
        finalsavedirectory=os.path.join(pathtomodel,savedirectory,'sigma'+sigmapointless+'_NS')        #check if savedirectory exists otherwise create it
        if not os.path.exists(finalsavedirectory):
            os.makedirs(finalsavedirectory)
        for gender in ['F','M']:
            for category in list(labels_dict.keys()):
                bubblesplane_total=np.zeros((128,128),np.float32) 
                for plane in bubblesdict['bubbleplaneNS_'+gender+'_'+str(category)+'_s'+sigmapointless]:
                    bubblesplane_total=bubblesplane_total+plane
                bubblesplane_total=vis_funcs.normalize(bubblesplane_total)
            
                AKDEFfile=gender+AKDEF_labels.get(category)+'S.JPG'
                AKDEFimg=skimage.io.imread(AKDEF_path + os.sep +  AKDEFfile)
                AKDEFimg=resize(AKDEFimg, (128,128),preserve_range=False, anti_aliasing=True)
             
                fig, axes = plt.subplots(1,2, figsize=(13, 7))       
                axes[0].axis('off')
                im1=axes[0].imshow(AKDEFimg,cmap='gray')   
                axes[0].set_title('AKDEF image. Gender={}, Emotion={}'.format(gender,category),fontsize=12)
                axes[1].axis('off')
                im2=axes[1].imshow(AKDEFimg,cmap='gray')  
                im2=axes[1].imshow(bubblesplane_total,alpha=0.55,cmap='jet')   
                axes[1].set_title('Average norm_Correct/scaled_Total',fontsize=12)
                plt.suptitle('Average Bubbles over KDEF dataset.\n Average norm_Correct/scaled_Total \n b_sigma={}, iter={}'.format(sigma, bubbles_iterations),fontsize=14)
                plt.savefig(finalsavedirectory+os.sep+'avgBubbles_KDEF_{}_{}_NS.png'.format(gender,category),dpi=100)
                plt.close()

    
    end=time.time()-start 
    print('... \n =(1 KDEF_bubbles_averaging)==== {:.2f} minutes ===='.format(end/60))    


#%% = 2 ======= 2 ================ 2 ================= 2 ========================
#   2) KDEF_saliency_average
#   =========================================================================

if KDEF_saliency_average: #controlled at the beginning of the script

    if num_classes==5:
        if platform=='win32':
            data_path='C:\\Users\\andre\\Documents\\Torino\\colliculus\\datasets\\KDEF_frontal'
        else:
            data_path= '/home/andres/Documents/colliculus/datasets/KDEF_frontal'   #KDEF dataset
        labels_dict={'afraid':0, 'angry': 1, 'happy':2, 'neutral':3, 'sad':4} 
        AKDEF_labels={'afraid':'AF', 'angry': 'AN', 'happy':'HA', 'neutral':'NE', 'sad':'SA'}
    elif num_classes==3:
        if platform=='win32':
            data_path='C:\\Users\\andre\\Documents\\Torino\\colliculus\\datasets\\KDEF_frontal_3class'
        else:
            data_path= '/home/andres/Documents/colliculus/datasets/KDEF_frontal_3class'   #KDEF dataset
        labels_dict={'happy':0, 'neutral':1, 'sad':2} 
        AKDEF_labels={'happy':'HA', 'neutral':'NE', 'sad':'SA'}
    else: 
        raise Exception("num_classes should be either 3 or 5" )
        
    if platform=='win32': 
        AKDEF_path='C:\\Users\\andre\\Documents\\Torino\\colliculus\\datasets\\AKDEF_square'
    else:
        AKDEF_path='/home/andres/Documents/colliculus/datasets/AKDEF_square'
        
    savedirectory='KDEF_avg_Saliency_test'


#%% =========  get saliency activations in KDEF dataset for averaging
    
    
    start=time.time()
    
    saliencydict={'subj':[]}
    for gen in ['F','M']:
        for cat in list(labels_dict.keys()):
            saliencydict['Saliency_v1_'+gen+'_'+str(cat)]=[]
            saliencydict['Saliency_v2_'+gen+'_'+str(cat)]=[]
            
    for root, dirs, files in os.walk(data_path, topdown=True):
        for file in files:      
            print(file)
                   
            category=root.split(os.sep)[-1] #extract CATEGORY string from the root directory, could be done otherwise with the dict or a list
            gender=file[1]
            
            saliencydict['subj'].append([category,file])
            
            #print('working on image: {}  {}'.format(category, file))
                
            # load base image============================================================== 
            baseimage = skimage.io.imread(root + os.sep +  file)
            baseimage=resize(baseimage, (128,128),preserve_range=True, anti_aliasing=True)
            inputimg=baseimage.astype(np.float32)/255.
            inputimg=inputimg[np.newaxis,...]# Make it a batch of one.
         
            #predictions=model.predict(inputimg) #get PREDICTIONS
            #label_index = np.argmax(predictions) #useful for later, the label of the max category    
            #correct_label=labels_dict.get(category) #get the numeric label        
    
    
            # ==============================================================================    
            # Attention Layer Activations =================================================   
            # get all attention masks and combine them in a single one    
    
            layer_names=['reshape_2','reshape_5','reshape_8'] #,reshape_11, reshape_14' #!!!!!!!!! (always check the model's graph plot in order to be sure which)
            masks=[]
            for layer in layer_names:
                outputs=vis_funcs.read_layer_singleinputmodel(model, inputimg, layer)
                mask=np.sum(outputs, axis=-1) #sum all the masks from a single AttModule to get a single 2D mask
                mask=mask.squeeze()
                mask = skimage.transform.resize(vis_funcs.normalize(mask), (128,128), order=1).copy()
                masks.append(mask)
            masks=np.stack(masks) # stack the resulting masks from the AttModules                
            mask=np.sum(masks, axis=0) #sum the masks from the AttModules and get a single mask for ALL the AttModules
            mask=vis_funcs.normalize(mask)
    
            saliencydict['Saliency_v1_'+gender+'_'+str(category)].append(mask)
    
            #====================================================================       
            finalsavedirectory=os.path.join(pathtomodel,'KDEF_all_saliency','v1',category)        #check if savedirectory exists otherwise create it
            if not os.path.exists(finalsavedirectory):
                os.makedirs(finalsavedirectory)
            
            titles=['uSGS', 'lSGS','SO']# 'uSGS_02',
            fig, axes = plt.subplots(2,2, figsize=(9, 10))
            # overall composite mask over image (previously calculated)
            axes[0,0].axis("off")
            axes[0,0].imshow(vis_funcs.normalize(baseimage)) 
            axes[0,0].imshow(mask,alpha=0.7,cmap='jet')
            axes[0,0].set_title('Total Salience map V1', fontsize=14, color='r')  
            #calcuate the rest
            for n,layer in enumerate(layer_names):
                outputs=vis_funcs.read_layer_singleinputmodel(model, inputimg, layer)
                outputs[outputs<.7]=0 #to avoid problems when doing boolean operations
                mask=np.any(outputs, axis=-1)
                mask = skimage.transform.resize(vis_funcs.normalize(mask), (128,128), order=1).copy()
                i=0 if n==0 else 1
                j=n+1 if n==0 else n-1            
                axes[i,j].imshow(mask,alpha=0.7,cmap='jet')            
                axes[i,j].axis("off")
                #axes[i,j].set_title('i{} j{} n{}'.format(i,j,n), fontsize=14)  
                axes[i,j].set_title(titles[n], fontsize=14)               
                axes[i,j].imshow(vis_funcs.normalize(baseimage)) 
                axes[i,j].imshow(mask,alpha=0.7,cmap='jet')
            plt.suptitle('Salience Maps V1 \n Image: {} {}'.format(category,file))    
            plt.savefig(os.path.join(finalsavedirectory,'{}_{}_saliency_v1.png'.format(category,file)),dpi=100)
            plt.close()   
            #==================================
    
    
    
    
    
    
            #======= SECOND WAY TO COMBINE MASKS!!!!! (using logical OR instead of summation)
            # get all attention masks and combine them in a single one    
            layer_names=['reshape_2','reshape_5','reshape_8'] 
            masks=[]
            for layer in layer_names:
                outputs=vis_funcs.read_layer_singleinputmodel(model, inputimg, layer)
                outputs[outputs<.7]=0 #threshold, useful for the first layers, pay attention
                mask=np.any(outputs, axis=-1) #Logical OR across last dimension
                mask=mask.squeeze()
                mask = skimage.transform.resize(vis_funcs.normalize(mask), (128,128), order=1).copy()
                masks.append(mask)
            masks=np.stack(masks) # stack the resulting masks from the AttModules
            mask=np.sum(masks, axis=0) #sum the masks from the AttModules and get a single mask for ALL the AttModules
            mask=vis_funcs.normalize(mask) #normaize to [0,1]
    
    
            saliencydict['Saliency_v2_'+gender+'_'+str(category)].append(mask)
    
            #====================================================================        
            finalsavedirectory=os.path.join(pathtomodel,'KDEF_all_saliency','v2',category)        #check if savedirectory exists otherwise create it
            if not os.path.exists(finalsavedirectory):
                os.makedirs(finalsavedirectory)
            
            titles=['uSGS', 'lSGS','SO']# 'uSGS_02',
            fig, axes = plt.subplots(2,2, figsize=(9, 10))
            # overall composite mask over image (previously calculated)
            axes[0,0].axis("off")
            axes[0,0].imshow(vis_funcs.normalize(baseimage)) 
            axes[0,0].imshow(mask,alpha=0.7,cmap='jet')
            axes[0,0].set_title('Total Salience map V2', fontsize=14, color='r')  
            #calcuate the rest
            for n,layer in enumerate(layer_names):
                outputs=vis_funcs.read_layer_singleinputmodel(model, inputimg, layer)
                outputs[outputs<.7]=0 #to avoid problems when doing boolean operations
                mask=np.any(outputs, axis=-1)
                mask = skimage.transform.resize(vis_funcs.normalize(mask), (128,128), order=1).copy()
                i=0 if n==0 else 1
                j=n+1 if n==0 else n-1            
                axes[i,j].imshow(mask,alpha=0.7,cmap='jet')            
                axes[i,j].axis("off")
                #axes[i,j].set_title('i{} j{} n{}'.format(i,j,n), fontsize=14)  
                axes[i,j].set_title(titles[n], fontsize=14)               
                axes[i,j].imshow(vis_funcs.normalize(baseimage)) 
                axes[i,j].imshow(mask,alpha=0.7,cmap='jet')
            plt.suptitle('Salience Maps V2 \n Image: {} {}'.format(category,file))    
            plt.savefig(os.path.join(finalsavedirectory,'{}_{}_saliency_v2.png'.format(category,file)),dpi=100)
            plt.close()   
            #==================================
    
                    
    #%% Save megadictionary
        
    with open(pathtomodel+os.sep+'saliencydict.pickle', 'wb') as f:
        pickle.dump(saliencydict, f)        
                
    #with open('train.pickle', 'rb') as f:
    #    bubblesdictionary = pickle.load(f)
    #%%  PLOT images using the megadictionary    
    # using untouched tensors (keeping outliers and such)
        
    for version in ['v1','v2']:              
        finalsavedirectory=os.path.join(pathtomodel,savedirectory,version)        #check if savedirectory exists otherwise create it
        if not os.path.exists(finalsavedirectory):
            os.makedirs(finalsavedirectory)
        for gender in ['F','M']:
            for category in list(labels_dict.keys()):
                saliency_total=np.zeros((128,128),np.float32) 
                for plane in saliencydict['Saliency_'+version+'_'+gender+'_'+category]:
                    saliency_total=saliency_total+plane
                saliency_total=vis_funcs.normalize(saliency_total)
            
                AKDEFfile=gender+AKDEF_labels.get(category)+'S.JPG'
                AKDEFimg=skimage.io.imread(AKDEF_path + os.sep +  AKDEFfile)
                AKDEFimg=resize(AKDEFimg, (128,128),preserve_range=False, anti_aliasing=True)
             
                fig, axes = plt.subplots(1,2, figsize=(13, 7))       
                axes[0].axis('off')
                im1=axes[0].imshow(AKDEFimg,cmap='gray')   
                axes[0].set_title('AKDEF image. Gender={}, Emotion={}'.format(gender,category),fontsize=12)
                axes[1].axis('off')
                im2=axes[1].imshow(AKDEFimg,cmap='gray')  
                im2=axes[1].imshow(saliency_total,alpha=0.55,cmap='jet')   
                axes[1].set_title('Average Correct_Plane/Total_Plane ',fontsize=12)
                plt.suptitle('Average Internal Saliency maps \n over images in KDEF dataset.',fontsize=14)
                plt.savefig(finalsavedirectory+os.sep+'avgSaliency_KDEF_{}_{}_{}.png'.format(gender,category, version),dpi=100)
                plt.close()
    
    end=time.time()-start 
    print('... \n ==(2) KDEF_saliency_average === {:.2f} minutes ===='.format(end/60))    
        
    


#%% = 3 ======= 3 ================ 3 ================= 3 ========================
#   3) KDEF_freq_model_analysis
#   =========================================================================

if KDEF_freq_model_analysis:
    
    if num_classes==5:
        if platform=='win32':
            data_path='C:\\Users\\andre\\Documents\\Torino\\colliculus\\datasets\\KDEF_frontal'
            KDEF_freq_filtered='C:\\Users\\andre\\Documents\\Torino\\colliculus\\datasets\\KDEF_freq_filtered_originalsize'
        else:
            data_path= '/home/andres/Documents/colliculus/datasets/KDEF_frontal'   #KDEF dataset
            KDEF_freq_filtered='/home/andres/Documents/colliculus/datasets/KDEF_freq_filtered_originalsize'
    elif num_classes==3:
        if platform=='win32':
            data_path='C:\\Users\\andre\\Documents\\Torino\\colliculus\\datasets\\KDEF_frontal_3class'
            KDEF_freq_filtered='C:\\Users\\andre\\Documents\\Torino\\colliculus\\datasets\\KDEF_freq_filtered_originalsize_3class'
        else:
            data_path= '/home/andres/Documents/colliculus/datasets/KDEF_frontal_3class'   #KDEF dataset
            KDEF_freq_filtered='/home/andres/Documents/colliculus/datasets/KDEF_freq_filtered_originalsize_3class'
    else: 
        raise Exception("num_classes should be either 3 or 5" )
    



    
    #   C O N F U S I O N - M A T R I X
    
    
    for freq in ['high_freq','low_freq']:
        # 1.Predict classes
    
        datatestpath=os.path.join(KDEF_freq_filtered,freq)
        
        
        test_generator = ImageDataGenerator(rescale = 1./255)
        test_data_generator = test_generator.flow_from_directory(
            directory=datatestpath,
             target_size=(img_width, img_height),
            batch_size=32,
            shuffle=False)
        test_steps_per_epoch = np.math.ceil(test_data_generator.samples / test_data_generator.batch_size)
        
        predictions = model.predict_generator(test_data_generator, steps=test_steps_per_epoch, verbose=1)
        # Get most likely class
        predicted_classes = np.argmax(predictions, axis=1)
    
        # 2.Get ground-truth classes and class-labels
        truelabels_numeric = test_data_generator.classes
        class_labels = list(test_data_generator.class_indices.keys())   # the category labels taken from the directory names (happy, neutral, sad...)
        truelabels_strings=[class_labels[i] for i in truelabels_numeric] #transform the numeric labels to string labels
        predictedlabels_strings=[class_labels[i] for i in predicted_classes]
        
        # 3. Use scikit-learn to get statistics
        report = metrics.classification_report(truelabels_numeric, predicted_classes, target_names=class_labels)
        print(report)  
        
        confmat = metrics.confusion_matrix(truelabels_numeric, predicted_classes)
        print(confmat)  
    
        df_cm = pd.DataFrame(confmat, index = [i for i in class_labels], columns = [i for i in class_labels])   
        
        fig, ax = plt.subplots(1,2, figsize=(12.5, 5.5))
        sns.heatmap(df_cm, annot=True, cmap='OrRd', fmt='g', square=True, cbar=False, ax=ax[0])
        plt.setp(ax[0], xlabel='Predicted Label')
        plt.setp(ax[0], ylabel='True Label')
        ax[0].set_title('Confusion Matrix')
        ax[1].axis('off')
        ax[1].text(0.88, 0.5, report,
                    ha="right", va="center",color='k' , fontsize=14) 
    
        fig.suptitle('Classification Overview: KDEF dataset,{} stimuli'.format(freq))
        plt.savefig(os.path.join(pathtomodel,'confmat_KDEF_{}.png'.format(freq)), dpi=150)
    
    
        # Calculate and plot UMAP projection ===========
        
        embedding = umap.UMAP().fit_transform(predictions)
        embedding=pd.DataFrame(embedding)
        embedding.columns=['x_umap','y_umap']
        
        #plotting
        cmap=sns.color_palette("husl",n_colors=num_classes)
        fig, ax = plt.subplots(1,2, figsize=(13, 6.2))
        sns.scatterplot(x='x_umap',y='y_umap', data=embedding, legend='full', 
                        hue=truelabels_strings, palette=cmap, s=75,
                        alpha=0.6,  ax=ax[0], hue_order = class_labels)     
        plt.setp(ax[0].get_legend().get_texts(), fontsize='14') # for legend text
        plt.setp(ax[0].get_legend().get_title(), fontsize='22') # for legend title
        ax[0].set_title('True classes')
        plt.xlabel('x-umap')
        plt.ylabel('y-umap')  
        sns.scatterplot(x='x_umap', y='y_umap', data=embedding, legend='full',
                        hue=predictedlabels_strings, palette=cmap, s=75,
                        alpha=0.6, ax=ax[1], hue_order = class_labels)   
        plt.setp(ax[1].get_legend().get_texts(), fontsize='14') # for legend text
        plt.setp(ax[1].get_legend().get_title(), fontsize='22') # for legend title
        #plt.legend(fontsize='x-large', title_fontsize='20')
        ax[1].set_title('Predicted classes')
        plt.subplots_adjust(top=.85)
        
    
        fig.suptitle('UMAP projection, last model layer:\n dataset: {}'.format('\\'.join(datatestpath.split(os.sep)[-2:])), size=16)
        if 'KDEF' in datatestpath.split(os.sep)[-2]:
            plt.savefig(pathtomodel+'\\'+ 'UMAP_KDEF_{}.png'.format(datatestpath.split(os.sep)[-1]))
        else:
            plt.savefig(pathtomodel+'\\'+ 'UMAP_{}.png'.format(datatestpath.split(os.sep)[-1]))
                  


    print('... \n ==(3) KDEF_freq_model_analysis (finished)')    



#%% = 4 ======= 4 ================ 4 ================= 4 ========================
#   4) KDEF_noise_model_analysis
#   Perform the analysis of the model with noisy inputs 
#   =========================================================================


# Define noise function to modify input images
def add_noise_image(noise_typ,image,sigma=1,spamount=0.005):
    """    
    Parameters
    ----------
    image : ndarray
        Input image data. Will be converted to float.
    mode : str
        One of the following strings, selecting the type of noise to add:
    
        'gauss'     Gaussian-distributed additive noise.
        'poisson'   Poisson-distributed noise generated from the data.
        's&p'       Replaces random pixels with 0 or 1.
        'speckle'   Multiplicative noise using out = image + n*image,where
                    n is uniform noise with specified mean & variance.
    """
    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = 0
        #var = 0.1
        #sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        #amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(spamount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
            for i in image.shape]
        out[coords] = 255

      # Pepper mode
        num_pepper = np.ceil(spamount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        mean=1
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)        
        noisy = image * gauss
        return noisy
#    elif noise_typ =="speckle":
#        row,col,ch = image.shape
#        gauss = np.random.randn(row,col,ch)
#        gauss = gauss.reshape(row,col,ch)        
#        noisy = image + image * gauss
#        return noisy
        
    

if KDEF_noise_model_analysis:
    
    
    if platform=='win32':
        basepath='C:\\Users\\andre\\Documents\\Torino\\colliculus'
    else:
        basepath= '/home/andres/Documents/colliculus'
      
    
    if num_classes==5:
        if platform=='win32':
            dataset='datasets\\KDEF_frontal'
        else:
            dataset= 'datasets/KDEF_frontal'   #KDEF dataset
    elif num_classes==3:
        if platform=='win32':
            dataset='datasets\\KDEF_frontal_3class'
        else:
            dataset= 'datasets/KDEF_frontal_3class'   #KDEF dataset
    else: 
        raise Exception("num_classes should be either 3 or 5" )
    


    #define dictionary to hold accuracies and other info (for plotting) ======
    noisedict={'SD_gaussian':[],'category_gaussian':[],'acc_gaussian':[], 
               'amount_sp':[],'category_sp':[], 'acc_sp':[]}

    for _ in range(20): #repeat the noise
    
        # first, evaluate model with GAUSSIAN noise inputs   ===================  
        datatestpath=os.path.join(basepath,dataset)
    
        # create save directory
        finalsavedirectory=os.path.join(pathtomodel,'KDEF_noisy_inputs')        #check if savedirectory exists otherwise create it
        if not os.path.exists(finalsavedirectory):
            os.makedirs(finalsavedirectory)
    
        sdvec=[5,10,20,30,40,50,75]# vector of standard deviations    
        for sd in sdvec:
            def noise_sdvariable(image): #define the processing function in every loop because it needs to have just a single argument
                return add_noise_image('gauss',image,sigma=sd)
            
            test_generator = ImageDataGenerator(rescale = 1./255,
                                                preprocessing_function=noise_sdvariable)
            
            test_data_generator = test_generator.flow_from_directory(
                                                directory=datatestpath,
                                                target_size=(128, 128),
                                                batch_size=32,
                                                shuffle=False)
           
            test_steps_per_epoch = np.math.ceil(test_data_generator.samples / test_data_generator.batch_size)
            predictions = model.predict_generator(test_data_generator, steps=test_steps_per_epoch, verbose=1)
            # Get most likely class
            predicted_classes = np.argmax(predictions, axis=1)
        
            # 2.Get ground-truth classes and class-labels
            truelabels_numeric = test_data_generator.classes
            class_labels = list(test_data_generator.class_indices.keys())   # the category labels taken from the directory names (happy, neutral, sad...)
            truelabels_strings=[class_labels[i] for i in truelabels_numeric] #transform the numeric labels to string labels
            predictedlabels_strings=[class_labels[i] for i in predicted_classes]
            
            # Save values in a dictionary so the distribution can be plotted subsequently
            fbeta=metrics.fbeta_score(truelabels_numeric, predicted_classes, beta=0.5, average=None)
            
            for n,classlabel in enumerate(class_labels): #Fb score by category           
                noisedict['SD_gaussian'].append(sd)
                noisedict['category_gaussian'].append(classlabel)
                noisedict['acc_gaussian'].append(fbeta[n])
            
            noisedict['SD_gaussian'].append(sd) #general accuracy
            noisedict['category_gaussian'].append('accuracy')
            noisedict['acc_gaussian'].append(metrics.accuracy_score(truelabels_numeric,predicted_classes))
            
             
            # 3. Use scikit-learn to get statistics
            report = metrics.classification_report(truelabels_numeric, predicted_classes, target_names=class_labels)
            print(report)  
            
            confmat = metrics.confusion_matrix(truelabels_numeric, predicted_classes)
            print(confmat)  
        
    
    
            df_cm = pd.DataFrame(confmat, index = [i for i in class_labels], columns = [i for i in class_labels])   
            
            fig, ax = plt.subplots(1,2, figsize=(12.5, 5.5))
            sns.heatmap(df_cm, annot=True, cmap='OrRd', fmt='g', square=True, cbar=False, ax=ax[0])
            plt.setp(ax[0], xlabel='Predicted Label')
            plt.setp(ax[0], ylabel='True Label')
            ax[0].set_title('Confusion Matrix')
            ax[1].axis('off')
            ax[1].text(0.88, 0.5, report,
                        ha="right", va="center",color='k' , fontsize=14) 
        
            fig.suptitle('Gaussian noise inputs, sd={} \n KDEF dataset'.format(sd))
            plt.savefig(os.path.join(finalsavedirectory,'gaussnoise_{}sd_confmat_KDEF.png'.format(sd)), dpi=150)
            plt.close()


#   second, evaluate model with salt & pepper noise inputs   ===================  
    for _ in range(20): #repeat the noise

        spvec=[0.005,0.01,0.03, 0.05, 0.075, 0.10]
        for peppa in spvec:
            def noise_sdvariable(image): #define the processing function in every loop because it needs to have just a single argument
                return add_noise_image('s&p',image,spamount=peppa)
    
            test_generator = ImageDataGenerator(rescale = 1./255,
                                                preprocessing_function=noise_sdvariable)
            test_data_generator = test_generator.flow_from_directory(
                directory=datatestpath,
                 target_size=(128, 128),
                batch_size=32,
                shuffle=False)
            test_steps_per_epoch = np.math.ceil(test_data_generator.samples / test_data_generator.batch_size)
            
            predictions = model.predict_generator(test_data_generator, steps=test_steps_per_epoch, verbose=1)
            # Get most likely class
            predicted_classes = np.argmax(predictions, axis=1)
    
            # 2.Get ground-truth classes and class-labels
            truelabels_numeric = test_data_generator.classes
            class_labels = list(test_data_generator.class_indices.keys())   # the category labels taken from the directory names (happy, neutral, sad...)
            truelabels_strings=[class_labels[i] for i in truelabels_numeric] #transform the numeric labels to string labels
            predictedlabels_strings=[class_labels[i] for i in predicted_classes]


            # Save values in a dictionary so the distribution can be plotted subsequently
            fbeta=metrics.fbeta_score(truelabels_numeric, predicted_classes, beta=0.5, average=None)
            for n,classlabel in enumerate(class_labels): #Fb score by category           
                noisedict['amount_sp'].append(peppa)
                noisedict['category_sp'].append(classlabel)
                noisedict['acc_sp'].append(fbeta[n])            
            noisedict['amount_sp'].append(peppa) #general accuracy
            noisedict['category_sp'].append('accuracy')
            noisedict['acc_sp'].append(metrics.accuracy_score(truelabels_numeric,predicted_classes))
            
            # 3. Use scikit-learn to get statistics
            report = metrics.classification_report(truelabels_numeric, predicted_classes, target_names=class_labels)
            print(report)  
            
            confmat = metrics.confusion_matrix(truelabels_numeric, predicted_classes)
            print(confmat)  
            
            df_cm = pd.DataFrame(confmat, index = [i for i in class_labels], columns = [i for i in class_labels])   
            
            fig, ax = plt.subplots(1,2, figsize=(12.5, 5.5))
            sns.heatmap(df_cm, annot=True, cmap='OrRd', fmt='g', square=True, cbar=False, ax=ax[0])
            plt.setp(ax[0], xlabel='Predicted Label')
            plt.setp(ax[0], ylabel='True Label')
            ax[0].set_title('Confusion Matrix')
            ax[1].axis('off')
            ax[1].text(0.88, 0.5, report,
                        ha="right", va="center",color='k' , fontsize=14) 
            
            fig.suptitle('Salt&Pepper noise inputs, amount={} \n KDEF dataset'.format(peppa))
            plt.savefig(os.path.join(finalsavedirectory,'S&P_noise_{}_confmat_KDEF.png'.format(peppa)), dpi=150)
    

    with open(pathtomodel+os.sep+'noisedict.pickle', 'wb') as f:
        pickle.dump(noisedict, f)  
        
    print('... \n ==(4) KDEF_noise_model_analysis (finished)')    




#%%  continue noise analysis but with SPECKLE NOISE
    
    

if KDEF_noise_model_SPECKLE:
    
    
    if platform=='win32':
        basepath='C:\\Users\\andre\\Documents\\Torino\\colliculus'
    else:
        basepath= '/home/andres/Documents/colliculus'
      
    
    if num_classes==5:
        if platform=='win32':
            dataset='datasets\\KDEF_frontal'
        else:
            dataset= 'datasets/KDEF_frontal'   #KDEF dataset
    elif num_classes==3:
        if platform=='win32':
            dataset='datasets\\KDEF_frontal_3class'
        else:
            dataset= 'datasets/KDEF_frontal_3class'   #KDEF dataset
    else: 
        raise Exception("num_classes should be either 3 or 5" )
    





#   first, evaluate model with SPECKLE noise inputs   ===================  

    datatestpath=os.path.join(basepath,dataset)

    # create save directory
    finalsavedirectory=os.path.join(pathtomodel,'KDEF_noisy_inputs')        #check if savedirectory exists otherwise create it
    if not os.path.exists(finalsavedirectory):
        os.makedirs(finalsavedirectory)

    sdvec=[.01,.05, .1, .2,.3, .5, 1, 5]# vector of standard deviations    
    for sd in sdvec:
        def noise_sdvariable(image): #define the processing function in every loop because it needs to have just a single argument
            return add_noise_image('speckle',image,sigma=sd)
        
        test_generator = ImageDataGenerator(rescale = 1./255,
                                            preprocessing_function=noise_sdvariable)
        
        test_data_generator = test_generator.flow_from_directory(
                                            directory=datatestpath,
                                            target_size=(128, 128),
                                            batch_size=32,
                                            shuffle=False)
       
        test_steps_per_epoch = np.math.ceil(test_data_generator.samples / test_data_generator.batch_size)
        predictions = model.predict_generator(test_data_generator, steps=test_steps_per_epoch, verbose=1)
        # Get most likely class
        predicted_classes = np.argmax(predictions, axis=1)
    
        # 2.Get ground-truth classes and class-labels
        truelabels_numeric = test_data_generator.classes
        class_labels = list(test_data_generator.class_indices.keys())   # the category labels taken from the directory names (happy, neutral, sad...)
        truelabels_strings=[class_labels[i] for i in truelabels_numeric] #transform the numeric labels to string labels
        predictedlabels_strings=[class_labels[i] for i in predicted_classes]
        
        # 3. Use scikit-learn to get statistics
        report = metrics.classification_report(truelabels_numeric, predicted_classes, target_names=class_labels)
        print(report)  
        
        confmat = metrics.confusion_matrix(truelabels_numeric, predicted_classes)
        print(confmat)  
    


        df_cm = pd.DataFrame(confmat, index = [i for i in class_labels], columns = [i for i in class_labels])   
        
        fig, ax = plt.subplots(1,2, figsize=(12.5, 5.5))
        sns.heatmap(df_cm, annot=True, cmap='OrRd', fmt='g', square=True, cbar=False, ax=ax[0])
        plt.setp(ax[0], xlabel='Predicted Label')
        plt.setp(ax[0], ylabel='True Label')
        ax[0].set_title('Confusion Matrix')
        ax[1].axis('off')
        ax[1].text(0.88, 0.5, report,
                    ha="right", va="center",color='k' , fontsize=14) 
    
        fig.suptitle('SPECKLE noise inputs, sd={} \n KDEF dataset'.format(sd))
        plt.savefig(os.path.join(finalsavedirectory,'SPECKLE_{}sd_confmat_KDEF.png'.format(sd)), dpi=150)
    




    print('... \n ==(4.5) SPECKLE NOISE (finished)')    



















#%% ======== 5 ========== 5 ============== 5 ================================
# 5) KDEF_partial_occlusion procedure
#   =========================================================================

start=time.time()

if KDEF_partial_occlusion:
         
    # Define Number of classes and other variables ============================   
    if platform=='win32':
        basepath='C:\\Users\\andre\\Documents\\Torino\\colliculus'
    else:
        basepath= '/home/andres/Documents/colliculus'
    

    
    if num_classes==5:
        if platform=='win32':
            dataset='datasets\\KDEF_frontal'
            dataset_baseimage='datasets\\KDEF_frontal'
        else:
            dataset= 'datasets/KDEF_frontal'   #KDEF dataset
            dataset_baseimage='datasets/KDEF_frontal'
        labels_dict={'afraid':0, 'angry': 1, 'happy':2, 'neutral':3, 'sad':4} 
        AKDEF_labels={'afraid':'AF', 'angry': 'AN', 'happy':'HA', 'neutral':'NE', 'sad':'SA'}            
    elif num_classes==3:
        if platform=='win32':
            dataset='datasets\\KDEF_frontal_3class'
            dataset_baseimage= 'datasets\\KDEF_frontal_3class' 
        else:
            dataset= 'datasets/KDEF_frontal_3class'   #KDEF dataset
            dataset_baseimage= 'datasets/KDEF_frontal_3class'
        labels_dict={'happy':0, 'neutral':1, 'sad':2} 
        AKDEF_labels={'happy':'HA', 'neutral':'NE', 'sad':'SA'}
    else: 
        raise Exception("num_classes should be either 3 or 5" )
    

    if platform=='win32': 
        AKDEF_path='C:\\Users\\andre\\Documents\\Torino\\colliculus\\datasets\\AKDEF_square'
    else:
        AKDEF_path='/home/andres/Documents/colliculus/datasets/AKDEF_square'
           

    savedirectory='KDEF_partial_occlusion'
        

    rowscolsvector=[5,7,9,10]
    
    occlusiondict={'subj':[]}
    for gen in ['F','M']:
        for cat in list(labels_dict.keys()):
            for rowscols in rowscolsvector:
                occlusiondict['occlusion_probsmatrix_'+gen+'_'+str(cat)+'_'+str(rowscols)]=[]
                occlusiondict['occlusion_probsnormalized_'+gen+'_'+str(cat)+'_'+str(rowscols)]=[]   
                occlusiondict['occlusion_probsvariation_'+gen+'_'+str(cat)+'_'+str(rowscols)]=[]
            
    for root, dirs, files in os.walk(dataset, topdown=True):
        for file in files:      
            for rowscols in rowscolsvector:
                print(file, f'rowscols={rowscols}')

                    
                category=root.split(os.sep)[-1] #extract CATEGORY string from the root directory, could be done otherwise with the dict or a list
                gender=file[1]
     
                #check if savedirectory exists otherwise create it
                finalsavedirectory=os.path.join(pathtomodel,savedirectory,category)
                if not os.path.exists(finalsavedirectory):
                    os.makedirs(finalsavedirectory)
                        
               
                occlusiondict['subj'].append([category,file])
                
                #print('working on image: {}  {}'.format(category, file))
                    
                # load base image============================================================== 
                baseimage = skimage.io.imread(root + os.sep +  file)
                baseimage=resize(baseimage, (128,128),preserve_range=True, anti_aliasing=True)
                inputimg=baseimage.astype(np.float32)/255.
                inputimg=inputimg[np.newaxis,...]# Make it a batch of one.
    
                predictions=model.predict(inputimg) #get base predictions
                label_index = np.argmax(predictions) #useful for later, the label of the max category    
                correct_label=labels_dict.get(category)
    

            #  Heatmap visualization ======================================================
            #  based on an sliding occlusion of the image and step-wise classification at each occlusion position
                #rowscols=10
                image_hm=vis_funcs.occlusion_tensor(inputimg,rowscols=rowscols,squarecolor=[0.,0.,0.])
                heatmap_y = model.predict(image_hm)
                # Histogram of the classification probabilities across occlusion cases (for the right class)
                probs = heatmap_y[:, correct_label] #choose the label index obtained with the classification of the complete image
    
                # normalize probabilities [0,1] to construct the mask
                mask=[]
                mask = (probs.max() - probs) / (probs.max()-probs.min())
                mask = np.reshape(mask, (rowscols, rowscols)) #reshape probs vector
                #vis_funcs.tensor_summary(mask)    
                #_ = plt.imshow(mask, cmap=plt.cm.Reds)        
        
                probs_matrix = np.reshape(probs, (rowscols, rowscols)) #reshape the probs vector (original values, no normalization like before)        
                vis_funcs.apply_mask_heatmap(baseimage, mask, probs_matrix, order=1, cmap='OrRd_r',
                                             title1='Image: {} {}'.format(category,file))   
                plt.suptitle('Occlusion technique: \n classification by occluding single facial features (correct category)')
                plt.savefig(os.path.join(finalsavedirectory,'{}_{}_{}rows_occlusion.png'.format(category,file.split('.')[0],rowscols)),dpi=100)
                plt.close()
    
    
    
                #append pobability matrix and the normalized version           
                occlusiondict['occlusion_probsmatrix_'+gender+'_'+str(category)+'_'+str(rowscols)].append(probs_matrix)
                #append normalized probmatrix (maybe it wont be necessary)
                occlusiondict['occlusion_probsnormalized_'+gender+'_'+str(category)+'_'+str(rowscols)].append(mask)            

                occlusiondict['occlusion_probsvariation_'+gender+'_'+str(category)+'_'+str(rowscols)].append(probs_matrix-predictions[0][correct_label]) #occlusion matrix - baseline accuracy


    #Save occlusion dictionary to disk
        
    with open(pathtomodel+os.sep+'occlusiondict.pickle', 'wb') as f:
        pickle.dump(occlusiondict, f)                        
#    with open(pathtomodel+os.sep+'occlusiondict.pickle', 'rb') as f:
#        occlusiondict = pickle.load(f)

    end=time.time()-start 
    print('... \n ==(4.1)occlusion === {:.2f} minutes ===='.format(end/60))     
            
       

    #%%  PLOT averages and std using the megadictionary    
    
    finalsavedirectory=os.path.join(pathtomodel,savedirectory,'averages_all')        #check if savedirectory exists otherwise create it
    if not os.path.exists(finalsavedirectory):
        os.makedirs(finalsavedirectory)
    for gender in ['F','M']:
        for category in list(labels_dict.keys()):
            for rowscols in rowscolsvector:                
                
                #calculate average occlusion by gender and category               
                occlusion_mean=np.stack(occlusiondict['occlusion_probsmatrix_'+gender+'_'+category+'_'+str(rowscols)])
                occlusion_mean=np.mean(occlusion_mean,axis=0)
                
                # calculate standard deviation over all occlusions (by gender and category)
                occlusion_std=np.stack(occlusiondict['occlusion_probsmatrix_'+gender+'_'+category+'_'+str(rowscols)])
                occlusion_std=np.std(occlusion_std,axis=0)
                           
                AKDEFfile=gender+AKDEF_labels.get(category)+'S.JPG'
                AKDEFimg=skimage.io.imread(AKDEF_path + os.sep +  AKDEFfile)
                AKDEFimg=resize(AKDEFimg, (128,128),preserve_range=False, anti_aliasing=True)
    
    
                vis_funcs.apply_mask_heatmap_jetcolormap(AKDEFimg, vis_funcs.normalize(occlusion_std), occlusion_std, order=1, cmap1='jet',cmap2='OrRd',
                                             title1='Standard Deviation plot, Image: {}'.format(category), title2='Standard Deviations of accuracies by position')   
                plt.suptitle(f'Occlusion technique, {category}, rowscols ={rowscols}: \n Standard Deviation plots', fontsize=18)
                plt.savefig(os.path.join(finalsavedirectory,'{}_{}rows_{}_STD_occ.png'.format(category, rowscols, gender)),dpi=100)
                plt.close()


                vis_funcs.apply_mask_heatmap_jetcolormap(AKDEFimg, vis_funcs.normalize(occlusion_mean), occlusion_mean, order=1, cmap1='jet_r', cmap2='OrRd_r',title1='Image: {}'.format(category))   
                plt.suptitle(f'Occlusion technique, {category}, rowscols={rowscols} \n Classification when locations occluded', fontsize=18)
                plt.savefig(os.path.join(finalsavedirectory,'{}_{}rows_{}_meanACC_occ.png'.format(category, rowscols, gender)),dpi=100)
                plt.close()




#%%
# Stack occlusion matrices over distinct rowcols and get better spatial estimates

    finalsavedirectory=os.path.join(pathtomodel,savedirectory,'averages_stacked_rowscols')        #check if savedirectory exists otherwise create it
    if not os.path.exists(finalsavedirectory):
        os.makedirs(finalsavedirectory)
        
        
    for gender in ['F','M']:
        for category in list(labels_dict.keys()):
            occlusion_big_order0=[]
            occlusion_big_order1=[]
            variation_big=[]
            for rowscols in rowscolsvector:                
   
                #create occlusion from baseline megamatrix (list) across rowscols
                occlusion=np.stack(occlusiondict['occlusion_probsmatrix_'+gender+'_'+category+'_'+str(rowscols)])
                occlusion_big_order0.append(skimage.transform.resize(occlusion, (occlusion.shape[0],128,128), order=0, mode='edge'))
                occlusion_big_order1.append(skimage.transform.resize(occlusion, (occlusion.shape[0],128,128), order=1, mode='edge'))
                

                #create variation from baseline megamatrix (list of arrays) across rowscols          
                variation=np.stack(occlusiondict['occlusion_probsvariation_'+gender+'_'+category+'_'+str(rowscols)])
                variation_big.append(skimage.transform.resize(variation, (variation.shape[0],128,128), order=1, mode='edge'))
  

                
            occlusion_big_order0=np.concatenate(occlusion_big_order0)  #concatenate big probsmatrices with all rowscols  
            occlusion_big_order1=np.concatenate(occlusion_big_order1)          
            std_occlusion_big=np.std(occlusion_big_order0,axis=0)
            mean_occlusion_big_order0=np.mean(occlusion_big_order0,axis=0)
            mean_occlusion_big_order1=np.mean(occlusion_big_order1,axis=0)        
        
        
            variation_big=np.concatenate(variation_big) #probability variation from baseline when occluded
            mean_variation=np.mean(variation_big, axis=0)
            std_variation=np.std(variation_big, axis=0)
            
            AKDEFfile=gender+AKDEF_labels.get(category)+'S.JPG'
            AKDEFimg=skimage.io.imread(AKDEF_path + os.sep +  AKDEFfile)
            AKDEFimg=resize(AKDEFimg, (128,128),preserve_range=False, anti_aliasing=True)


            #probability avg, interpolation order 0
            fig, axes = plt.subplots(1,2, figsize=(13, 7))       
            axes[0].axis('off')
            im1=axes[0].imshow(AKDEFimg,cmap='gray')   
            axes[0].set_title('AKDEF image. Gender={}, Emotion={}'.format(gender,category),fontsize=12)
            axes[1].axis('off')
            im2=axes[1].imshow(AKDEFimg,cmap='gray')  
            im2=axes[1].imshow(vis_funcs.normalize(mean_occlusion_big_order0),alpha=0.55,cmap='jet')   
            axes[1].set_title('Average probability when occluded',fontsize=12)
            plt.suptitle(f'Occlusion technique, gender={gender}, category={category}.\n Average over rowscols={rowscolsvector}, interpolation=0',fontsize=14)
            plt.savefig(finalsavedirectory+os.sep+'avg_probs_occlusion_{}_{}_order0.png'.format(category,gender),dpi=100)
            plt.close()

            #probability avg, interpolation order 1
            fig, axes = plt.subplots(1,2, figsize=(13, 7))       
            axes[0].axis('off')
            im1=axes[0].imshow(AKDEFimg,cmap='gray')   
            axes[0].set_title('AKDEF image. Gender={}, Emotion={}'.format(gender,category),fontsize=12)
            axes[1].axis('off')
            im2=axes[1].imshow(AKDEFimg,cmap='gray')  
            im2=axes[1].imshow(vis_funcs.normalize(mean_occlusion_big_order1),alpha=0.55,cmap='jet')   
            axes[1].set_title('Average probability when occluded',fontsize=12)
            plt.suptitle(f'Occlusion technique, gender={gender}, category={category}.\n Average over rowscols={rowscolsvector}, interpolation=1',fontsize=14)
            plt.savefig(finalsavedirectory+os.sep+'avg_probs_occlusion_{}_{}_order1.png'.format(category,gender),dpi=100)
            plt.close()

            #probability STD
            fig, axes = plt.subplots(1,2, figsize=(13, 7))       
            axes[0].axis('off')
            im1=axes[0].imshow(AKDEFimg,cmap='gray')   
            axes[0].set_title('AKDEF image. Gender={}, Emotion={}'.format(gender,category),fontsize=12)
            axes[1].axis('off')
            im2=axes[1].imshow(AKDEFimg,cmap='gray')  
            im2=axes[1].imshow(std_occlusion_big, alpha=0.55,cmap='jet')   
            axes[1].set_title('Average Standard Deviation when occluded',fontsize=12)
            plt.suptitle(f'Standard deviation of probabilities when location occluded \n  {gender}, {category}',fontsize=14)
            plt.savefig(finalsavedirectory+os.sep+'std_probs_occlusion_{}_{}.png'.format(category,gender),dpi=100)
            plt.close()


            #VARIATION from baseline accuracy
            fig, axes = plt.subplots(1,3, figsize=(13, 7))       
            axes[0].axis('off')
            im1=axes[0].imshow(AKDEFimg,cmap='gray')   
            axes[0].set_title('AKDEF image. Gender={}, Emotion={}'.format(gender,category),fontsize=12)
            axes[1].axis('off')
            im2=axes[1].imshow(AKDEFimg,cmap='gray')  
            im2=axes[1].imshow(mean_variation, alpha=.8,cmap='seismic_r',vmin=np.min(mean_variation), vmax=-1*np.min(mean_variation))   
            axes[1].set_title('Variation from baseline acc \n when occluded',fontsize=12)
            axes[2].axis('off')
            im2=axes[2].imshow(AKDEFimg,cmap='gray')  
            im2=axes[2].imshow(std_variation, alpha=0.55,cmap='jet')   
            axes[2].set_title('Standard Deviation of variation \n across subjects',fontsize=12)          
            plt.suptitle(f'Variation from baseline when occluded \n  {gender}, {category}',fontsize=14)
            plt.savefig(finalsavedirectory+os.sep+'VARIATION_from_baseline_{}_{}.png'.format(category,gender),dpi=100)
            plt.close()









#%% ======== 6 ========== 6 ============== 6 ================================
# 6) evaluate_customPMK_models_2021.py (this one does a lot)
#   =========================================================================


if general_evaluation_bubbles:

    for a,b in enumerate([KDEF_section4, V6_samples]):    
        print(a,b)
    
        if a==0 and b==True: # if KDEF is true
            print ('KDEF=True')
            KDEF=True
        elif a==0 and b==False:
            print ('KDEF=False')
            continue
        elif a==1 and b==True:
            print('V6_samples=True')
            KDEF=False #execute the V6_samples analysis
        elif a==1 and b==False:
            print('V6_samples=False')
            continue
            
            
        # Define Number of classes and other variables ============================
        
        if platform=='win32':
            basepath='C:\\Users\\andre\\Documents\\Torino\\colliculus'
        else:
            basepath= '/home/andres/Documents/colliculus'
        
        train_or_test='test' #just to reference the test or train folder in V6 dataset
        
        
        if KDEF:
            if num_classes==3:
                imagesdict={'happy':['AF06HAS','AF05HAS','AF19HAS','AM03HAS','AM17HAS'],
                            'neutral':['AF06NES','AF05NES','AF19NES','AM03NES','AM17NES'],
                            'sad':['AF06SAS','AF05SAS','AF19SAS','AM03SAS','AM17SAS']}
                labels_dict={'happy':0, 'neutral':1, 'sad':2}
                if platform=='win32':
                    dataset='datasets\\KDEF_frontal_3class'
                    dataset_baseimage= 'datasets\\KDEF_frontal_3class' 
                else:
                    data_path= 'datasets/KDEF_frontal_3class'  
                    dataset_baseimage= 'datasets/KDEF_frontal_3class' 
            elif num_classes==5:
                # for dataset_V6
                imagesdict={'afraid':['AF06AFS','BM34AFS', 'BM24AFS', 'BF30AFS','BF06AFS','AM19AFS'],
                    'angry':[ 'AF06ANS','AF07ANS','AF19ANS','AF20ANS','AM03ANS','AM17ANS'],
                    'happy':['AF06HAS','AF05HAS','AF19HAS','AM03HAS','AM17HAS'],
                    'neutral':['AF06NES','AF05NES','AF19NES','AM03NES','AM17NES'],
                    'sad':['AF06SAS','AF05SAS','AF19SAS','AM03SAS','AM17SAS']}
                labels_dict={'afraid':0, 'angry': 1, 'happy':2, 'neutral':3, 'sad':4}
                if platform=='win32':
                    dataset='datasets\\KDEF_frontal'
                    dataset_baseimage= 'datasets\\KDEF_frontal' 
                else:
                    dataset= 'datasets/KDEF_frontal'  
                    dataset_baseimage= 'datasets/KDEF_frontal' 
            else: 
                raise Exception("num_classes should be either 3 or 5" )
        else:        
            if num_classes==3:
                imagesdict={'happy':['0001','0006','0050','0059','0118','0119','0168','0174','0218','0190','0188'],
                    'neutral':['0003','0008','0012','0020','0046','0051','0058','0082','0087','0143','0223','0248'],
                    'sad':['0002','0003','0021','0024','0048','0066','0070','0103','0141','0174','0212','0218','0239']}
                labels_dict={'happy':0, 'neutral':1, 'sad':2}
                if platform=='win32':
                    dataset='datasets\\dataset_V6_3class'
                    dataset_baseimage= 'datasets\\dataset_V6_3class' 
                else:
                    dataset= 'datasets/dataset_V6_3class'  
                    dataset_baseimage= 'datasets/dataset_V6_3class' 
            elif num_classes==5:
                # for dataset_V6
                imagesdict={'afraid':['0010', '0008', '0016','0022','0023', '0024','0025', '0032','0055','0064', '0093', '0186', '0248'],
                    'angry':[ '0001','0017','0018','0026','0030','0033','0048','0069','0088','0092','0098','0127','0240'],
                    'happy':['0001','0006','0050','0059','0118','0119','0168','0174','0218','0190','0188'],
                    'neutral':['0003','0008','0012','0020','0046','0051','0058','0082','0087','0143','0223','0248'],
                    'sad':['0002','0003','0021','0024','0048','0066','0070','0103','0141','0174','0212','0218','0239']}
                labels_dict={'afraid':0, 'angry': 1, 'happy':2, 'neutral':3, 'sad':4}
                if platform=='win32':
                    dataset='datasets\\dataset_V6'
                    dataset_baseimage= 'datasets\\dataset_V6' 
                else:
                    dataset= 'datasets/dataset_V6'  
                    dataset_baseimage= 'datasets/dataset_V6' 
            else: 
                raise Exception("num_classes should be either 3 or 5" )
            
        
        #========= start loop
        if KDEF:
            savedirectory='KDEF_test'    
        else:
            savedirectory='AttentionImages'
        
        
        start=time.time()
        for category, imagelist in imagesdict.items():
            print('category: {}'.format(category))
            for image_number in imagelist:
                print('working on image: {}  {}'.format(category, image_number))
                
                #check if savedirectory exists otherwise create it
                finalsavedirectory=os.path.join(pathtomodel,savedirectory,category)
                if not os.path.exists(finalsavedirectory):
                    os.makedirs(finalsavedirectory)
        
        # load base image============================================================== 
                if KDEF:
                    baseimage = skimage.io.imread(os.path.join(basepath, dataset_baseimage, category, image_number + '.JPG'))
                else:    
                    baseimage = skimage.io.imread(os.path.join(basepath, dataset_baseimage, train_or_test, category,train_or_test+'_'+category+'_'+image_number + '.jpg'))
                
                baseimage=resize(baseimage, (128,128),preserve_range=True, anti_aliasing=True)
                inputimg=baseimage.astype(np.float32)/255.
                inputimg=inputimg[np.newaxis,...] # Make it a batch of one. The model expects a batch, not a single image
                predictions=model.predict(inputimg) #get pre
                label_index = np.argmax(predictions) #useful for later, the label of the max category    
                correct_label=labels_dict.get(category)
        
        
        
                fig, (ax0,ax1) = plt.subplots(1,2, figsize=(10, 5))
                ax0.axis("off")
                ax0.set_title('Image: {} {}'.format(category,image_number), fontsize=14)
                ax0.imshow(vis_funcs.normalize(baseimage), interpolation=None)
                #=====
                ax1.axis("off")
                ax1.imshow(vis_funcs.normalize(baseimage)*0.1+0.85)
                #plt.colorbar(im1,ticks=[0.1,0.3,0.5,0.7,0.9], orientation ='vertical')
                #ax1.set_title('YYYYY', fontsize=10)
                if num_classes==3:
                    ax1.text(30, 65, 
                        '''predictions: \n 
                          happy {:.3f} \n
                          neutral {:.3f} \n
                          sad {:.3f}'''.format(predictions[0][0],predictions[0][1], 
                          predictions[0][2]),
                          ha="left", va="center",color='k' , fontsize=14)
                elif num_classes==5:
                    ax1.text(30, 65, 
                    '''predictions: \n 
                      afraid {:.3f} \n 
                      anger {:.3f} \n 
                      happy {:.3f} \n
                      neutral {:.3f} \n
                      sad {:.3f}'''.format(predictions[0][0],predictions[0][1], 
                      predictions[0][2], predictions[0][3], predictions[0][4]),
                      ha="left", va="center",color='k' , fontsize=14)
                else:      
                    raise Exception("num_classes should be either 3 or 5" )  
                plt.savefig(os.path.join(finalsavedirectory,'{}_{}_predictions.png'.format(category,image_number)),dpi=100)
                plt.close()
        
        #  Heatmap visualization ======================================================
        #  based on an sliding occlusion of the image and step-wise classification at each occlusion position
                rowscols=10
                image_hm=vis_funcs.occlusion_tensor(inputimg,rowscols=rowscols,squarecolor=[0.,0.,0.])
                heatmap_y = model.predict(image_hm)
                # Histogram of the classification probabilities across occlusion cases (for the right class)
                probs = heatmap_y[:, correct_label] #choose the label index obtained with the classification of the complete image
        #        vis_funcs.tensor_summary(probs) 
        #        fig=plt.figure(figsize=(8,5)) 
        #        plt.hist(probs)
        #        plt.title('Histogram of classification probabilities \n Image: {} {}'.format(category,image_number))
        #        #plt.show()
        #        plt.savefig(finalsavedirectory+'\\{}_{}_occlusion_histogram.png'.format(category,image_number),dpi=100)
                    
                # normalize probabilities [0,1] to construct the mask
                mask=[]
                mask = (probs.max() - probs) / (probs.max()-probs.min())
                mask = np.reshape(mask, (rowscols, rowscols)) #reshape probs vector
                #vis_funcs.tensor_summary(mask)    
                #_ = plt.imshow(mask, cmap=plt.cm.Reds)        
        
                probs_matrix = np.reshape(probs, (rowscols, rowscols)) #reshape the probs vector (original values, no normalization like before)        
                vis_funcs.apply_mask_heatmap(baseimage, mask, probs_matrix, order=1, cmap='OrRd_r',
                                             title1='Image: {} {}'.format(category,image_number))   
                plt.suptitle('Occlusion technique: \n classification by occluding single facial features (correct category)'.format(category,image_number))
                plt.savefig(os.path.join(finalsavedirectory,'{}_{}_occlusion_heatmap_probs.png'.format(category,image_number)),dpi=100)
                plt.close()
        
        
        
        # variation from baseline when occluded (pyramidal scale, variable size of occlusion element) ====================================      
        
                # interpolation order=1
                vis_funcs.occlusion_variationfrombaseline(model,inputimg,baseimage, correct_label, 
                                                          predictions,category,image_number,
                                                          rowscols_list=[5,7,9,10],order=1)
                plt.savefig(os.path.join(finalsavedirectory,'{}_{}_occlusion_VARIATION.png'.format(category,image_number)),dpi=100)
                
                # interpolation order=0
                vis_funcs.occlusion_variationfrombaseline(model,inputimg, baseimage, correct_label, 
                                                          predictions,category,image_number,
                                                          rowscols_list=[5,7,9,10],order=0)
                plt.savefig(os.path.join(finalsavedirectory,'{}_{}_occlusion_VARIATION_order0.png'.format(category,image_number)),dpi=100)
                plt.close('all')
        
        
        
        # ==============================================================================    
        # Attention Layer Activations =================================================   
                # get all attention masks and combine them in a single one    
        
                layer_names=['reshape_2','reshape_5','reshape_8'] #,reshape_11, reshape_14' #!!!!!!!!! (always check the model's graph plot in order to be sure which)
                masks=[]
                for layer in layer_names:
                    outputs=vis_funcs.read_layer_singleinputmodel(model, inputimg, layer)
                    mask=np.sum(outputs, axis=-1) #sum all the masks from a single AttModule to get a single 2D mask
                    mask=mask.squeeze()
                    mask = skimage.transform.resize(vis_funcs.normalize(mask), (128,128), order=1).copy()
                    masks.append(mask)
                masks=np.stack(masks) # stack the resulting masks from the AttModules
                         
                mask=np.sum(masks, axis=0) #sum the masks from the AttModules and get a single mask for ALL the AttModules
                mask=vis_funcs.normalize(mask)
        
                #overlay of mask on image with side colorbar
                fig=plt.figure(figsize=(6,4))    
                plt.axis('off')
                im1=plt.imshow(vis_funcs.normalize(baseimage))   
                im2=plt.imshow(mask,alpha=0.7,cmap='jet')   
                plt.title('Salience Overlay V1\n Image: {} {}'.format(category,image_number),fontsize=12)
                plt.colorbar(im2,ticks=[0.1,0.3,0.5,0.7,0.9], orientation ='vertical')
                plt.savefig(os.path.join(finalsavedirectory,'{}_{}_Attention_overlay_V1.png'.format(category,image_number)),dpi=100)
                #plt.show()
        
                
                
                titles=['uSGS', 'lSGS','SO'] 
                fig, axes = plt.subplots(2,2, figsize=(9, 10))
                # overall composite mask over image
                axes[0,0].axis("off")
                axes[0,0].imshow(vis_funcs.normalize(baseimage)) 
                axes[0,0].imshow(mask,alpha=0.7,cmap='jet')
                axes[0,0].set_title('Total Salience map V1', fontsize=14, color='r')  
                for n,layer in enumerate(layer_names):
                    outputs=vis_funcs.read_layer_singleinputmodel(model, inputimg, layer)
                    outputs[outputs<.5]=0 #to avoid problems when doing boolean operations
                    mask=np.sum(outputs, axis=-1) #sum all the masks from a single AttModule to get a single 2D mask  
                    mask = skimage.transform.resize(vis_funcs.normalize(mask), (128,128), order=1).copy()
                    i=0 if n==0 else 1
                    j=n+1 if n==0 else n-1
                    axes[i,j].imshow(mask,alpha=0.7,cmap='jet')            
                    axes[i,j].axis("off")
                    #axes[i,j].set_title('i{} j{} n{}'.format(i,j,n), fontsize=14)  
                    axes[i,j].set_title(titles[n], fontsize=14)               
                    axes[i,j].imshow(vis_funcs.normalize(baseimage)) 
                    axes[i,j].imshow(mask,alpha=0.7,cmap='jet')
                plt.suptitle('Salience Maps V1 \n Image: {} {}'.format(category,image_number))    
                plt.savefig(os.path.join(finalsavedirectory,'{}_{}_SalienceMaps_by_SClayer.png'.format(category,image_number)),dpi=100)
                #plt.show()     
        
        
        
        
        #======= SECOND WAY TO COMBINE MASKS!!!!! (using logical OR instead of summation)
                # get all attention masks and combine them in a single one    
                layer_names=['reshape_2','reshape_5','reshape_8'] 
                masks=[]
                for layer in layer_names:
                    outputs=vis_funcs.read_layer_singleinputmodel(model, inputimg, layer)
                    outputs[outputs<.7]=0 #threshold, useful for the first layers, pay attention
                    mask=np.any(outputs, axis=-1) #Logical OR across last dimension
                    mask=mask.squeeze()
                    mask = skimage.transform.resize(vis_funcs.normalize(mask), (128,128), order=1).copy()
                    masks.append(mask)
                masks=np.stack(masks) # stack the resulting masks from the AttModules
                mask=np.sum(masks, axis=0) #sum the masks from the AttModules and get a single mask for ALL the AttModules
                mask=vis_funcs.normalize(mask) #normaize to [0,1]
        
        
                #overlay of mask on image with side colorbar
                fig=plt.figure(figsize=(6,4))    
                plt.axis('off')
                im1=plt.imshow(vis_funcs.normalize(baseimage))   
                im2=plt.imshow(mask,alpha=0.7,cmap='jet')   
                plt.title('Salience Overlay V2 \n Image: {} {}'.format(category,image_number),fontsize=12)
                plt.colorbar(im2,ticks=[0.1,0.3,0.5,0.7,0.9], orientation ='vertical')
                plt.savefig(os.path.join(finalsavedirectory,'{}_{}_Attention_overlay_V2.png'.format(category,image_number)),dpi=100)
                plt.close()
        
        
                titles=['uSGS', 'lSGS','SO']# 'uSGS_02',
                fig, axes = plt.subplots(2,2, figsize=(9, 10))
                # overall composite mask over image (previously calculated)
                axes[0,0].axis("off")
                axes[0,0].imshow(vis_funcs.normalize(baseimage)) 
                axes[0,0].imshow(mask,alpha=0.7,cmap='jet')
                axes[0,0].set_title('Total Salience map V2', fontsize=14, color='r')  
                #calcuate the rest
                for n,layer in enumerate(layer_names):
                    outputs=vis_funcs.read_layer_singleinputmodel(model, inputimg, layer)
                    outputs[outputs<.7]=0 #to avoid problems when doing boolean operations
                    mask=np.any(outputs, axis=-1)
                    mask = skimage.transform.resize(vis_funcs.normalize(mask), (128,128), order=1).copy()
                    i=0 if n==0 else 1
                    j=n+1 if n==0 else n-1            
                    axes[i,j].imshow(mask,alpha=0.7,cmap='jet')            
                    axes[i,j].axis("off")
                    #axes[i,j].set_title('i{} j{} n{}'.format(i,j,n), fontsize=14)  
                    axes[i,j].set_title(titles[n], fontsize=14)               
                    axes[i,j].imshow(vis_funcs.normalize(baseimage)) 
                    axes[i,j].imshow(mask,alpha=0.7,cmap='jet')
                plt.suptitle('Salience Maps V2 \n Image: {} {}'.format(category,image_number))    
                plt.savefig(os.path.join(finalsavedirectory,'{}_{}_SalienceMaps_by_SClayer_2ndProcessing.png'.format(category,image_number)),dpi=100)
                plt.close()   
        
        
        end=time.time()-start 
        print('... \n ==(4.1)saliency etc === {:.2f} minutes ===='.format(end/60))     
        
        
        
        
        
        
        
        #% ========Bubbles_type3==================================================================
        #  ==========================================================================
        
        if KDEF:
            savedirectory='KDEF_test/KDEF_Bubbles_type3_102'    
        else:
            savedirectory='Bubbles_type3'
        
        start=time.time()
        
        for category, imagelist in sorted(imagesdict.items()):
            print('category: {}'.format(category))
            for image_number in imagelist:
                print('working on image: {}  {}'.format(category, image_number))
                #check if savedirectory exists otherwise create it
                finalsavedirectory=os.path.join(pathtomodel,savedirectory,category)
                if not os.path.exists(finalsavedirectory):
                    os.makedirs(finalsavedirectory)
        
        # load base image============================================================== 
                if KDEF:
                    baseimage = skimage.io.imread(os.path.join(basepath, dataset_baseimage, category, image_number + '.JPG'))
                else:    
                    baseimage = skimage.io.imread(os.path.join(basepath, dataset_baseimage, train_or_test, category, train_or_test +'_'+ category +'_'+ image_number + '.jpg'))
              
                baseimage=resize(baseimage, (128,128),preserve_range=True, anti_aliasing=True)
                inputimg=baseimage.astype(np.float32)/255.
                # Make it a batch of one. The model expects a batch, not a single image
                inputimg=inputimg[np.newaxis,...]
             
                predictions=model.predict(inputimg) #get PREDICTIONS
                label_index = np.argmax(predictions) #useful for later, the label of the max category    
                correct_label=labels_dict.get(category)
                
                # finally, apply the  bubbles technique version 3 (look function definitions)      
                bubbles_iterations=500
                bubbles_number=10
                maxbubbles=100
                bubbles_sigma=10.24 #6.4
                correct_plane, total_plane, bubbles_vec, acc_vec = vis_funcs.bubbles_technique_3(iterations=bubbles_iterations, 
                                                                                     model=model, 
                                                                                     imagelist=inputimg, 
                                                                                     bubbles=bubbles_number, 
                                                                                     sigma=bubbles_sigma, 
                                                                                     img_label_idx=correct_label,
                                                                                     acc_baseline=predictions[0,correct_label],
                                                                                     maxbubbles=maxbubbles)
        
                
                # convert total_plane and correct_plane from lists to normalized images
                correctplane_img=vis_funcs.bubbles_correct_plane(correct_plane)
                totalplane_img=vis_funcs.bubbles_total_plane(total_plane)
                
                # acc_vec contains all the classification results across trials, we are only interested in the vector of the correct label
                real_acc_vec=[]
                for acc in acc_vec:
                    real_acc_vec.append(acc[0,correct_label])
                
                # remove outliers from the images, typically in the corners
                totalplane_img=vis_funcs.bubbles_remove_img_outliers(totalplane_img, std_limit_down=2, std_limit_up=4)
                correctplane_img=vis_funcs.bubbles_remove_img_outliers(correctplane_img, std_limit_down=2, std_limit_up=4)
        
                
                
                #====================================================================
                fig, axes = plt.subplots(2,2, figsize=(13, 10))       
                axes[0,0].axis('off')
                im1=axes[0,0].imshow(vis_funcs.normalize(baseimage))   
                im2=axes[0,0].imshow(vis_funcs.normalize(correctplane_img/(totalplane_img+1e-5)),alpha=0.55,cmap='jet')   
                axes[0,0].set_title('Correct_Plane/Total_Plane ',fontsize=12)
                fig.colorbar(im2,ax=axes[0,0],ticks=[0.1,0.3,0.5,0.7,0.9], orientation ='vertical') 
                axes[0,1].axis('off')
                im1=axes[0,1].imshow(vis_funcs.normalize(baseimage))                   
                axes[1,0].axis("off")
                axes[1,0].set_title('Image: {} {}'.format(category,image_number), fontsize=14)
                axes[1,0].imshow(vis_funcs.normalize(baseimage)*0.2+0.8)
                axes[1,0].text(15, 65, 
                    '''
                    category: {}\n
                    Acc baseline: {:.2f}\n
                    Iterations: {}\n
                    sigma: {}\n
                    Mean bubbles: {:.2f}\n
                    '''.format(category,predictions[0,correct_label], bubbles_iterations, 
                        bubbles_sigma, np.mean(bubbles_vec)),
                      ha="left", va="center",color='k' , fontsize=14)
                axes[1,1].hist(real_acc_vec)
                axes[1,1].axvline(x=predictions[0,correct_label], color='r',linestyle='--', label='baseline accuracy \ncategory: {}'.format(category))
                #axes[1,1].axvline(x=relative_acc_threshold*predictions[0,correct_label], color='b',linestyle='--', label='Acc Threshold'.format(category))
                axes[1,1].legend()
                axes[1,1].set_xlabel('Accuracy')
                axes[1,1].set_ylabel('Count')        
                axes[1,1].set_title('Histogram of accuracy across Bubbles trials ')
                plt.suptitle('Bubbles technique (Version 3) \n Image: {} {}'.format(category,image_number), fontsize=15)  
                plt.savefig(os.path.join(finalsavedirectory,'{}_{}_full_BUBBLES.png'.format(category,image_number)),dpi=100)
                plt.close()     
            
        
        end=time.time()-start 
        print('... \n =(4.2)BUBBLES type 3==== {:.2f} minutes ===='.format(end/60))    

