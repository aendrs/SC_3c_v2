# -*- coding: utf-8 -*-
"""
Main Model for the paper:
A Deep Neural Network model of the Superior Colliculus in emotion recognition

September 2021
@author: Andrés Méndez
"""


#%%  Import libraries

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import time

from keras.models import Model
from keras.layers import (
        Input, Lambda, Dense, Conv2D, BatchNormalization, Activation, 
        concatenate, MaxPooling2D, AveragePooling2D, Dropout, GlobalAveragePooling2D, SpatialDropout2D)
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adagrad, Adam
from keras.callbacks import ReduceLROnPlateau, CSVLogger


#================= For usage with the RTX2070super GPU ==================
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
#config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras
#========================================================================

from PMK_customlayer import P_CustomLayer, M_CustomLayer, K_CustomLayer

from fractalnet import fractal_net
import attention_mechanism

from helperfuncs.tictoc import *
from helperfuncs.savemodelandhistory import *
from helperfuncs.plot_model_history import *
from helperfuncs.save_weights_history import *


#%% Declaring variables ====================================================

train_dir='/home/andres/Documents/colliculus/datasets/dataset_V6_3class/train'
test_dir='/home/andres/Documents/colliculus/datasets/dataset_V6_3class/test'

trainsetsize=20689 
testsetsize=750


classes=3
img_height=128
img_width=128
PMK_dropout_rate=0.25
SpatialDropoutRate_Attention=0.25
SpatialDropoutRate_interfractal=0.3

batch_size=48
epochs=100



#%% PMK functions and connections ==========================================

# PMK connections to layers (expressed in number of featuremaps)
# suffixes: u= upperSGS, l=lowerSGS, o=StratumOpticum


#P10
Pu=6
Pl=4
Po=0

#M48
Mu=15
Ml=30
Mo=3

#K33
Ku=20
Kl=10
Ko=3

#===========================
kernel_PMK=(3,3)
strides_PMK=1


# declare input
input_img=Input(shape=(img_height,img_width,3),name='input')

P_input=P_CustomLayer(name='P_custom')(input_img) 
P_input=BatchNormalization()(P_input)

M_input=M_CustomLayer(name='M_custom')(input_img) 
M_input=BatchNormalization()(M_input)

K_input=K_CustomLayer(name='K_custom')(input_img) 
K_input=BatchNormalization()(K_input)

# derive the PMK connections which will be projected to the SC layers
blockP=Conv2D(Pu+Pl+Po, kernel_PMK, strides=strides_PMK, padding='same', name='P_Conv')(P_input)
blockP=Activation('relu')(blockP)
blockP=BatchNormalization()(blockP)
blockP=SpatialDropout2D(rate=PMK_dropout_rate)(blockP)

blockM=Conv2D(Mu+Ml+Mo, kernel_PMK, strides=strides_PMK, padding='same', name='M_Conv')(M_input)
blockM=Activation('relu')(blockM)
blockM=BatchNormalization()(blockM)
blockM=SpatialDropout2D(rate=PMK_dropout_rate)(blockM)

blockK=Conv2D(Ku+Kl+Ko, kernel_PMK, strides=strides_PMK, padding='same', name='K_Conv1')(K_input)
blockK=Activation('relu')(blockK)
blockK=BatchNormalization()(blockK)
blockK=SpatialDropout2D(rate=PMK_dropout_rate)(blockK)

Pu_out = Lambda(lambda x : x[:,:,:,0:Pu])(blockP)
Pl_out = Lambda(lambda x : x[:,:,:,Pu:(Pu+Pl)])(blockP)
#Po_out = Lambda(lambda x : x[:,:,:,(Pu+Pl):])(blockP)  #since Po=0 we dont need this

Mu_out = Lambda(lambda x : x[:,:,:,0:Mu])(blockM)
Ml_out = Lambda(lambda x : x[:,:,:,Mu:(Mu+Ml)])(blockM)
Mo_out = Lambda(lambda x : x[:,:,:,(Mu+Ml):])(blockM)

Ku_out = Lambda(lambda x : x[:,:,:,0:Ku])(blockK)
Kl_out = Lambda(lambda x : x[:,:,:,Ku:(Ku+Kl)])(blockK)
Ko_out = Lambda(lambda x : x[:,:,:,(Ku+Kl):])(blockK)



#%% uSGS ===================================================================

uSGS_input=concatenate([Pu_out,Mu_out,Ku_out], axis=-1)
uSGS_input=BatchNormalization()(uSGS_input)



conv_uSGS_01=[(41,3,3)] #in: 41 out:41    
drop_path_uSGS=0.15
uSGS_01_out=fractal_net(c=2,
                   b=1,
                   conv=conv_uSGS_01,
                   drop_path=drop_path_uSGS,
                   deepest=False,
                   prefix_str='uSGS_01',
                   finalMaxPool=True)(uSGS_input)
uSGS_01_out=BatchNormalization()(uSGS_01_out)
Att_uSGS_01=attention_mechanism.AttentionModule(Kheads=5,kernel=(5,5),classes=classes,spatialdropoutrate=SpatialDropoutRate_Attention)(uSGS_01_out)
uSGS_01_out=SpatialDropout2D(rate=SpatialDropoutRate_interfractal)(uSGS_01_out)




#%% lSGS ===================================================================


lSGS_PMK=concatenate([Pl_out,Ml_out,Kl_out], axis=-1)
lSGS_PMK=AveragePooling2D(pool_size=(2,2))(lSGS_PMK) #for spatially downsampling input to lSGS

lSGS_input=concatenate([lSGS_PMK, uSGS_01_out], axis=-1)
lSGS_input=BatchNormalization()(lSGS_input)



conv_lSGS_01=[(85,3,3)] #in:85(41+44) out:85
drop_path_lSGS=0.15
lSGS_01_out=fractal_net(c=2,
                   b=1,
                   conv=conv_lSGS_01,
                   drop_path=drop_path_lSGS,
                   deepest=False,
                   prefix_str='lSGS_01',
                   finalMaxPool=True)(lSGS_input)
lSGS_01_out=BatchNormalization()(lSGS_01_out)
Att_lSGS_01=attention_mechanism.AttentionModule(Kheads=5,kernel=(3,3),classes=classes,spatialdropoutrate=SpatialDropoutRate_Attention)(lSGS_01_out)
lSGS_01_out=SpatialDropout2D(rate=SpatialDropoutRate_interfractal)(lSGS_01_out)




#%% SO =====================================================================


SO_PMK=concatenate([Mo_out,Ko_out], axis=-1) 
SO_PMK=AveragePooling2D(pool_size=(4,4))(SO_PMK) #for spatially downsampling input to lSGS

SO_input=concatenate([SO_PMK, lSGS_01_out], axis=-1)
SO_input=BatchNormalization()(SO_input)

conv_SO=[(91,3,3)]  #in:91 (85+6) out:91
drop_path_SO=0.15
SO_out=fractal_net(c=2,
                   b=1,
                   conv=conv_SO,
                   drop_path=drop_path_SO,
                   deepest=False,
                   prefix_str='SO',
                   finalMaxPool=False)(SO_input) #NOT add final Max Pool inside the Fractal Block, add it adfter the Attention Module

SO_out=BatchNormalization()(SO_out)
Att_SO=attention_mechanism.AttentionModule(Kheads=5,kernel=(1,1),classes=classes,spatialdropoutrate=SpatialDropoutRate_Attention)(SO_out)
SO_out=SpatialDropout2D(rate=SpatialDropoutRate_interfractal)(SO_out)

SO_out=MaxPooling2D(pool_size=(2,2), strides=(2,2))(SO_out) 


#%% GlobalFeatures =========================================================
GlobalFeatures=Conv2D(classes,(3,3), padding='same', name='GlobalFeatureMaps')(SO_out)
GlobalFeatures=Activation('relu')(GlobalFeatures)
GlobalFeatures=BatchNormalization()(GlobalFeatures)

GlobalFeatures=GlobalAveragePooling2D()(GlobalFeatures) #inputdim=[b,h,w,classes] outdim=[b,classes]
O_network = Dense(classes, activation='softmax', name='predictions')(GlobalFeatures)

#%% 
O_attmodules=concatenate([Att_uSGS_01, Att_lSGS_01,  Att_SO], axis=1)
out=attention_mechanism.global_gates(AttModules=3 , O_attmodules=O_attmodules, O_network=O_network, gatecase=3, classes=classes, kernel=(3,3), O_network_weight=0.5)(SO_out)


#%%

#create model
model = Model(input_img, out, name='ColliculusFractalAttention')
model.summary()


#%%
#compile model 

optimizer = Adam()

csv_logger=CSVLogger(filename='log.csv', separator=',', append=False)


model.compile(optimizer=optimizer, 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

class_weights={0:1.,1:2.,2:2.}



#%%  Image generators

from PIL import ImageFile  
ImageFile.LOAD_TRUNCATED_IMAGES = True  #lines to avoid truncated file errors
 
train_datagen=ImageDataGenerator( rescale=1./255,
				       shear_range = 0.3,
                                       zoom_range = 0.3,
                                       rotation_range=10,
                                       fill_mode='wrap',
                                       cval=0,
                                       horizontal_flip = True,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2)


test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(train_dir,
                                       target_size = (img_height,img_width),
                                       class_mode = 'categorical',
                                       batch_size = batch_size,
                                       shuffle=True, 
                                       seed=7,
                                       color_mode='rgb')

test_set = test_datagen.flow_from_directory(test_dir,
                                       target_size = (img_height,img_width),
                                       class_mode = 'categorical',
                                       batch_size = batch_size,
                                       shuffle=True, 
                                       seed=7,
                                       color_mode='rgb')


#%%

def main():
    tic = time.time()
    history=model.fit_generator(training_set,
                        steps_per_epoch=trainsetsize//batch_size,
                        epochs = epochs,
                        validation_data = test_set,
                        validation_steps = testsetsize//batch_size,
                        verbose=1,
                        callbacks=[csv_logger],
                        use_multiprocessing=False,
                        workers=1,
                        class_weight=class_weights) 
    toc = time.time()-tic
    print('elapsed time: {} hr/min/sec'.format (printelapsedtime(toc)))
    # SAVE trainingtime, History and Model (HD5file+weights) in a Year-Month-Day folder
    directorypath=save_weights_history(history, model, scriptname= os.path.basename(__file__),timeelapsed=toc)  
    model.save(directorypath +'/modelfull.h5')



#%%

""" Main """
if __name__ == "__main__":
    main()

