# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 12:22:40 2021

P M K processing Keras custom layers

useful info for 
color management: https://fairyonice.github.io/Color-space-defenitions-in-python-RGB-and-LAB.html
keras custom layer: https://stackoverflow.com/questions/42883501/how-to-use-opencv-functions-in-keras-lambda-layer

@author: andresM
"""

import tensorflow as tf
from scipy import ndimage as ndi
import numpy as np
from skimage.color import rgb2lab
from keras.engine.topology import Layer
from keras import backend as K

#===========================================================================
def normalize(image):
    """Takes a tensor of 3 dimensions (height, width, colors) and normalizes it's values
    to be between 0 and 1 so it's suitable for displaying as an image."""
    image = image.astype(np.float32)
    return (image - image.min()) / (image.max() - image.min() + 1e-5)

#%% ==========================================================================   
# functions to perform the DoG relative to the different PMK streams, 
# they need to be used with the color processing functions

def DiffGauss(image,sigma1,sigma2):
    img1=ndi.gaussian_filter(image, np.array([sigma1,sigma1,0]))
    img2=ndi.gaussian_filter(image, np.array([sigma2,sigma2,0]))
    return img1-img2
def Pchannel(img):
    out=DiffGauss(img,sigma1=0.64,sigma2=2.98) #out=DiffGauss(img,sigma1=0.31,sigma2=1.49)
    return out
def Mchannel(img):
    out=DiffGauss(img,sigma1=0.96,sigma2=4.1) #out=DiffGauss(img,sigma1=0.8,sigma2=3.39)
    return out
def Kchannel(img):
    out=DiffGauss(img,sigma1=1.65,sigma2=4.85) #out=DiffGauss(img,sigma1=1.54,sigma2=4.49)
    return out

#%% ==========================================================================

def scale_LAB_to_01(img):
    #original LAB channel ranges: L[0,100]  A[-128,128] B[-128,128]
    # the normalization converts L directly to [0-1], A and B also [0-1], but keeping 0.5 as an origin (instead of 0)
    imgx=np.zeros_like(img)
    imgx[:,:,0]=(img[:,:,0] - img[:,:,0].min()) / (img[:,:,0].max() - img[:,:,0].min()) #L
    imgx[:,:,1]=(img[:,:,1]+128)/255
    imgx[:,:,2]=(img[:,:,2]+128)/255
    return imgx    
    

#   PARVOCELLULAR
def P_image_processing_func(img): 
    # Parvocellular image processing function.
    # input is already [0,1] float32 due to scaling in the Keras input processing
    # Reminder(skimage function): 
    # rgb_lab:[0,1]x[0,1]x[0,1] -> [0,100] x [-128,128] x [-128,128]
    # lab_rgb:[0,100] x [-128,128] x [-128,128] --> [0,1]x[0,1]x[0,1]
    scale=10
    imgx=rgb2lab(img) #convert RGB to LAB (returns a float64 with negative numbers, see range)
    imgx=Pchannel(imgx) #perform the DoG, returns a float64 LAB
    imgx[:,:,2]=imgx[:,:,2]/scale #scale the b:BLUE-YELLOW channel
    imgx=scale_LAB_to_01(imgx) #normalize each channel [0-1] (different methods for L and AB)
    return imgx.astype('float32')

#   MAGNOCELLULAR
def M_image_processing_func(img): 
    # Magnocellular image processing function.
    # input is already [0,1] float32 due to scaling in the Keras input processing
    # Reminder(skimage function): 
    # rgb_lab:[0,1]x[0,1]x[0,1] -> [0,100] x [-128,128] x [-128,128]
    # lab_rgb:[0,100] x [-128,128] x [-128,128] --> [0,1]x[0,1]x[0,1]
    scale=10
    imgx=rgb2lab(img) #convert RGB to LAB (returns a float64 with negative numbers, see range)
    imgx=Mchannel(imgx) #perform the DoG, returns a float64 LAB
    imgx[:,:,1]=imgx[:,:,1]/scale #scale the a:RED-GREEN channel
    imgx[:,:,2]=imgx[:,:,2]/scale #scale the b:BLUE-YELLOW channel
    imgx=scale_LAB_to_01(imgx) #normalize each channel [0-1] (different methods for L and AB)
    return imgx.astype('float32')

#   KONIOCELLULAR
def K_image_processing_func(img): 
    # Koniocellular image processing function.
    # input is already [0,1] float32 due to scaling in the Keras input processing
    # Reminder(skimage function): 
    # rgb_lab:[0,1]x[0,1]x[0,1] -> [0,100] x [-128,128] x [-128,128]
    # lab_rgb:[0,100] x [-128,128] x [-128,128] --> [0,1]x[0,1]x[0,1]
    scale=10
    imgx=rgb2lab(img) #convert RGB to LAB (returns a float64 with negative numbers, see range)
    imgx=Kchannel(imgx) #perform the DoG, returns a float64 LAB
    imgx[:,:,1]=imgx[:,:,1]/scale #scale the a:RED-GREEN channel
    imgx=scale_LAB_to_01(imgx) #normalize each channel [0-1] (different methods for L and AB)
    return imgx.astype('float32')

#%%
def P_image_tensor_func(img4d) :
    results = []
    for img3d in img4d :
        rimg3d = P_image_processing_func(img3d )
        results.append( np.expand_dims( rimg3d, axis=0 ) )
    return np.concatenate( results, axis = 0 )

def M_image_tensor_func(img4d) :
    results = []
    for img3d in img4d :
        rimg3d = M_image_processing_func(img3d )
        results.append( np.expand_dims( rimg3d, axis=0 ) )
    return np.concatenate( results, axis = 0 )

def K_image_tensor_func(img4d) :
    results = []
    for img3d in img4d :
        rimg3d = K_image_processing_func(img3d )
        results.append( np.expand_dims( rimg3d, axis=0 ) )
    return np.concatenate( results, axis = 0 )

#%%

class P_CustomLayer( Layer ) :
    def call( self, xin )  :
        xout = tf.py_func( P_image_tensor_func, 
                           [xin],
                           'float32',
                           stateful=False,
                           name='P_layer')
        xout = K.stop_gradient( xout ) # explicitly set no grad
        xout.set_shape( [xin.shape[0], 128, 128, xin.shape[-1]] ) # explicitly set output shape
        return xout
    def compute_output_shape( self, sin ) :
        return ( sin[0], 128, 128, sin[-1] )


class M_CustomLayer( Layer ) :
    def call( self, xin )  :
        xout = tf.py_func( M_image_tensor_func, 
                           [xin],
                           'float32',
                           stateful=False,
                           name='P_layer')
        xout = K.stop_gradient( xout ) # explicitly set no grad
        xout.set_shape( [xin.shape[0], 128, 128, xin.shape[-1]] ) # explicitly set output shape
        return xout
    def compute_output_shape( self, sin ) :
        return ( sin[0], 128, 128, sin[-1] )


class K_CustomLayer( Layer ) :
    def call( self, xin )  :
        xout = tf.py_func( K_image_tensor_func, 
                           [xin],
                           'float32',
                           stateful=False,
                           name='P_layer')
        xout = K.stop_gradient( xout ) # explicitly set no grad
        xout.set_shape( [xin.shape[0], 128, 128, xin.shape[-1]] ) # explicitly set output shape
        return xout
    def compute_output_shape( self, sin ) :
        return ( sin[0], 128, 128, sin[-1] )





def test_image_processing_func(img): 
    #scale=10
    imgx=rgb2lab(img) #convert RGB to LAB (returns a float64 with negative numbers, see range)
    imgx=scale_LAB_to_01(imgx) #normalize each channel [0-1] (different methods for L and AB)
    return imgx.astype('float32')

def test_image_tensor_func(img4d) :
    results = []
    for img3d in img4d :
        rimg3d = test_image_processing_func(img3d )
        results.append( np.expand_dims( rimg3d, axis=0 ) )
    return np.concatenate( results, axis = 0 )

class test_CustomLayer( Layer ) :
    def call( self, xin )  :
        xout = tf.py_func( test_image_tensor_func, 
                           [xin],
                           'float32',
                           stateful=False,
                           name='test_custom_layer')
        xout = K.stop_gradient( xout ) # explicitly set no grad
        xout.set_shape( [xin.shape[0], 128, 128, xin.shape[-1]] ) # explicitly set output shape
        return xout
   # def compute_output_shape( self, sin ) :
   #     return ( sin[0], 128, 128, sin[-1] )



