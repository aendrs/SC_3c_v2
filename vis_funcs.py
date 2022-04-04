# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 15:32:00 2019

@author: andre
"""
import skimage.io
from skimage.transform import resize
import numpy as np
import math
import matplotlib.pyplot as plt
from keras import backend as K




def load_imagePMK(image_number, category, 
                   image_size=(128,128),
                   dataset='dataset_V5_clahe_PMK_gauss', basepath='C:\\Users\\andre\\Documents\\Torino\\colliculus', train_or_test='test' ):
    imageP = skimage.io.imread(basepath + '\\' + dataset + '\\' + 'P' + '\\' + train_or_test + '\\' +  category + '\\' + train_or_test + '_' + category + '_' + image_number + '.jpg')
    imageM = skimage.io.imread(basepath + '\\' + dataset + '\\' + 'M' + '\\' + train_or_test + '\\' +  category + '\\' + train_or_test + '_' + category + '_' + image_number + '.jpg')
    imageK = skimage.io.imread(basepath + '\\' + dataset + '\\' + 'K' + '\\' + train_or_test + '\\' +  category + '\\' + train_or_test + '_' + category + '_' + image_number + '.jpg')
    imageP=resize(imageP,image_size,preserve_range=True, anti_aliasing=True)
    imageM=resize(imageM,image_size,preserve_range=True, anti_aliasing=True)
    imageK=resize(imageK,image_size,preserve_range=True, anti_aliasing=True)
    return [imageP,imageM,imageK]



def imagePMK_preprocessing(imagePMK):   
    post_imagePMK=[]
    for image in imagePMK:
        x = image.astype(np.float32) # Convert image to float
        x = x[np.newaxis,...] # Make it a batch of one. The model expects a batch, not a single image
        print(x.shape)
        x=x/255. #remember tha the input tensor needs to be [0,1]
        post_imagePMK.extend(x)
    return post_imagePMK    
    



'''
def heatmap_imagePMK(imagePMK, rowscols=10, image_size=128):
    """
    input:  list of 3 numpy arrays (h,w,3)
    """
    step = np.ceil(image_size/rowscols)
    step=step.astype(np.uint8)
    heatmap=[]
    for i, image in enumerate(imagePMK):
        temp=[]
        print('i= ' +str(i))
        print(image.shape)
        print(type(image))
        for row in range(0, image.shape[0], step):
            for col in range(0, image.shape[1], step):
                new_image = image.copy()
                # Add a square patch. Using a bright color here to make it easier to see.
                new_image[row:row+step, col:col+step, :] = [255,255,255]
                temp.append(new_image)
                print(len(temp))
        heatmap.append( np.stack(temp))
    return heatmap        
'''


#================================================================
# these functions were taken from 
# https://github.com/waleedka/cnn-visualization/blob/master/cnn_visualization.ipynb

    
def tensor_summary(tensor):
    """Display shape, min, and max values of a tensor."""
    print("shape: {}  min: {}  max: {}  type: {}".format(tensor.shape, tensor.min(), tensor.max(), tensor.dtype))

    
def normalize(image):
    """Takes a tensor of 3 dimensions (height, width, colors) and normalizes it's values
    to be between 0 and 1 so it's suitable for displaying as an image."""
    image = image.astype(np.float32)
    return (image - image.min()) / (image.max() - image.min() + 1e-5)

def scale(image):
    'just scale the matrix by its larger value, does not normalize [0-1]'
    return(image/np.max(image))

def display_images(images, titles=None, cols=5, interpolation=None, cmap="Greys_r"):
    """
    images: A list of images. I can be either:
        - A list of Numpy arrays. Each array represents an image.
        - A list of lists of Numpy arrays. In this case, the images in
          the inner lists are concatentated to make one image.
    """
    titles = titles or [""] * len(images)
    rows = math.ceil(len(images) / cols)
    height_ratio = 1.2 * (rows/cols) * (0.5 if type(images[0]) is not np.ndarray else 1)
    plt.figure(figsize=(15, 15 * height_ratio))
    i = 1
    for image, title in zip(images, titles):
        plt.subplot(rows, cols, i)
        plt.axis("off")
        # Is image a list? If so, merge them into one image.
        if type(image) is not np.ndarray:
            image = [normalize(g) for g in image]
            image = np.concatenate(image, axis=1)
        else:
            image = normalize(image)
        plt.title(title, fontsize=9)
        plt.imshow(image, cmap=cmap, interpolation=interpolation)
        i += 1

#==================================================================

           
        
            
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])



def occlusion_tensor(image, rowscols=10, image_size=(128,128), squarecolor=[0,0,0]):  
    '''
    Creates a square which is overlaid and slid over the image, the output is a stack of images
    image input shape (1,h,w,3)
    output shape (rowscols**2,h,w,3)
    '''
    image=np.squeeze(image) #we squeeze the first empty dimension in order to use np.stack at the end
    if len(image.shape)>3:
        raise Exception ('image should have 3 dimensions after np.squeeze, check function and inputs')    
    step = np.ceil(image_size[0]/rowscols) #size of occlusion square
    step=step.astype(np.uint8)
    out_image = []
    for row in range(0, image.shape[0], step):
        for col in range(0, image.shape[1], step):
            new_image = image.copy()
            # Add a square patch. Using a bright color here to make it easier to see.
            new_image[row:row+step, col:col+step, :] = squarecolor
            out_image.append(new_image)
    out_image = np.stack(out_image)
    out_image.shape
    return out_image




def twoD_Gaussian(x,y, xo, yo, sigma_x, sigma_y):
    a = 1./(2*sigma_x**2) + 1./(2*sigma_y**2)
    c = 1./(2*sigma_x**2) + 1./(2*sigma_y**2)
    g = np.exp( - (a*((x-xo)**2) + c*((y-yo)**2)))
    return g.ravel()




def bubbles_mask_tensor(image,rowscols=5, image_size=(128,128), gaussian=False, sigma=10., biasvalue=0.5): 
    image=np.squeeze(image) #we squeeze the first empty dimension in order to use np.stack at the end
    if len(image.shape)>3:
        raise Exception ('image should have 3 dimensions after np.squeeze, check function and inputs')  
    step = np.ceil(image_size[0]/rowscols) #size of occlusion square to slide over the image
    step=step.astype(np.uint8)
    halfstep=step//2
    out_image = []
    if gaussian:
        for row in range(0, image_size[0], step):
            for col in range(0, image_size[1], step):
                y, x = np.mgrid[0:image_size[0], 0:image_size[1]]
                gaussmask = twoD_Gaussian(x, y, col+halfstep, row+halfstep, sigma, sigma)
                gaussmask=gaussmask.reshape(image_size[0], image_size[1])
                gaussmask = np.repeat(gaussmask[...,None],3,axis=-1)        
                maskcomplement=np.abs(1-gaussmask) #to offset the zeros around the gaussian
                new_image = image.copy()
                new_image=new_image*gaussmask
                new_image=new_image+(maskcomplement*biasvalue) #in order o correct the surrounding zero value around the 2D gaussian and take it to a baseline value, normally biasvalue is 0.5
                out_image.append(new_image)
    else:
        for row in range(0, image_size[0], step):
            for col in range(0, image_size[1], step):
                new_image = image.copy()
                mask = np.zeros((image_size[0],image_size[1],3),np.float32)
                biasmask=np.ones((image_size[0],image_size[1],3),np.float32)                 
                mask[row:row+step, col:col+step, :] = 1.
                biasmask[row:row+step, col:col+step, :] = 0.
                new_image=new_image*mask
                biasmask=biasmask*0.5 
                new_image=new_image+biasmask                  
                out_image.append(new_image)
    out_image = np.stack(out_image)
    #print(out_image.shape)
    return out_image


def bubbles_pyramid(model,imagePMK,baseimage, correct_category_label, predictions,
                    category, image_number, rowscols_list=[4,5,6], sigma_list=[16,15,13], biasvalue=0.5, order=1, cmap='jet'):
    total_map=[]
    for rowscols,sigma in zip(rowscols_list,sigma_list):
        #print(rowscols,sigma)
        imagePMK_bubbles=[bubbles_mask_tensor(image,rowscols=rowscols, gaussian=True, sigma=sigma , biasvalue=biasvalue) for image in imagePMK]
        bubbles_y = model.predict(imagePMK_bubbles) 
        probs = bubbles_y[:, correct_category_label] #choose the CORRECT label index, not the max prob, based on the category name of the image
        probs_matrix = np.reshape(probs, (rowscols, rowscols)) #reshape probs vector into a mask
        probs_matrix_big = skimage.transform.resize(probs_matrix, (128,128), order=order, mode='edge')
        total_map.append(probs_matrix_big)
    total_map=np.stack(total_map)
    total_map=np.sum(total_map, axis=0)
    total_map=normalize(total_map)
    #--plot--
    fig=plt.figure(figsize=(6.5,5))    
    plt.axis('off')
    im1=plt.imshow(normalize(baseimage))   
    im2=plt.imshow(total_map,alpha=.8,cmap=cmap, vmin=np.min(total_map))   
    plt.title('Bubbles technique: pyramidal stacking \n Image: {} {}'.format(category,image_number),fontsize=12)
    cbar=plt.colorbar(im2, orientation ='vertical')
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('Saliency', rotation=270,fontsize=10)
    #plt.savefig(finalsavedirectory+'\\{}_{}_occlusion_VARIATION_ord0.png'.format(category,image_number),dpi=100)    

    
  
    
 


def occlusion_variationfrombaseline(model,imagePMK,baseimage, label_index, 
                                    predictions,category,image_number,rowscols_list=[5,7,9,10],order=1):
    classprob=predictions[0,label_index]
    probs_total=[]
    probs_var=[]
    for rowscols in rowscols_list:
        imagePMK_hm=[occlusion_tensor(image,rowscols=rowscols,squarecolor=[0.,0.,0.]) for image in imagePMK]
        heatmap_y = model.predict(imagePMK_hm)
        probs = heatmap_y[:, label_index] #choose the label index obtained with the classification of the complete image
        probs_matrix = np.reshape(probs, (rowscols, rowscols)) #reshape the probs vector (original values, no normalization like before)
        probs_matrix_big = skimage.transform.resize(probs_matrix, (128,128), order=order, mode='edge')
        probs_total.append(probs_matrix_big)
        probs_var_temp=probs_matrix_big-classprob
        probs_var.append(probs_var_temp)
    probs_var_total=np.stack(probs_var)
    print(probs_var_total.shape)
    probs_var_total=np.sum(probs_var_total, axis=0)   
    probs_var_total=probs_var_total/len(rowscols_list) 
    #--plot--
    fig=plt.figure(figsize=(6.5,5))    
    plt.axis('off')
    im1=plt.imshow(normalize(baseimage))   
    im2=plt.imshow(probs_var_total,alpha=.8,cmap='seismic_r',vmin=np.min(probs_var_total), vmax=-1*np.min(probs_var_total))   
    plt.title('Variation from baseline when occluded \n Image: {} {}'.format(category,image_number),fontsize=12)
    cbar=plt.colorbar(im2, orientation ='horizontal')#ticks=[0,-.1,-.2,-.3,-.4,-.5,-.6,.1,.2,0.3,0.4,0.5,0.6]
    cbar.ax.tick_params(labelsize=20)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('Probability variation from baseline', rotation=270,fontsize=10)
    #plt.savefig(finalsavedirectory+'\\{}_{}_occlusion_VARIATION_ord0.png'.format(category,image_number),dpi=100)    










def apply_mask_heatmap(image, mask, heatmap, order=1, cmap="Greys_r", title1='Salience Overlay', title2='Classification Probability when occluded'):
    if len(image.shape)==3:
        mask = skimage.transform.resize(normalize(mask), (image.shape[0],image.shape[1]), order=order)[:,:,np.newaxis].copy()
    else:
        mask = skimage.transform.resize(normalize(mask), (image.shape[0],image.shape[1]), order=order).copy()    
    imagemask = image * mask # Apply mask to image
    fig, (ax0,ax1) = plt.subplots(1,2, figsize=(10, 5.5))
    ax0.axis("off")
    imagemask = normalize(imagemask)
    ax0.set_title(title1, fontsize=10)
    ax0.imshow(imagemask, cmap=cmap, interpolation=None)
    #=====
    ax1.axis("off")
    im1=ax1.imshow(normalize(heatmap), cmap=cmap)
    #plt.colorbar(im1,ticks=[0.1,0.3,0.5,0.7,0.9], orientation ='vertical')
    ax1.set_title(title2, fontsize=10)
    heatmap = np.around(heatmap, decimals=2)
    for i in range(len(heatmap)):
        for j in range(len(heatmap)):  
            valuenormalized=(heatmap[i, j]-heatmap.min())/(heatmap.max()-heatmap.min())
            if cmap=='OrRd':
                color="w" if valuenormalized > 0.5  else "k"
            elif cmap=='OrRd_r':
                color="w" if valuenormalized < 0.5  else "k"   
            else:
                color="w" if heatmap[i, j]>0.5  else "k"
            text = ax1.text(j, i, heatmap[i, j],
                           ha="center", va="center",color=color , fontsize=8)
            



def apply_mask_heatmap_jetcolormap(image, mask, heatmap, order=1, cmap1='jet_r', cmap2="Greys_r", title1='Salience Overlay', title2='Classification Probability when occluded'):
    if len(image.shape)==3:
        mask = skimage.transform.resize(normalize(mask), (image.shape[0],image.shape[1]), order=order)[:,:,np.newaxis].copy()
    else:
        mask = skimage.transform.resize(normalize(mask), (image.shape[0],image.shape[1]), order=order).copy()    
    imagemask = image * mask # Apply mask to image
#    fig, (ax0,ax1) = plt.subplots(1,2, figsize=(10, 5.5))
#    ax0.axis("off")
#    imagemask = normalize(imagemask)
#    ax0.set_title(title1, fontsize=10)
#    ax0.imshow(imagemask, cmap=cmap, interpolation=None)
#    #=====
#    ax1.axis("off")
#    im1=ax1.imshow(normalize(heatmap), cmap=cmap)
#    #plt.colorbar(im1,ticks=[0.1,0.3,0.5,0.7,0.9], orientation ='vertical')
#    ax1.set_title(title2, fontsize=10)
#    heatmap = np.around(heatmap, decimals=2)
#    for i in range(len(heatmap)):
#        for j in range(len(heatmap)):  
#            valuenormalized=(heatmap[i, j]-heatmap.min())/(heatmap.max()-heatmap.min())
#            if cmap=='OrRd':
#                color="w" if valuenormalized > 0.5  else "k"
#            elif cmap=='OrRd_r':
#                color="w" if valuenormalized < 0.5  else "k"   
#            else:
#                color="w" if heatmap[i, j]>0.5  else "k"
#            text = ax1.text(j, i, heatmap[i, j],
#                           ha="center", va="center",color=color , fontsize=8)
            
    fig, (ax0,ax1,ax2) = plt.subplots(1,3, figsize=(16, 7))       
    ax0.axis('off')
    ax0.imshow(image,cmap='gray')   
    #axes[0].set_title('AKDEF image. Gender={}, Emotion={}'.format(gender,category),fontsize=12)
    ax1.axis("off")
    ax1.imshow(image,cmap='gray')  
    ax1.imshow(mask,alpha=0.55,cmap=cmap1)   
    ax1.set_title(title1, fontsize=10)
    #axes[1].set_title('Average scaled_correct/scaled_total Plane ',fontsize=12)
    #plt.suptitle('Average Bubbles over KDEF dataset. \n Average scaled_correct/scaled_total Plane \n b_sigma={}, iter={}'.format(sigma, bubbles_iterations),fontsize=14)
    #===
    
    ax2.axis("off")
    ax2.imshow(normalize(heatmap), cmap=cmap2)
    #plt.colorbar(im1,ticks=[0.1,0.3,0.5,0.7,0.9], orientation ='vertical')
    ax2.set_title(title2, fontsize=10)
    heatmap = np.around(heatmap, decimals=2)
    for i in range(len(heatmap)):
        for j in range(len(heatmap)):  
            valuenormalized=(heatmap[i, j]-heatmap.min())/(heatmap.max()-heatmap.min())
            if cmap2=='OrRd':
                color="w" if valuenormalized > 0.5  else "k"
            elif cmap2=='OrRd_r':
                color="w" if valuenormalized < 0.5  else "k"   
            else:
                color="w" if heatmap[i, j]>0.5  else "k"
            text = ax2.text(j, i, heatmap[i, j],
                           ha="center", va="center",color=color , fontsize=8)
    fig.tight_layout()









def apply_mask(image, mask, order=1, avoid_black=True,  imagemultiplier=0.1, 
               include_side_image=True, cmap="jet", title='Salience Overlay'):
    mask = skimage.transform.resize(normalize(mask), image.shape[:2], order=order)[:,:,np.newaxis].copy()
    if avoid_black==True:
        imagemask=image*imagemultiplier + (image*mask) 
    else:
        imagemask = image * mask # Apply mask to image
    imagemask = normalize(imagemask)
    imagemaskgray=rgb2gray(imagemask)
    if include_side_image:
        fig, (ax0,ax1) = plt.subplots(1,2, figsize=(10, 5))
        ax0.axis("off")
        ax0.set_title(title, fontsize=10)
        ax0.imshow(imagemask)
        ax1.axis("off")
        ax1.imshow(imagemaskgray, cmap=cmap)
        ax1.set_title(title, fontsize=10)
    else:    
        fig, ax0 = plt.subplots(1, figsize=(7, 7))
        plt.axis("off")
        imagemask = normalize(imagemask)
        tensor_summary(imagemask)
        ax0.set_title(title, fontsize=10)
        ax0.imshow(imagemask)

            










def predict5emotions(model, imagePMK):
    predictions = model.predict(imagePMK)
    print('''predictions: \n 
          afraid {:.3f} \n 
          anger {:.3f} \n 
          happy {:.3f} \n
          neutral {:.3f} \n
          sad {:.3f}'''.format(predictions[0][0],predictions[0][1], 
          predictions[0][2], predictions[0][3], predictions[0][4]))
    return predictions



def predict3emotions(model, imagePMK):
    predictions = model.predict(imagePMK)
    print('''predictions: \n 
          happy {:.3f} \n
          neutral {:.3f} \n
          sad {:.3f}'''.format(predictions[0][0],predictions[0][1], 
          predictions[0][2]))
    return predictions


#==========================================================================
# functions for visualizing the layer activations      

def read_layer(model, x, layer_name):
    """Return the activation values for the specifid layer
       Made for a MULTIPLE Input model (like a PMK list of processed images)    
    """
    # Create Keras function to read the output of a specific layer
    get_layer_output = K.function(model.input, [model.get_layer(layer_name).output])
    outputs = get_layer_output(x)[0]
    #tensor_summary(outputs)
    return outputs[0]
    
def view_layer(model, x, layer_name, cols=5):
    outputs = read_layer(model, x, layer_name)
    display_images([outputs[:,:,i] for i in range(outputs.shape[2])], cols=cols)

def read_layer_singleinputmodel(model, x, layer_name):
    """Return the activation values for the specifid layer
       Made for a Single Input model  
    """
    # Create Keras function to read the output of a specific layer
    get_layer_output = K.function([model.input], [model.get_layer(layer_name).output])
    outputs = get_layer_output([x])[0]
    #tensor_summary(outputs)
    return outputs[0]


#=========================================================================
# functions for the proper bubbles iterative technique, check Schyns paper
#   B  U  B  B  L  E  S



def create_bubbles_mask(bubbles=10, sigma=4., image_size=(128,128), offsetpx=0): 
    mask=np.zeros(image_size,np.float32)    
    for _ in range(bubbles):
        #x0=np.random.randint(0-offsetpx,image_size[1]+1+offsetpx)
        #y0=np.random.randint(0-offsetpx,image_size[0]+1+offsetpx)
        x0=np.random.random_sample()*(image_size[1]+2*offsetpx)-offsetpx
        y0=np.random.random_sample()*(image_size[0]+2*offsetpx)-offsetpx
        y, x = np.mgrid[0:image_size[0], 0:image_size[1]]
        protomask=twoD_Gaussian(x, y, x0, y0, sigma, sigma)
        protomask=protomask.reshape(image_size[0], image_size[1])
        mask=mask+protomask  
    mask[mask>1.]=1  #cap the values to 1, not scale to 0-1!!! because that would potentially affect the values of the non overlapping blobs  
    #plt.imshow(mask)  
    return mask.astype('float32') #returns a 2D mask


def apply_bubbles_mask(imagelist, mask, usecomplementmask=True, biasvalue=0.5):
    '''
    imagelist: list of 3 images (1, h, w, color)
    '''
    mask=np.repeat(mask[:,:,np.newaxis],3,-1) # repeat mask along the 3rd axis in prder to multiply image
    imagePMK_bubbles=[ mask*image.squeeze() for image in imagelist] #squeeze the empty batch dimension of the input image and multiply with the mask
    if usecomplementmask: # AFTER applying the mask to the image
        #maskcomplement=np.abs(1-mask) #to offset the zeros around the gaussian
        #imagePMK_bubbles=[ image+(maskcomplement*biasvalue) for image in imagePMK_bubbles]    
        imagePMK_bubbles=[ (image*0.5+biasvalue) for image in imagePMK_bubbles] #scale [0,1]to [0.5,1]
    return imagePMK_bubbles
    


def bubbles_technique(iterations, model, imagelist, bubbles, sigma, img_label_idx, acc_baseline, acc_threshold=0.75, 
                      image_size=(128,128),usecomplementmask=True, biasvalue=0.48):
    '''
    imagelist: list of 3 3D image matrices, PMK inputs
    model: trained Keras model (only valid for the 3 input model architecture)
    biasvalue: in order o correct the surrounding zero value around the 2D gaussian and take it to
                a baseline value, normally biasvalue is around 0.5, corresponding to medium grey
    bubbles:  number of gaussian bubbles
    sigma:    std of the gaussian bubbles   
    '''
    correct_plane=[]
    total_plane=[]    
    bubbles_vec=[]
    acc_vec=[] 
    if acc_baseline<0.5: #if the baseline is too low, then raise the threshold to 0.5
        acc_threshold=0.5/acc_baseline
    for _ in range(iterations):
        mask=create_bubbles_mask(bubbles, sigma, image_size)
        img_bubbles=apply_bubbles_mask(imagelist, mask, usecomplementmask, biasvalue)
        img_bubbles=[image[np.newaxis,:,:,:] for image in img_bubbles] #add batch dimension needed for classification  
        bubbles_acc = model.predict(img_bubbles) #prediction of the new bubbles image
        bubbles_vec.append(bubbles) #add the actual number of bubbles to the vector for analysis
        acc_vec.append(bubbles_acc) # append the whole accuracy vector, not only the relevant class
        if bubbles_acc[0,img_label_idx]>acc_baseline*acc_threshold: #if acc>threshold then take one bubble out, if not add one
            bubbles -=3
            correct_plane.append(mask)
            total_plane.append(mask)
        else:
            bubbles +=1
            total_plane.append(mask) #only to the total, not to the correct      
    return correct_plane, total_plane, bubbles_vec, acc_vec, acc_threshold    
        
        

#==========================
# update 23/09/2019
def bubbles_technique_2(iterations, model, imagelist, bubbles, sigma, img_label_idx, acc_baseline, acc_threshold=0.75, 
                      image_size=(128,128),usecomplementmask=True, biasvalue=0.48):
    '''
    imagelist: list of 3 3D image matrices, PMK inputs
    model: trained Keras model (only valid for the 3 input model architecture)
    biasvalue: in order o correct the surrounding zero value around the 2D gaussian and take it to
                a baseline value, normally biasvalue is around 0.5, corresponding to medium grey
    bubbles:  number of gaussian bubbles
    sigma:    std of the gaussian bubbles   
    '''
    correct_plane=[]
    total_plane=[]    
    bubbles_vec=[]
    acc_vec=[] 
    temp_acc_vec=[]
    real_accuracy_threshold=acc_baseline*acc_threshold
    print('real_accuracy_threshold:{}'.format(real_accuracy_threshold))
    if acc_baseline<0.5: #if the baseline is too low, then raise the threshold to 0.5
        real_accuracy_threshold=0.5
        acc_threshold=0.5/acc_baseline #relative accuracy threshold that will be passed for plotting
    for i in range(iterations):
        
        if i%10==0 and i>0: #every ten iterations review results and decide if a new bubble is needed
            acc_round=[]
            for acc in temp_acc_vec: #get the relevant accuracy vector
                acc_round.append(acc[0,img_label_idx])                
            temp_acc_vec=[] #reset the temporary accuracy vector
            
            if np.mean(acc_round)<real_accuracy_threshold and bubbles<=50: #if MeanAcc<RealThreshold  add one bubble
                #print('i:{}.B:{}. mean_acc: {:.2f}'.format(i,bubbles, np.mean(acc_round)))
                #print(['{:.2f}'.format(x) for x in acc_round])
                bubbles +=1 
            else:
                bubbles-=1
                     
            if bubbles<10:
                bubbles=20
                
        mask=create_bubbles_mask(bubbles, sigma, image_size)
        img_bubbles=apply_bubbles_mask(imagelist, mask, usecomplementmask, biasvalue)
        img_bubbles=[image[np.newaxis,:,:,:] for image in img_bubbles] #add batch dimension needed for classification  
        bubbles_acc = model.predict(img_bubbles) #prediction of the new bubbles image
        bubbles_vec.append(bubbles) #add the actual number of bubbles to the vector for analysis
        acc_vec.append(bubbles_acc) # append the whole accuracy vector, not only the relevant class
        temp_acc_vec.append(bubbles_acc) #append the accuracy vec to a temporary vector that will be used each round of 10
        
        if bubbles_acc[0,img_label_idx]>real_accuracy_threshold: #if acc>threshold 
            correct_plane.append(mask)
            total_plane.append(mask)
        else:
            total_plane.append(mask) #only to the total, not to the correct                  
    return correct_plane, total_plane, bubbles_vec, acc_vec, acc_threshold    
        

#!!!
#=========================
# Nov 04 2020

def bubbles_technique_3(iterations, model, imagelist, bubbles, sigma, img_label_idx,
                      acc_baseline,image_size=(128,128),usecomplementmask=True, 
                      biasvalue=0.5, maxbubbles=50, offsetpx=0):
    '''
    imagelist: list of 3 3D image matrices, PMK inputs
    model: trained Keras model (only valid for the 3 input model architecture)
    biasvalue: in order o correct the surrounding zero value around the 2D gaussian and take it to
                a baseline value, normally biasvalue is around 0.5, corresponding to medium grey
    bubbles:  number of gaussian bubbles
    sigma:    std of the gaussian bubbles   
    img_label_idx: real label index of the image (i.e. the real clas it belongs to)
    maxbubbles: number of maximum bubbles allowed
    offsetpx=offset pixels along height and lenght for the generation of the masks, avoids dark borders
    '''
    correct_plane=[]
    total_plane=[]    
    bubbles_vec=[]
    acc_vec=[] 
    temp_acc_vec=[]

    for i in range(iterations):
        
        if i%10==0 and i>0: #every ten iterations review results and decide if a new bubble is needed
            acc_round=[]
            acc_mean=[]
            for acc in temp_acc_vec: #get the relevant accuracy vector  
                if img_label_idx==np.argmax(acc[0,:]): #if classification is correct, add a 1 to an indicator vector
                    acc_mean.append(acc[0,img_label_idx])  
                    acc_round.append(1)
                else:
                    acc_round.append(0)                  
            #print(acc_round)    
            temp_acc_vec=[] #reset the temporary accuracy vector
            
            if sum(acc_round)<=np.round(acc_baseline/2*10) and bubbles<maxbubbles: #if less than 50% acc_baseline out of 10 trials were classiffied correctly
                #print('IF:it:{},acc_sum:{},acc_mean:{}, bubbles:{}'.format(i,sum(acc_round),acc_mean,str(bubbles)))
                bubbles +=1 
            else:
                #print('ELSE:it:{},acc_sum:{}, bubbles:{}'.format(i,sum(acc_round),bubbles))
                bubbles-=1
            if bubbles<10:
                bubbles=20                       
               
        mask=create_bubbles_mask(bubbles, sigma, image_size, offsetpx=offsetpx)
        img_bubbles=apply_bubbles_mask(imagelist, mask, usecomplementmask, biasvalue)
        img_bubbles=[image[np.newaxis,:,:,:] for image in img_bubbles] #add batch dimension needed for classification  
        bubbles_acc = model.predict(img_bubbles) #prediction of the new bubbles image
        bubbles_vec.append(bubbles) #add the actual number of bubbles to the vector for analysis
        acc_vec.append(bubbles_acc) # append the whole accuracy vector, not only the relevant class
        temp_acc_vec.append(bubbles_acc) #append the accuracy vec to a temporary vector that will be used each round of 10
        
        
        if img_label_idx==np.argmax(bubbles_acc[0,:]): #if the classification is correct (argmax(probs) is equal to ground truth label)
            correct_plane.append(mask)
            total_plane.append(mask)
        else:
            total_plane.append(mask) #only to the total, not to the correct                  
    return correct_plane, total_plane, bubbles_vec, acc_vec      




#=====================

def bubbles_correct_plane(correct_plane_list, normalizeimg=True, image_size=(128,128)):
    correctplane_img=np.zeros(image_size,np.float32) 
    for plane in correct_plane_list:
        correctplane_img=correctplane_img+plane
    if normalizeimg:
        correctplane_img=normalize(correctplane_img)
    return correctplane_img    
        

def bubbles_total_plane(total_plane_list, normalizeimg=True, image_size=(128,128)):
    totalplane_img=np.zeros(image_size,np.float32) 
    for plane in total_plane_list:
        totalplane_img=totalplane_img+plane
    if normalizeimg:
        totalplane_img=normalize(totalplane_img)
    return totalplane_img    

def bubbles_remove_img_outliers(img, std_limit_down=3, std_limit_up=3):
    assert len(img.shape)==2, 'Input image should be 2D'
    std=np.std(img)
    mean=np.mean(img)
    img[img>mean+std_limit_up*std]=mean+std_limit_up*std
    img[img<mean-std_limit_down*std]=mean-std_limit_down*std
    return img







'''

tic=time.time()
correct_plane, total_plane, bubbles_vec, acc_vec = bubbles_technique(iterations=1000, 
                                                                     model=model, 
                                                                     imagelist=imagePMK, 
                                                                     bubbles=10, 
                                                                     sigma=10, 
                                                                     img_label_idx=2, 
                                                                     acc_baseline=0.99, 
                                                                     acc_threshold=0.75)
toc=time.time()-tic
print(toc)




correctplane_img=bubbles_correct_plane(correct_plane)
totalplane_img=bubbles_total_plane(total_plane)


plt.imshow(normalize(correctplane_img/(totalplane_img+1e-5)))

plt.imshow(normalize(correctplane_img-(totalplane_img+1e-5)))

plt.hist(normalize(correctplane_img-(totalplane_img+1e-5)).ravel())



mask=create_bubbles_mask(bubbles, sigma, image_size)
img_bubbles=apply_bubbles_mask(imagelist=imagePMK,mask=mask,usecomplementmask=True, biasvalue=0.48)
plt.imshow(img_bubbles[1])
    
'''
