#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 15:28:29 2017

@author: cmendezg
"""

def savemodelandhistory_3losses(history, model=None, scriptname=None,timeelapsed=0, YMDfolder=True, directorypath='.'): 
    #saves history and model in YMD folder (uness otherwise specified)
    #history should be the history object returned by keras 
    import time
    import os
    import pickle
    import sys
    
    if YMDfolder == True:
        directorypath=time.strftime("./%Y-%m-%d") # make dir with YearMonthDay
        numbah=1
        while os.path.isdir(directorypath)== True: #if the directory exists create a new name instead of overwriting
            directorypath=directorypath.split('_')[0]
            directorypath=directorypath+'_'+str(numbah)
            numbah+=1   
    else:  #in case the user wants a specific directory      
        directorypath=directorypath    
    os.makedirs(directorypath, exist_ok=True)
    
    if scriptname is None:
        scriptname=''
    
    #save elapsed time in a txt file    
    filepath = directorypath + "/ElapsedTimeCNN.txt"
    


    orig_stdout=sys.stdout # to save model,summary() , look into the next block
    with open(filepath, "a") as text_file:
            print('script: ', scriptname, file=text_file)
            print('=======================================================', file=text_file)
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),file=text_file)
            m, s = divmod(timeelapsed, 60)
            h, m = divmod(m, 60)
            print ("%d:%02d:%02d" % (h, m, s))
            print('Training elapsed time: {} seconds'.format (timeelapsed), file=text_file)
            print('Training elapsed time in H:M:S: {0:.0f}:{1:.0f}:{2:.0f}'.format(h, m, s), file=text_file)
            print('=======================================================', file=text_file)
            print('', file=text_file)
            #-------------------------------------------
            # a non elegant way to redirect the stdout model.summary()
            sys.stdout=text_file
            print(model.summary(), file=text_file)
            sys.stdout=orig_stdout
            #-------------------------------------------
            print('=======================================================', file=text_file)
            print('', file=text_file)
            print('---------')           
            print('loss:', file=text_file)
            print(history.history['loss'], file=text_file)
            print('---------')
            print('predictions_loss:', file=text_file)
            print(history.history['predictions_loss'], file=text_file)
            print('---------')
            print('P_loss:', file=text_file)
            print(history.history['softmaxLossP_loss'], file=text_file)
            print('---------')            
            print('M_loss:', file=text_file)
            print(history.history['softmaxLossM_loss'], file=text_file)
            print('---------') 
            print('K_loss:', file=text_file)
            print(history.history['softmaxLossK_loss'], file=text_file)
            print('---------')            
            print('---------')
            print('predictions_acc:', file=text_file)
            print(history.history['predictions_acc'], file=text_file)
            print('---------')
            print('P_acc:', file=text_file)
            print(history.history['softmaxLossP_acc'], file=text_file)
            print('---------')            
            print('M_acc:', file=text_file)
            print(history.history['softmaxLossM_acc'], file=text_file)
            print('---------') 
            print('K_acc:', file=text_file)
            print(history.history['softmaxLossK_acc'], file=text_file)
            print('---------')            
            print('')

            print('', file=text_file)
            print('---------')           
            print('val_loss:', file=text_file)
            print(history.history['val_loss'], file=text_file)
            print('---------')
            print('val_predictions_loss:', file=text_file)
            print(history.history['val_predictions_loss'], file=text_file)
            print('---------')
            print('val_P_loss:', file=text_file)
            print(history.history['val_softmaxLossP_loss'], file=text_file)
            print('---------')            
            print('val_M_loss:', file=text_file)
            print(history.history['val_softmaxLossM_loss'], file=text_file)
            print('---------') 
            print('val_K_loss:', file=text_file)
            print(history.history['val_softmaxLossK_loss'], file=text_file)
            print('---------')            
            print('---------')
            print('val_predictions_acc:', file=text_file)
            print(history.history['val_predictions_acc'], file=text_file)
            print('---------')
            print('val_P_acc:', file=text_file)
            print(history.history['val_softmaxLossP_acc'], file=text_file)
            print('---------')            
            print('val_M_acc:', file=text_file)
            print(history.history['val_softmaxLossM_acc'], file=text_file)
            print('---------') 
            print('val_K_acc:', file=text_file)
            print(history.history['val_softmaxLossK_acc'], file=text_file)
            print('---------')            
            print('')

            print('=======================================================', file=text_file)
            print('=======================================================', file=text_file)

    #save training history as a dictionary
    with open(directorypath+'/trainHistory.obj', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)   
        
    # SAVE MODEL and TRAINING HISTORY
    model_json = model.to_json()  # serialize model to JSON
    with open(directorypath+"/model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(directorypath + "/model.h5")
    print("Saved model to disk")
    
    return directorypath	
        
    
    
'''        
    # save training graphs
    import matplotlib.pyplot as plt
    # summarize history for accuracy
    plt.figure(0)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(directorypath+'/acc.png')
    
    plt.figure(1)
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(directorypath+'/loss.png')
'''
    
