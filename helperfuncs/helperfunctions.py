# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 12:36:12 2017

@author: USUARIO

# analogous to Matlab's tic toc,  
# usage:   tic=tic()    timeelapsed=toc(tic)  printelapsedtime(timeelapsed)

def tic():
    import time
    tic = time.time()
    return tic

def toc(tic):
    import time
    now = time.time()
    toc=now - tic
    return toc

def printelapsedtime(toc):
    m, s = divmod(toc, 60)
    h, m = divmod(m, 60)
    tocstr= ("%d:%02d:%02d" % (h, m, s))
    print (tocstr)
    return tocstr
    
    
    

def savemodelandhistory(history, model, timeelapsed=0, YMDfolder=True, directorypath='.'): 
    #saves history and model in YMD folder (uness otherwise specified)
    #history should be the history object returned by keras 
    import time
    import os
    import pickle
    if YMDfolder == True:
        directorypath=time.strftime("./%Y-%m-%d") # make dir with YearMonthDay
        while os.path.isdir(directorypath)== True: #if the directory exists create a new name instead of overwriting
            numbah=1
            directorypath=directorypath+'_'+str(numbah)
            numbah+=1   
    else:  #in case the user wants a specific directory      
        directorypath=directorypath    
    os.makedirs(directorypath, exist_ok=True)
    
    #save elapsed time in a txt file    
    filepath = directorypath + "/ElapsedTimeCNN.txt"
    with open(filepath, "a") as text_file:
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),file=text_file)
            m, s = divmod(timeelapsed, 60)
            h, m = divmod(m, 60)
            print ("%d:%02d:%02d" % (h, m, s))
            print('Training elapsed time: {} seconds'.format (timeelapsed), file=text_file)
            print('Training elapsed time in H:M:S: {0:.0f}:{1:.0f}:{2:.0f}'.format(h, m, s), file=text_file)
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




"""
