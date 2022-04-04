#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 12:05:37 2017

@author: cmendezg
"""

def loadmodelandhistory(pathtofiles):
    from keras.models import model_from_json
    import pickle
    # load json and create model
    json_file = open(pathtofiles+'/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    
    
    # load weights into new model
    loaded_model.load_weights(pathtofiles+"/model.h5")
    print("Loaded model from disk")
    
    file_pi2 = open(pathtofiles+'/trainHistory.obj', 'rb')
    loaded_history = pickle.load(file_pi2)
    file_pi2.close()
    
    return (loaded_history, loaded_model)


