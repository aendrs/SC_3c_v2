#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 11:48:34 2017

@author: cmendezg
"""
def plot_model_history(pathtofiles='.'):
    
    import matplotlib.pyplot as plt
    from keras.models import model_from_json
    import pickle
    
    # load json and create model
    json_file = open(pathtofiles+'/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    
    
    # load weights into new model
    model.load_weights(pathtofiles+"/model.h5")
    print("Loaded model from disk")
    file_pickle = open(pathtofiles+'/trainHistory.obj', 'rb')
    history = pickle.load(file_pickle)
    file_pickle.close()
    
    
    
    # save training graphs
    # summarize history for accuracy
    plt.figure(0)
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(pathtofiles+'/acc.png')
    
    plt.figure(1)
    # summarize history for loss
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(pathtofiles+'/loss.png')