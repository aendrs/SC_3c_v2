#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 11:48:34 2017

@author: cmendezg
"""
def plot_model_history_3losses(pathtofiles='.'):
    
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
    #plt.figure(0)
    plt.figure(num=0, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    ax = plt.subplot(111)
    plt.plot(history['predictions_acc'])
    plt.plot(history['val_predictions_acc'])
    plt.plot(history['softmaxLossP_acc'])
    plt.plot(history['softmaxLossM_acc'])
    plt.plot(history['softmaxLossK_acc'])
    plt.plot(history['val_softmaxLossP_acc'])
    plt.plot(history['val_softmaxLossM_acc'])
    plt.plot(history['val_softmaxLossK_acc'])
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.legend(['train_acc', 'test_acc','P_train_acc','M_train_acc','K_train_acc','P_test_acc','M_test_acc','K_test_acc'], loc='center left',bbox_to_anchor=(1, 0.5))
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.savefig('./acc.png')



    #plt.figure(1)
    plt.figure(num=1, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    ax = plt.subplot(111)
    plt.plot(history['loss'])
    plt.plot(history['predictions_loss'])
    plt.plot(history['val_predictions_loss'])
    plt.plot(history['softmaxLossP_loss'])
    plt.plot(history['softmaxLossM_loss'])
    plt.plot(history['softmaxLossK_loss'])
    plt.plot(history['val_softmaxLossP_loss'])
    plt.plot(history['val_softmaxLossM_loss'])
    plt.plot(history['val_softmaxLossK_loss'])
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.legend(['overall loss','main_loss', 'test_loss','P_train_loss','M_train_loss','K_train_loss','P_test_loss','M_test_loss','K_test_loss'], loc='center left',bbox_to_anchor=(1, 0.5))
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig('./loss.png')


    
    
    