# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 09:27:33 2022

@author: eshan
"""
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import mean_absolute_percentage_error

from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, LSTM

#%%

class DataPreprocessing:
    def X_y_split(self,win_size,X):
        X_train = []
        y_train = []
        for i in range(win_size,len(X)):
            X_train.append(X[i-win_size:i])
            y_train.append(X[i])
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        return X_train,y_train
    
class ModelDevelopment:
    def dl_model(self, X_train, nb_node=64):
        model = Sequential()
        model.add(Input(shape=np.shape(X_train)[1:]))
        model.add(LSTM(nb_node))
        model.add(Dense(1,activation='linear'))
        model.summary()
        return model
    
class ModelEvaluation:
     
    def plot_prediction(self,X_test,y_test,model,mms):
        predicted_case = model.predict(X_test)

        actual_case = mms.inverse_transform(y_test)
        actual_predicted_case = mms.inverse_transform(predicted_case)

        plt.figure()
        plt.plot(actual_case, color='r')
        plt.plot(actual_predicted_case, color='b')
        plt.legend(['Actual','Predicted'])
        plt.show()
        
        mape = mean_absolute_percentage_error(actual_case, actual_predicted_case)
        print('MAPE:', mape)


    
    
