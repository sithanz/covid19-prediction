# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 09:20:56 2022

@author: eshan
"""

import os
import pickle
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import TensorBoard

from covid_prediction_module import DataPreprocessing
from covid_prediction_module import ModelEvaluation, ModelDevelopment

#%% Functions
def lineplot(a):
    plt.figure()
    plt.plot(a)
    plt.show()
    
#%% Constants
MMS_PATH = os.path.join(os.getcwd(),'model', 'mms.pkl')
CSV_PATH_TRAIN = os.path.join(os.getcwd(),'dataset','cases_malaysia_train.csv')
CSV_PATH_TEST = os.path.join(os.getcwd(),'dataset','cases_malaysia_test.csv')
LOGS_PATH = os.path.join(os.getcwd(),'logs',datetime.datetime.now().
                         strftime('%Y%m%d-%H%M%S'))
MODEL_PATH = os.path.join(os.getcwd(),'model','model.h5')

#%% Data Loading

df = pd.read_csv(CSV_PATH_TRAIN, na_values=' ')
df_test = pd.read_csv(CSV_PATH_TEST)

#%% Exploratory Data Analysis - Training data

df.info() 

# Change dtype of columns
df['cases_new'] = pd.to_numeric(df['cases_new'], errors='coerce')

df.describe().T
df.isna().sum() # 12 NaN in cases_new

# Plot cases_new, slice to view part with NaN

lineplot(df['cases_new'][80:120])

#%% Exploratory Data Analysis - Test data

df_test.info()
df_test.describe().T
df_test.isna().sum() # 1 NaN in cases_new

lineplot(df_test['cases_new'])

#%% Data Cleaning

# Fill in NaN by data interpolation
df['cases_new'] = df['cases_new'].interpolate(method='linear')
df_test['cases_new'] = df_test['cases_new'].interpolate(method='linear')

df.isna().sum() # 0 NaN in cases_new

lineplot(df['cases_new'][80:120])
lineplot(df['cases_new'])

#%% Feature Selection

X = df['cases_new']

mms = MinMaxScaler()
X = mms.fit_transform(np.expand_dims(X,axis=-1))

# Save scaler as pkl file
with open(MMS_PATH,'wb') as file:
    pickle.dump(mms,file) 
    
#%% Data Preprocessing - Training data

# Set window size
win_size=30

# Split into X & y training data, convert into array
dp=DataPreprocessing()
X_train, y_train = dp.X_y_split(win_size, X)

#%% Data Preprocessing - Test data
# Concatenate dataframes to extend window for test data
df_concat = pd.concat((df['cases_new'],df_test['cases_new']))

# Determine total number of inputs needed and slice accordingly
length_days = len(df_test) + win_size
tot_input = df_concat[-length_days:]

# Use previously fitted scaler to scale test data
X2 = mms.transform(np.expand_dims(tot_input,axis=-1))

# Split into X & y test data
X_test, y_test = dp.X_y_split(win_size,X2)

#%% Model Development

md = ModelDevelopment()

model = md.dl_model(X_train)

plot_model(model,show_shapes=True,show_layer_names=True)

model.compile(optimizer='adam', loss='mse',
              metrics=['mean_absolute_percentage_error'])

#%%
tensorboard_callback = TensorBoard(log_dir=LOGS_PATH, histogram_freq=1)

hist = model.fit(X_train,y_train, epochs=400,
                 callbacks=[tensorboard_callback])

#%% Model Evaluation

# Plot actual vs predicted values & calculate error (MAPE)
me = ModelEvaluation()
me.plot_prediction(X_test, y_test, model, mms)

#%% Save model

model.save(MODEL_PATH)



















