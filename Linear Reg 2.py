from __future__ import print_function
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras import metrics
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam, RMSprop
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.utils import plot_model
from keras.models import load_model
from sklearn.model_selection import train_test_split

#df = pd.read_csv('F:\kc_house_data.csv')

dataframe = pd.read_csv("GWDataDL.csv")
#print(dataframe.head())
#dataset = dataframe.values
# split into input (X) and output (Y) variables

X = dataframe.iloc[:,3:7]
Y = dataframe.iloc[:,7]
# print(X)
# print(Y)

x_train, x_test, y_train, y_test = train_test_split(X,Y,
                                                    test_size=0.3, random_state=87)
# print(x_train.shape)
# print(x_test.shape)

np.random.seed(155)
def norm_stats(df1, df2):
    dfs = df1.append(df2)
    minimum = np.min(dfs)
    maximum = np.max(dfs)
    mu = np.mean(dfs)
    sigma = np.std(dfs)
    return (minimum, maximum, mu, sigma)
def z_score(col, stats):
    m, M, mu, s = stats
    df2 = pd.DataFrame()
    for c in col.columns:
        df2[c] = (col[c]-mu[c])/s[c]
    return df2
stats = norm_stats(x_train, x_test)

arr_x_train = np.array(z_score(x_train, stats))
#print(arr_x_train)
arr_y_train = np.array(y_train)
arr_x_valid = np.array(z_score(x_test, stats))
arr_y_valid = np.array(y_test)
# print('Training shape:', arr_x_train.shape)
# print('ddd',arr_y_train.shape)
# print('Training samples: ', arr_x_train.shape[0])
# print('Validation samples: ', arr_x_valid.shape[0])

def basic_model_3(x_size, y_size):
    t_model = Sequential()
    t_model.add(Dense(80, activation="tanh", kernel_initializer='normal', input_shape=(x_size,)))
    t_model.add(Dropout(0.2))
    t_model.add(Dense(120, activation="relu", kernel_initializer='normal',
        kernel_regularizer=regularizers.l1(0.01), bias_regularizer=regularizers.l1(0.01)))
    t_model.add(Dropout(0.1))
    t_model.add(Dense(20, activation="relu", kernel_initializer='normal',
        kernel_regularizer=regularizers.l1_l2(0.01), bias_regularizer=regularizers.l1_l2(0.01)))
    t_model.add(Dropout(0.1))
    t_model.add(Dense(10, activation="relu", kernel_initializer='normal'))
    t_model.add(Dropout(0.0))
    t_model.add(Dense(y_size))
    t_model.compile(
        loss='mean_squared_error',
        optimizer='nadam',
        metrics=[metrics.mae])
    return(t_model)
model = basic_model_3(arr_x_train.shape[1], 1)
model.summary()
epochs = 800
batch_size =128
history = model.fit(arr_x_train, arr_y_train,
    batch_size=batch_size,
    epochs=epochs,
    shuffle=True,
    verbose=2, # Change it to 2, if wished to observe execution
    validation_data=(arr_x_valid, arr_y_valid),)
train_score = model.evaluate(arr_x_train, arr_y_train, verbose=0)
valid_score = model.evaluate(arr_x_valid, arr_y_valid, verbose=0)

print('Train MAE: ', round(train_score[1], 4), ', Train Loss: ', round(train_score[0], 4))
print('Val MAE: ', round(valid_score[1], 4), ', Val Loss: ', round(valid_score[0], 4))
def plot_hist(h, xsize=6, ysize=10):
    # Prepare plotting
    fig_size = plt.rcParams["figure.figsize"]
    plt.rcParams["figure.figsize"] = [xsize, ysize]
    fig, axes = plt.subplots(nrows=4, ncols=4, sharex=True)

# Prediction

# Plotting obs versus Pred

y_pred = model.predict(x_test)#.flatten()

plt.scatter(y_test, y_pred)
plt.xlabel('Observed GW [m]')
plt.ylabel('Predicted GW [m]')
plt.axis('equal')
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_= plt.plot([-100, 100], [-100, 100])
plt.show()



#
#
# # summarize history for MAE
#     plt.subplot(211)
#     plt.plot(h['mean_absolute_error'])
#     plt.plot(h['val_mean_absolute_error'])
#     plt.title('Training vs Validation MAE')
#     plt.ylabel('MAE')
#     plt.xlabel('Epoch')
#     plt.legend(['Train', 'Validation'], loc='upper left')
#
# # summarize history for loss
#     plt.subplot(212)
#     plt.plot(h['loss'])
#     plt.plot(h['val_loss'])
#     plt.title('Training vs Validation Loss')
#     plt.ylabel('Loss')
#     plt.xlabel('Epoch')
#     plt.legend(['Train', 'Validation'], loc='upper left')
#
# # Plot it all in IPython (non-interactive)
#     plt.draw()
#     plt.show()
#     return
# plot_hist(history.history, xsize=8, ysize=12)
