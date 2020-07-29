from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv1D, \
    LSTM, BatchNormalization, MaxPool1D, Dropout, \
    Activation, Add, Input, concatenate, \
    UpSampling1D, LeakyReLU
from tensorflow.keras.activations import relu
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
import numpy as np
from tensorflow.keras.metrics import categorical_accuracy
import pyedflib
from scipy.signal import spectrogram
import random


#Нейронная сеть, которая основа на U-net, к ней добавлен слой LSTM
# U-net neural network with LSTM layers
def Unet(Input_shape):
    # Input_shape - tuple (shape of NN input)
    net_input = Input(Input_shape)
    x = LSTM(100,return_sequences="True")(net_input)
    x = LSTM(100,return_sequences="True")(x)
    x = LSTM(100,return_sequences="True")(x)
    
    x = Conv1D(512, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation(LeakyReLU())(x)
    x = Conv1D(512, 3, padding="same")(x)
    x = BatchNormalization()(x)
    out_1 = x = Activation(LeakyReLU())(x)
    
    x = MaxPool1D()(x)
    x = Conv1D(128, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation(LeakyReLU())(x)
    x = Conv1D(128, 3, padding="same")(x)
    x = BatchNormalization()(x)
    out_2 = x = Activation(LeakyReLU())(x)
    
    x = MaxPool1D()(x)
    x = Conv1D(64, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation(LeakyReLU())(x)
    
    x = Conv1D(64, 3, padding="same")(x)
    x = BatchNormalization()(x)
    out_3 = x = Activation(LeakyReLU())(x)

    x = UpSampling1D()(x)
    x = concatenate([x, out_2])

    x = Conv1D(128, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation(LeakyReLU())(x)
    x = Conv1D(128, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation(LeakyReLU())(x)

    x = UpSampling1D()(x)
    x = concatenate([x, out_1])

    x = Conv1D(512, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation(LeakyReLU())(x)
    x = Conv1D(512, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation(LeakyReLU())(x)
    
    x = Conv1D(3, 1, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation(LeakyReLU())(x)
    
    x = Flatten()(x)
    x = Dense(3, activation="softmax")(x)
    model = Model(net_input, x)
    model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=1e-2), metrics=['accuracy'])
    # returns NN model (keras.Model)
    return model

