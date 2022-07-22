# python3
# -*- coding: utf-8 -*-
# @Author  : liang
# @Time    : 2021/11/23 14:05
"""
Autoencoder with 4 layer encoder and 4 layer decoder.
"""
import numpy as np

#from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Input
import matplotlib.pyplot as plt
from keras import regularizers
from helpers.dataframe_helper import load_data
import helpers.dataframe_helper
import pandas as pd
from helpers.dataframe_helper import df_get, df_transform_to_numeric, df_encode_objects, save_to_csv, write_to_csv
import scipy.io
from keras.models import model_from_json
np.random.seed(33)   # random seed，to reproduce results.

ENCODING_DIM_INPUT = 16
ENCODING_DIM_LAYER1 = 128
ENCODING_DIM_LAYER2 = 32
ENCODING_DIM_LAYER3 = 8
ENCODING_DIM_OUTPUT = 2
EPOCHS = 20
BATCH_SIZE = 64

def train(x_train):

    # input placeholder
    input_image = Input(shape=(ENCODING_DIM_INPUT, ))

    # encoding layer
    # *****this code is changed compared with Autoencoder, adding the activity_regularizer to make the input sparse.
    encode_layer1 = Dense(ENCODING_DIM_LAYER1, activation='relu', activity_regularizer=regularizers.l1(10e-6),name="Dense_1")(input_image)
    # ******************************
    encode_layer2 = Dense(ENCODING_DIM_LAYER2, activation='relu', name="Dense_2")(encode_layer1)
    encode_layer3 = Dense(ENCODING_DIM_LAYER3, activation='relu',name="Dense_3")(encode_layer2)
    encode_output = Dense(ENCODING_DIM_OUTPUT)(encode_layer3)

    # decoding layer
    decode_layer1 = Dense(ENCODING_DIM_LAYER3, activation='relu')(encode_output)
    decode_layer2 = Dense(ENCODING_DIM_LAYER2, activation='relu')(decode_layer1)
    decode_layer3 = Dense(ENCODING_DIM_LAYER1, activation='relu')(decode_layer2)
    decode_output = Dense(ENCODING_DIM_INPUT, activation='tanh')(decode_layer3)

    # build autoencoder, encoder
    autoencoder = Model(inputs=input_image, outputs=decode_output)
    encoder = Model(inputs=input_image, outputs=encode_output)

    # compile autoencoder
    autoencoder.compile(optimizer='adam', loss='mse')

    # training
    autoencoder.fit(x_train, x_train, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True)

    return encoder, autoencoder


def Auto_train(x_train_data, output_model_dir):
    print('.......................Training Data loading........................')
    #train_file_path='/media/liang/data4T/Onedrive/IoT-23/4_experiments/F14_S04_R_10_000//data//normal_10_000_clean.csv_train.csv'
    #train_file_path = source_file_name

    #classification_col_name='detailed-label'
    #x_train, y_train = load_data(train_file_path, classification_col_name)
    x_train=x_train_data.values
    x_train = x_train.reshape((x_train.shape[0],x_train.shape[1],1)) # Output clean speech (Y_Train.shape[0],Y_Train.shape[1],1 to match dimension)


    encoder, autoencoder = train(x_train=x_train)
#%%
#save model
    model_json = autoencoder.to_json()
    output_model_path=output_model_dir+'model.json'
    with open(output_model_path,"w") as json_file:
        json_file.write(model_json)
    output_weights_path=output_model_dir+'model.h5'
    autoencoder.save_weights(output_weights_path)
    print(".............Saved model to disk successfully......")