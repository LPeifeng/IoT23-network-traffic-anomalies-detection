import tensorflow as tf
import tensorflow

from tensorflow import keras

from keras.utils import plot_model
from keras.models import Model
import numpy as np
from keras.layers import Convolution1D, Dense, MaxPooling1D, AveragePooling1D, Flatten, Input, UpSampling1D
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.optimizers import SGD
from keras import regularizers
from helpers.dataframe_helper import load_data

VERBOSE=1
VALIDATION_SPLIT=0.15  # The portion of data to use for validation

def train(x_train):
    inpt = Input(shape=(16,1))

    conv1 = Convolution1D(128,3,activation='relu',padding='same',strides=1)(inpt)
    conv2 = Convolution1D(128,3,activation='relu',padding='same',strides=1)(conv1)
    pool1 = MaxPooling1D(pool_size=2, strides=None, padding='valid')(conv2)


    conv3 = Convolution1D(64,3,activation='relu',padding='same',strides=1)(pool1)
    conv4 = Convolution1D(128,3,activation='relu',padding='same',strides=1)(conv3)
    pool2 = MaxPooling1D(pool_size=2, strides=None, padding='valid')(conv4)


    conv5 = Convolution1D(64,3,activation='relu',padding='same',strides=1)(pool2)
    conv6 = Convolution1D(32,3,activation='relu',padding='same',strides=1)(conv5)
    pool3 = MaxPooling1D(pool_size=2, strides=None, padding='valid')(conv6)


    conv7 = Convolution1D(64,3,activation='relu',padding='same',strides=1)(pool3)
    conv8 = Convolution1D(64,3,activation='relu',padding='same',strides=1)(conv7)
    pool4 = MaxPooling1D(pool_size=2, strides=None, padding='valid')(conv8)


    conv9 = Convolution1D(32,3,activation='relu',padding='same',strides=1)(pool4)
    conv10 = Convolution1D(16,3,activation='relu',padding='same',strides=1, name='cnn_last_layer')(conv9)

    #encode_layer2 = Dense(32, activation='relu', name="Dense_1")(conv10)
    #encode_layer3 = Dense(10, activatiupsmp1 = UpSampling1D(size=2)(conv11) on='relu',name="Dense_2")(encode_layer2)
    #encode_layer2 = Dense(32, activation='relu', name="Dense_1")(conv10)
    encode_layer3 = Dense(16, activation='relu',name="Dense_2")(conv10)
    encoder=Model(inputs=inpt, outputs=encode_layer3)
    encoder.summary()

    #input_decoder = Input(shape = (1, 16))

    #############

    input_decoder = Input(shape = (1, 16)) ############# 
    #encode_layer2 = Dense(32, activation='relu', name="Dense_1")(input_decoder)
    conv11 = Convolution1D(16,3,activation='relu',padding='same',strides=1)(input_decoder)
    conv12= Convolution1D(32,3,activation='relu',padding='same',strides=1)(conv11)    
    upsmp1 = UpSampling1D(size=2)(conv12) 

    conv13 = Convolution1D(64,3,activation='relu',padding='same',strides=1)(upsmp1)
    conv14 = Convolution1D(64,3,activation='relu',padding='same',strides=1)(conv13)
    upsmp2 = UpSampling1D(size=2)(conv14)  

    conv15 = Convolution1D(64,3,activation='relu',padding='same',strides=1)(upsmp2)
    conv16 = Convolution1D(32,3,activation='relu',padding='same',strides=1)(conv15)
    upsmp3 = UpSampling1D(size=2)(conv16)

    conv17 = Convolution1D(64,3,activation='relu',padding='same',strides=1)(upsmp3)
    conv18 = Convolution1D(128,3,activation='relu',padding='same',strides=1)(conv17)
    upsmp4 = UpSampling1D(size=2)(conv18) 

    conv19 = Convolution1D(128,3,activation='relu',padding='same',strides=1)(upsmp4)
    conv20 = Convolution1D(128,3,activation='relu',padding='same',strides=1)(conv19)

    '''
    conv11 = Convolution1D( 4, 3, activation='relu', padding='same')(upsmp1) 
    upsmp1 = UpSampling1D(size=2)(conv11) 

    conv11 = Convolution1D( 8, 3, activation='relu', padding='same')(upsmp1) 
    conv12 = Convolution1D( 8, 3, activation='relu', padding='same')(conv11) 
    pool4 = UpSampling1D(size=4)(conv12) 
    '''

    conv21 = Convolution1D( 1, kernel_size = (3), activation='tanh', padding='same')(conv20) 
    decoder = Model(inputs=input_decoder, outputs=conv21)
    decoder.summary()

    autoencoder_outputs = decoder(encoder(inpt))
    autoencoder= Model(inpt, autoencoder_outputs, name='AE')
    
    print('....................Model Created Sucessfully...................')
    print(autoencoder.summary())
    #
    print('........................Model Compiling.........................')
    autoencoder.compile(loss='mse',optimizer='adam', metrics=['mse'])
    print('.................Model Compiled Successfully....................')
    #
    #
    devices = tf.config.experimental.list_physical_devices('GPU')
    #tf.config.experimental.set_memory_growth(devices[0], True)
    print('........................Model Training.........................')
    autoencoder.fit(x_train,x_train,512,35,verbose=VERBOSE,validation_split=VALIDATION_SPLIT) # batch_size = 512, epoch = 50
    print('.................Model Trained Succesfully.....................')
    return encoder,autoencoder

def g_CNN_auto_train(x_train_data, output_model_dir):
    print('.......................Training Data loading........................')
    #train_file_path='/media/liang/data4T/Onedrive/IoT-23/4_experiments/F14_S04_R_10_000//data//normal_10_000_clean.csv_train.csv'
    
    x_train=x_train_data.values
    x_train = x_train.reshape((x_train.shape[0],x_train.shape[1],1)) 


    encoder, autoencoder = train(x_train=x_train)
#%%
#save model
    model_json = encoder.to_json()
    output_model_path=output_model_dir+'g_CNN_model.json'
    with open(output_model_path,"w") as json_file:
        json_file.write(model_json)
    output_weights_path=output_model_dir+'g_CNN_model.h5'
    encoder.save_weights(output_weights_path)
    print(".............Saved model to disk successfully......")


def softmax_train(x_train,y_train):

    inpt = Input(shape=(16,))

    FC_layer2 = Dense(64, activation='relu', name="FC_layer2")(inpt)
    FC_layer3 = Dense(32, activation='relu',name="FC_layer3")(FC_layer2)
    y=Dense(1,activation='softmax')(FC_layer3)
       
    FCnet=Model(inputs=inpt, outputs=y)
    #FCnet.summary()


    FCnet.compile(loss='binary_crossentropy',
    optimizer='rmsprop')
    #,    metrics=['accuracy'])

    FCnet.fit(x_train, y_train,
    epochs=20,
    batch_size=128)

    print('....................Softmax Model Created Sucessfully...................')
    print(FCnet.summary())
    #
    print('........................Softmax  Model Compiling.........................')
    FCnet.compile(loss='binary_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy'])
    print('.................Softmax Model Compiled Successfully....................')
    #
    #
    #devices = tf.config.experimental.list_physical_devices('GPU')
    print('........................Softmax Model Training.........................')
    FCnet.fit(x_train, y_train,
        epochs=20,
        batch_size=128)
    print('.................Model Trained Succesfully.....................')
    return FCnet

def softmax_classifier_train(x_train, y_train, output_model_dir):
    print('.......................Training Data loading........................')
       
    #x_train=x_train_data.values
    #x_train = x_train.reshape((x_train.shape[0],x_train.shape[1],1)) 

    FCnet= softmax_train(x_train=x_train,y_train=y_train)
#%%
#save model
    model_json = FCnet.to_json()
    output_model_path=output_model_dir+'sf_model.json'
    with open(output_model_path,"w") as json_file:
        json_file.write(model_json)
    output_weights_path=output_model_dir+'sf_model.h5'
    FCnet.save_weights(output_weights_path)
    print(".............Saved model to disk successfully......")