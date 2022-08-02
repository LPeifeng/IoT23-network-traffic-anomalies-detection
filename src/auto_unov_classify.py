#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Liang, Peifeng 
# @Link    : ${link}
# @Version : $Id$
import logging
import os
from keras.models import Model
import numpy as np
from sklearn import svm
import scipy.io
import evaluate
import copy
from helpers.data_helper import split_into_train_and_test
from helpers.dataframe_helper import load_data
from sparse_autoencoder import Auto_train
from keras.models import model_from_json

def auto_unov_classify(data_file_dir,data_file_prename,experiments_dir):
	print('Split dataset into training and testing data...')
	data_file_name=data_file_prename+'.csv'
	

	split_into_train_and_test(data_file_dir,
	                                  data_file_name,
	                                  experiments_dir)
	logging.info("****** Experiment data is in " + experiments_dir)

	print('Loading data from file...')


	train_data_path=experiments_dir + data_file_prename + '.csv_train.csv'
	x_train, y_train = load_data(train_data_path, '16')
	test_data_path=experiments_dir + data_file_prename + '.csv_test.csv'
	x_test, y_test = load_data(test_data_path, '16')

	#train autoencoder and save autoencoder model
	Auto_train(x_train, experiments_dir)


	#load autoencoder model and generate output of encoder
	model_file_path = experiments_dir + 'model.json'
	with open(model_file_path, 'r') as file:
	        model_json = file.read()
	weight_file_path = experiments_dir + 'model.h5' 
	encoder = model_from_json(model_json)

	encoder.load_weights(weight_file_path)

	print(".............Load model successfully......")

	#generate output of train data
	dense1_layer_model = Model(inputs=encoder.input,outputs=encoder.get_layer('Dense_1').output)
	h6_train = dense1_layer_model.predict(x_train)
	dense1_layer_model = Model(inputs=encoder.input,outputs=encoder.get_layer('Dense_2').output)
	h7_train= dense1_layer_model.predict(x_train)
	dense1_layer_model = Model(inputs=encoder.input,outputs=encoder.get_layer('Dense_3').output)
	h8_train = dense1_layer_model.predict(x_train)
	save_title = experiments_dir + 'auto_output_train.mat'
	scipy.io.savemat(save_title, {'h6_train': h6_train, 'h7_train': h7_train, 'h8_train': h8_train})

	#generate output of test data
	dense1_layer_model = Model(inputs=encoder.input,outputs=encoder.get_layer('Dense_1').output)
	h6_test = dense1_layer_model.predict(x_test)
	dense1_layer_model = Model(inputs=encoder.input,outputs=encoder.get_layer('Dense_2').output)
	h7_test = dense1_layer_model.predict(x_test)
	dense1_layer_model = Model(inputs=encoder.input,outputs=encoder.get_layer('Dense_3').output)
	h8_test = dense1_layer_model.predict(x_test)
	save_title = experiments_dir + 'auto_output_test.mat'
	scipy.io.savemat(save_title, {'h6_test': h6_test, 'h7_test': h7_test, 'h8_test': h8_test})


	#compute kernel matric
	print('Computing kernel of training data...')
	kernel_train = (1 + np.dot(x_train, x_train.T)) * np.dot(h6_train, h6_train.T)
	kernel_train = (1 + kernel_train) * np.dot(h7_train, h7_train.T)
	#print('Computing kernel of training data1...')
	kernel_train = (1 + kernel_train) * np.dot(h8_train, h8_train.T)
	#print('Computing kernel of training data2...')
	kernel_train_dial = np.diag(kernel_train)
	#print('Computing kernel of training data3...')



	#oversampling
	y_train=y_train.values
	y_test=y_test.values
	y_train[np.where(y_train>=1)]=1
	y_test[np.where(y_test>=1)]=1
	train_size=x_train.shape[0]
	num_minority= int(np.sum(y_train))
	print('SVM classifying...')
	clf = svm.SVC(kernel='precomputed')
	clf.fit(kernel_train, y_train)

	num_minority= int(np.sum(y_train))


	#compute kernel matric
	print('Computing kernel of testing data...')
	kernel_test = (1 + np.dot(x_test, x_train.T)) * np.dot(h6_test, h6_train.T)
	kernel_test = (1 + kernel_test) * np.dot(h7_test, h7_train.T)
	#print('Computing kernel of training data1...')
	kernel_test = (1 + kernel_test) * np.dot(h8_test, h8_train.T)
	#test_size=x_test.shape[0]
	
	#print('Begining %d class predicting:'% k)
	predict_labels = clf.predict(kernel_test)
	print('Printing class result:')
	c=evaluate.evaluation(predict_labels,y_test)
	d=c.evalue()
	value=d[0]
	evalue=d[1]

	save_title = experiments_dir + '/'+data_file_prename + 'auto_unsampl_result.mat'
	scipy.io.savemat(save_title,{'value':value,'evalue':evalue})
	return train_size,num_minority,d
