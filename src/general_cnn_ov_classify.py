#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Liang, Peifeng 
# @Link    : ${link}
# @Version : $Id$
# Reference: autoEncoder:一维CNN自动编码, the website is following: https://download.csdn.net/download/weixin_42166105/16510702?ops_
#   request_misc=%257B%2522request%255Fid%2522%253A%2522166461659716782414934672%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&
#   request_id=166461659716782414934672&biz_id=1&utm_medium=distribute.pc_search_result.none-task-download-2~all~first_rank_ecpm_v1~rank_v31_ecpm-1-16510702-null-null.
#   142^v51^control,201^v3^control_1&utm_term=autoEncoder%3A%E4%B8%80%E7%BB%B4&spm=1018.2226.3001.4187.1


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
from general_cnn_auto import g_CNN_auto_train
from general_cnn_auto import softmax_classifier_train
from keras.models import model_from_json

def g_cnn_auto_ov_classify(data_file_dir,data_file_prename,experiments_dir):
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
	train_size=x_train.shape[0]
	test_size = x_test.shape[0]

	#train autoencoder and save autoencoder model
	g_CNN_auto_train(x_train, experiments_dir)


	#load autoencoder model and generate output of encoder
	model_file_path = experiments_dir + 'g_CNN_model.json'
	with open(model_file_path, 'r') as file:
		model_json = file.read()
	weight_file_path = experiments_dir + 'g_CNN_model.h5' 
	encoder = model_from_json(model_json)

	encoder.load_weights(weight_file_path)

	print(".............Load model successfully......")


	dense1_layer_model = Model(inputs=encoder.input,outputs=encoder.get_layer('Dense_2').output)
	h6_train = dense1_layer_model.predict(x_train).reshape(train_size,16)
	h6_train=x_train

	save_title = experiments_dir + 'g_cnn_output_train.mat'
	scipy.io.savemat(save_title, {'h6_train': h6_train})

	dense1_layer_model = Model(inputs=encoder.input,outputs=encoder.get_layer('Dense_2').output)
	h6_test = dense1_layer_model.predict(x_test).reshape(test_size,16)
	h6_test = x_test
	save_title = experiments_dir + 'g_cnn_output_test.mat'
	scipy.io.savemat(save_title, {'h6_test': h6_test})


	#training softmax classifier:
	softmax_classifier_train(h6_train, y_train, experiments_dir)
	model_file_path = experiments_dir + 'sf_model.json'
	with open(model_file_path, 'r') as file:
		model_json = file.read()
	weight_file_path = experiments_dir + 'sf_model.h5' 
	FCnet = model_from_json(model_json)

	FCnet.load_weights(weight_file_path)

	FC_layer_model = Model(inputs=FCnet.input,outputs=FCnet.get_layer('FC_layer2').output)
	h7_train = FC_layer_model.predict(h6_train)
	FC_layer_model = Model(inputs=FCnet.input,outputs=FCnet.get_layer('FC_layer3').output)
	h8_train= FC_layer_model.predict(h6_train)
	save_title = experiments_dir + 'auto_output_train.mat'
	scipy.io.savemat(save_title, {'h6_train': h6_train, 'h7_train': h7_train, 'h8_train': h8_train})

	FC_layer_model = Model(inputs=FCnet.input,outputs=FCnet.get_layer('FC_layer2').output)
	h7_test = FC_layer_model.predict(h6_test)
	FC_layer_model = Model(inputs=FCnet.input,outputs=FCnet.get_layer('FC_layer3').output)
	h8_test= FC_layer_model.predict(h6_test)
	save_title = experiments_dir + 'auto_output_test.mat'
	scipy.io.savemat(save_title, {'h6_test': h6_test, 'h7_test': h7_train, 'h8_train': h8_train})


	#compute kernel matric
	print('Computing kernel of training data...')
	kernel_train = (1 + np.dot(h6_train, h6_train.T)) * np.dot(h7_train, h7_train.T)
	#kernel_train = (1 + kernel_train) * np.dot(h8_train, h8_train.T)
	#print('Computing kernel of training data1...')
	#kernel_train = (1 + kernel_train) * np.dot(h8_train, h8_train.T)
	#print('Computing kernel of training data2...')
	kernel_train_dial = np.diag(kernel_train)
	#print('Computing kernel of training data3...')

	#print('Oversampling...')
	y_train=y_train.values
	y_test=y_test.values
	y_train[np.where(y_train>=1)]=1
	y_test[np.where(y_test>=1)]=1
	num_minority= int(np.sum(y_train))

	print('Computing k-nn of minority class... ')
	id_p=np.where(y_train==1)[0]
	kernel_p=np.zeros((int(num_minority),int(num_minority))).astype(np.float64)
	for i in range(id_p.shape[0]):
		for j in range(id_p.shape[0]):
			kernel_p[i,j]=kernel_train[id_p[i],id_p[j]]
	kernel_p_dial=np.diag(kernel_p)
	re_kernel_p_dial=np.tile(kernel_p_dial,(int(num_minority),1))
	dist =re_kernel_p_dial+re_kernel_p_dial.T-2*kernel_p
	for i in range(int(num_minority)):
		dist[i,i] = np.NaN
	dist_id=np.argsort(dist,0)

	add_num_per=5
	add_id=dist_id[:add_num_per,:]
	add_id_original=id_p[add_id]

	print('Begining oversampling...')
	kernel_train_i=np.zeros((train_size+add_num_per*num_minority,train_size+add_num_per*num_minority))
	kernel_train_i[:train_size,:train_size]=kernel_train
	n_row=train_size
	add_infor=np.zeros((add_num_per*num_minority,3))
	add_count=0
	#print('oversampling ...')
	for i in range(num_minority):
		for j in range(add_num_per):
			lamda=np.random.random()
			ii=id_p[i]
			#add_row=np.zeros((n_row,1))
			kernel_train_i[:n_row,n_row]=(1-lamda)*kernel_train_i[:n_row,ii]+lamda*kernel_train_i[:n_row,add_id_original[j,i]]
			kernel_train_i[n_row,:n_row]=kernel_train_i[:n_row,n_row].T
			knn=(1-lamda)*(1-lamda)*kernel_train_i[ii,ii]+2*lamda*(1-lamda)*kernel_train_i[ii,add_id_original[j,i]]+lamda*lamda*kernel_train_i[add_id_original[j,i],add_id_original[j,i]]
				
				#kernel_train_i[:n_row,n_row]=add_row
				#kernel_train_i[n_row+1,:n_row]=add_row.T
				
			kernel_train_i[n_row,n_row]=knn
			n_row=n_row+1
			add_infor[add_count,0]=lamda
			add_infor[add_count,1]=ii
			add_infor[add_count,2]=add_id_original[j,i]
			add_count=add_count+1
	#print('Preprocess %d oversampled training kernel:'% k)		
	add_labels=np.ones((train_size+add_num_per*num_minority,))
	add_labels[:train_size]=y_train

	print('SVM classifying...')
	clf = svm.SVC(kernel='precomputed')
	clf.fit(kernel_train_i, add_labels)


	#compute kernel matric
	print('Computing kernel of testing data...')
	kernel_test = (1 + np.dot(h6_test, h6_train.T)) * np.dot(h7_test, h7_train.T)
	kernel_test = (1 + kernel_test) * np.dot(h8_test, h8_train.T)

	test_size=x_test.shape[0]
	print('Computing kernel of testing data after oversampling...')
	kernel_test_ov = np.zeros((test_size,train_size+add_num_per*num_minority))
	kernel_test_ov[:,:train_size]=kernel_test
	add_test_row = train_size
	for i in range(add_infor.shape[0]):
		kernel_test_ov[:,add_test_row] = (1-add_infor[i,0])*kernel_test_ov[:,int(add_infor[i,1])]+add_infor[i,0]*kernel_test_ov[:,int(add_infor[i,2])]
		add_test_row = add_test_row +1

	predict_labels = clf.predict(kernel_test_ov)
	print('Printing class result:')
	c=evaluate.evaluation(predict_labels,y_test)
	d=c.evalue()
	value=d[0]
	evalue=d[1]

	save_title = experiments_dir + '/'+data_file_prename + 'cnn_auto_ovsampl_result.mat'
	scipy.io.savemat(save_title,{'value':value,'evalue':evalue})
	return train_size,num_minority,d