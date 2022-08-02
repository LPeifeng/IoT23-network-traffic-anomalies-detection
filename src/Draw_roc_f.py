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
from Mwmote import MWMOTE,Borderline_SMOTE1,SDSMOTE,SMOTE_ENN,SMOTE
from sklearn.metrics import (log_loss, roc_auc_score, accuracy_score,
                             confusion_matrix, f1_score)
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

def roc_drawing(data_file_dir,data_file_prename,experiments_dir):
	#print('Split dataset into training and testing data...')
	data_file_name=data_file_prename+'.csv'
	

	'''
	split_into_train_and_test(data_file_dir,
	                                  data_file_name,
	                                  experiments_dir)
	logging.info("****** Experiment data is in " + experiments_dir)
	'''

	print('Loading data from file...')


	train_data_path=experiments_dir + data_file_prename + '.csv_train.csv'
	x_train, y_train = load_data(train_data_path, '16')
	test_data_path=experiments_dir + data_file_prename + '.csv_test.csv'
	x_test, y_test = load_data(test_data_path, '16')
	y_train=y_train.values
	y_test=y_test.values
	y_train[np.where(y_train>=1)]=1
	y_test[np.where(y_test>=1)]=1
	num_minority= int(np.sum(y_train))

	x_train1=x_train
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
	

	#print('Computing k-nn of %d class of positive... '% k)
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
	train_size=x_train.shape[0]
	#print('Begining %d oversampling:'% k)
	kernel_train_i=np.zeros((train_size+add_num_per*num_minority,train_size+add_num_per*num_minority))
	kernel_train_i[:train_size,:train_size]=kernel_train
	n_row=train_size
	add_infor=np.zeros((add_num_per*num_minority,3))
	add_count=0
	print('oversampling ...')
	for i in range(num_minority):
		for j in range(add_num_per):
			lamda=np.random.random()
			ii=id_p[i]
			#add_row=np.zeros((n_row,1))
			kernel_train_i[:n_row,n_row]=(1-lamda)*kernel_train_i[:n_row,ii]+lamda*kernel_train_i[:n_row,add_id_original[j,i]]
			kernel_train_i[n_row,:n_row]=kernel_train_i[:n_row,n_row].T
			knn=(1-lamda)*(1-lamda)*kernel_train_i[ii,ii]+2*lamda*(1-lamda)*kernel_train_i[ii,add_id_original[j,i]]+lamda*lamda*kernel_train_i[add_id_original[j,i],add_id_original[j,i]]

			kernel_train_i[n_row,n_row]=knn
			n_row=n_row+1
			add_infor[add_count,0]=lamda
			add_infor[add_count,1]=ii
			add_infor[add_count,2]=add_id_original[j,i]
			add_count=add_count+1
	#print('Preprocess %d oversampled training kernel:'% k)		
	add_labels=np.ones((train_size+add_num_per*num_minority,))
	add_labels[:train_size]=y_train

	print('Proposed classifying...')
	auto_clf = svm.SVC(kernel='precomputed')
	auto_clf.fit(kernel_train_i, add_labels)

	#compute kernel matric
	print('Computing kernel of testing data...')
	kernel_test = (1 + np.dot(x_test, x_train.T)) * np.dot(h6_test, h6_train.T)
	kernel_test = (1 + kernel_test) * np.dot(h7_test, h7_train.T)
	#print('Computing kernel of training data1...')
	kernel_test = (1 + kernel_test) * np.dot(h8_test, h8_train.T)
	test_size=x_test.shape[0]
	print('Computing kernel of testing data after oversampling...')
	kernel_test_ov = np.zeros((test_size,train_size+add_num_per*num_minority))
	kernel_test_ov[:,:train_size]=kernel_test
	add_test_row = train_size
	for i in range(add_infor.shape[0]):
		kernel_test_ov[:,add_test_row] = (1-add_infor[i,0])*kernel_test_ov[:,int(add_infor[i,1])]+add_infor[i,0]*kernel_test_ov[:,int(add_infor[i,2])]
		add_test_row = add_test_row +1


	x_train1=x_train.values
	x_test=x_test.values
	print('MWMOTE oversampling and classifying...')
	Syntheic_sample = MWMOTE(proportion=0.5)
	x_train_new,y_train_new = Syntheic_sample.sample(x_train1,y_train)
	mwmote_clf = svm.SVC(C=0.3,kernel='rbf')
	mwmote_clf.fit(x_train_new, y_train_new)

	print('EFS oversampling and classifying...')
	Syntheic_sample = SMOTE_ENN(proportion=0.5)
	x_train_new,y_train_new = Syntheic_sample.sample(x_train1,y_train)
	efs_clf = svm.SVC(C=0.3,kernel='rbf')
	efs_clf.fit(x_train_new, y_train_new)

	print('B_SMOTE oversampling and classifying...')
	Syntheic_sample = Borderline_SMOTE1(proportion=0.5)
	x_train_new,y_train_new = Syntheic_sample.sample(x_train1,y_train)
	bsmote_clf = svm.SVC(C=0.3,kernel='rbf')
	bsmote_clf.fit(x_train_new, y_train_new)

	print('S_SMOTE oversampling and classifying...')
	Syntheic_sample = SDSMOTE(proportion=0.5)
	x_train_new,y_train_new = Syntheic_sample.sample(x_train1,y_train)
	ssmote_clf = svm.SVC(C=0.3,kernel='rbf')
	ssmote_clf.fit(x_train_new, y_train_new)

	mwmote_auc = roc_auc_score(y_test,mwmote_clf.decision_function(x_test))
	auto_auc = roc_auc_score(y_test,auto_clf.predict(kernel_test_ov))
	fpr,tpr, thresholds = roc_curve(y_test,mwmote_clf.decision_function(x_test))
	plt.plot(fpr,tpr,color='darkviolet',label='Proposed(area = %0.2f)' % mwmote_auc)
	fpr,tpr, thresholds = roc_curve(y_test,auto_clf.predict(kernel_test_ov))
	plt.plot(fpr,tpr,color='r',label='MWMOTE(area = %0.2f)' % auto_auc)
	

	
	
	efs_auc = roc_auc_score(y_test,efs_clf.decision_function(x_test))
	fpr,tpr, thresholds = roc_curve(y_test,efs_clf.decision_function(x_test))
	plt.plot(fpr,tpr,color='g',label='EFS(area = %0.2f)' % efs_auc)

	bsmote_auc = roc_auc_score(y_test,bsmote_clf.decision_function(x_test))
	fpr,tpr, thresholds = roc_curve(y_test,bsmote_clf.decision_function(x_test))
	plt.plot(fpr,tpr,color='b',label='B-SMOTE(area = %0.2f)' % bsmote_auc)

	ssmote_auc = roc_auc_score(y_test,ssmote_clf.decision_function(x_test))
	fpr,tpr, thresholds = roc_curve(y_test,ssmote_clf.decision_function(x_test))
	plt.plot(fpr,tpr,color='darkred',label='S_SMOTE(area = %0.2f)' % ssmote_auc)
	'''
	fpr,tpr, thresholds = roc_curve(y_test,clf.decision_function(X_test))
	 
	plt.plot(fpr,tpr,label='ROC')
	 
	plt.xlabel('FPR')
	plt.ylabel('TPR')
	plt.show()
	'''
	plt.plot([0, 1], [0, 1], color='k', lw=2)
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title(data_file_prename)
	plt.legend(loc="lower right")
	file=data_file_prename+'.jpg'
	plt.savefig(file,dpi=800)
	plt.show()