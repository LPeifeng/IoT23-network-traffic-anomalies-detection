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

def or_feature_svm_classify(data_file_dir,data_file_prename,experiments_dir):
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



	#oversampling
	y_train=y_train.values
	y_test=y_test.values
	y_train[np.where(y_train>=1)]=1
	y_test[np.where(y_test>=1)]=1
	train_size=x_train.shape[0]
	num_minority= int(np.sum(y_train))
	print('SVM classifying...')
	clf = svm.SVC(C=0.01,kernel='rbf')
	clf.fit(x_train, y_train)

	num_minority= int(np.sum(y_train))


	
	
	#print('Begining %d class predicting:'% k)
	predict_labels = clf.predict(x_test)
	print('Printing class result:')
	c=evaluate.evaluation(predict_labels,y_test)
	d=c.evalue()
	value=d[0]
	evalue=d[1]

	save_title = experiments_dir + '/'+data_file_prename + 'orig_feature_svm_result.mat'
	scipy.io.savemat(save_title,{'value':value,'evalue':evalue})
	return train_size,num_minority,d
