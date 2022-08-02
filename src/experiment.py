#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Liang, Peifeng ()
# @Link    : ${link}
# @Version : $Id$

#original features and svm classification

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
from CNN_autoencoder import CNN_auto_train
from keras.models import model_from_json
from orig_feature_svm_classify import or_feature_svm_classify
from auto_ov_classify import auto_ov_classify
from iot23 import get_data_Lable_name
import numpy as np
from mwmote_svm import Vsmote_svm

data_file_dir='/media/liang/data4T/Onedrive/IoT-23/3_data/imbalance_data/Prepared_datasets/'
experiments_dir='/media/liang/data4T/Onedrive/IoT-23/3_data/imbalance_data/Prepared_datasets/experiment/'
tr_size=np.zeros((1,16))
mi_size=np.zeros((1,16))
value=np.zeros((10,7,6))
#evalue=np.zeros((16,6))
#l=['S02','S12','S32','S42','S52','S62','7-1','17-1','20-1','21-1','39-1','42-1','44-1','48-1','52-1','60-1']
l=['7-1','17-1','20-1','21-1','39-1','42-1','44-1','48-1','52-1','60-1']
alg=['Proposed','QLSVM','SMOTE','MWMOTE','EFS','S-SMOTE','B-SMOTE']
x=0
for data_file_pre_name in l:

	data_file_name=data_file_pre_name + '.csv'
	

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
	'''
	print('Running', data_file_pre_name, 'dataset...')
	num_alg=0
	print(alg[num_alg],':')
	tr_size[0,x],mi_size[0,x],d=auto_ov_classify(data_file_dir,data_file_pre_name,experiments_dir)
	value[x,num_alg,:]=d[1].T
	num_alg=num_alg+1
	print(alg[num_alg],':')
	tr_size[0,x],mi_size[0,x],d=or_feature_svm_classify(data_file_dir,data_file_pre_name,experiments_dir)
	value[x,num_alg,:]=d[1].T
	for i in range(5):
		num_alg=num_alg+1
		tr_size[0,x],mi_size[0,x],d=Vsmote_svm(data_file_dir,data_file_pre_name,experiments_dir,alg[num_alg])
		value[x,num_alg,:]=d[1].T
	x=x+1
print('Results of svm classification (using original features):')
x=0
for data_file_pre_name in l:
	label_name=get_data_Lable_name(data_file_pre_name)
	print('Dataset:', data_file_pre_name)
	print('Majority class:', [label_name[0]], 'num:',int(tr_size[0,x]), ';  Minority class:', [label_name[1]], 'num:',int(mi_size[0,x]))
	i=0
	for a in alg:
		print(a,':')	
		print('Recall: %.4f Pre: %.4f TNR: %.4f F-score: %.4f G-mean: %.4f Acc: %.4f' % (value[x,i,0],value[x,i,1],value[x,i,2],value[x,i,3],value[x,i,4],value[x,i,5]))
		i=i+1
		print(' ')	
	x=x+1	