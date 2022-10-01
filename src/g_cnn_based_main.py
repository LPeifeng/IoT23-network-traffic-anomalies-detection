#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Liang, Peifeng ()
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
from CNN_autoencoder import CNN_auto_train
from keras.models import model_from_json
from general_cnn_ov_classify import g_cnn_auto_ov_classify
from iot23 import get_data_Lable_name
import numpy as np

data_file_dir='/media/liang/data4T/Onedrive/IoT-23/3_data/imbalance_data/Prepared_datasets/'
experiments_dir='/media/liang/data4T/Onedrive/IoT-23/3_data/imbalance_data/Prepared_datasets/experiment/'
tr_size=np.zeros((1,16))
mi_size=np.zeros((1,16))
value=np.zeros((16,6))
evalue=np.zeros((16,6))
#l=['S02','S12','S32','S42','S52','S62','7-1','17-1','20-1','21-1','39-1','42-1','44-1','48-1','52-1','60-1']
l=['17-1','20-1','21-1','39-1','42-1','44-1','48-1','52-1','60-1']
x=0
for data_file_pre_name in l:
	print('Running', data_file_pre_name, 'dataset...')
	tr_size[0,x],mi_size[0,x],d=g_cnn_auto_ov_classify(data_file_dir,data_file_pre_name,experiments_dir)
	value[x,:]=d[0].T
	evalue[x,:]=d[1].T
	x=x+1
print('Results of oversampling classification (using autoencoder as pre-trained NN):')
x=0
for data_file_pre_name in l:
	label_name=get_data_Lable_name(data_file_pre_name)
	print('Dataset:', data_file_pre_name)
	print('Majority class:', [label_name[0]], 'num:',int(tr_size[0,x]), ';  Minority class:', [label_name[1]], 'num:',int(mi_size[0,x]))
	print('TP: %.1f FP: %.1f FN: %.1f TN: %.1f TD: %.1f FD: %.1f ' % (value[x,0],value[x,1],value[x,2],value[x,3],value[x,4],value[x,5]))
	print('Recall: %.4f Pre: %.4f TNR: %.4f F-score: %.4f G-mean: %.4f Acc: %.4f' % (evalue[x,0],evalue[x,1],evalue[x,2],evalue[x,3],evalue[x,4],evalue[x,5]))
	print(' ')	
	x=x+1	