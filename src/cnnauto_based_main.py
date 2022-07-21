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
from cnn_auto_ov_classify import cnn_auto_ov_classify
from iot23 import get_data_Lable_name
import numpy as np

data_file_dir='/media/liang/data4T/Onedrive/IoT-23/3_data/imbalance_data/Prepared_datasets/'
experiments_dir='/media/liang/data4T/Onedrive/IoT-23/3_data/imbalance_data/Prepared_datasets/experiment/'
tr_size=np.zeros((1,7))
mi_size=np.zeros((1,7))
value=np.zeros((7,6))
evalue=np.zeros((7,6))
for x in range(7):
	data_file_pre_name='S'+str(x)+'2'
	tr_size[0,x],mi_size[0,x],d=cnn_auto_ov_classify(data_file_dir,data_file_pre_name,experiments_dir)
	value[x,:]=d[0].T
	evalue[x,:]=d[1].T
print('Results of oversampling classification (using CNN-autoencoder as pre-trained NN):')
for x in range(7):
	data_file_pre_name='S'+str(x)+'2'
	label_name=get_data_Lable_name(data_file_pre_name)
	print('Dataset:', data_file_pre_name)
	print('Majority class:', [label_name[0]], 'num:',int(tr_size[0,x]), ';  Minority class:', [label_name[1]], 'num:',int(mi_size[0,x]))
	print('TP: %.1f FP: %.1f FN: %.1f TN: %.1f TD: %.1f FD: %.1f ' % (value[x,0],value[x,1],value[x,2],value[x,3],value[x,4],value[x,5]))
	print('Recall: %.4f Pre: %.4f TNR: %.4f F-score: %.4f G-mean: %.4f Acc: %.4f' % (evalue[x,0],evalue[x,1],evalue[x,2],evalue[x,3],evalue[x,4],evalue[x,5]))
	print(' ')		