from sklearn.svm import SVC
from sklearn.metrics import roc_curve
from sklearn.datasets import make_blobs
from sklearn. model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import (log_loss, roc_auc_score, accuracy_score,
                             confusion_matrix, f1_score)

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
from Draw_roc_f import roc_drawing

data_file_dir='/media/liang/data4T/Onedrive/IoT-23/3_data/imbalance_data/Prepared_datasets/'
experiments_dir='/media/liang/data4T/Onedrive/IoT-23/3_data/imbalance_data/Prepared_datasets/experiment/'
l=['17-1','42-1','60-1']
for data_file_pre_name in l:
	roc_drawing(data_file_dir,data_file_pre_name,experiments_dir)

