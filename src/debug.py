import numpy as np

#from keras.datasets import mnist
from keras.models import Model
import pandas as pd
from keras.layers import Dense, Input
import matplotlib.pyplot as plt
from helpers.dataframe_helper import load_data
import helpers.dataframe_helper
from helpers.dataframe_helper import df_get, df_transform_to_numeric, df_encode_objects, save_to_csv, write_to_csv

np.random.seed(33)   # random seed，to reproduce results.

ENCODING_DIM_INPUT = 14
ENCODING_DIM_LAYER1 = 10
ENCODING_DIM_LAYER2 = 8
ENCODING_DIM_LAYER3 = 4
ENCODING_DIM_OUTPUT = 2
EPOCHS = 20
BATCH_SIZE = 64


def max_min_normalization(data_value):
    """
    函数主体，归一化处理
    Data normalization using max value and min value
    Args:
        data_value: The data to be normalized
    """
    data_shape = data_value.shape
    print(data_value.shape)
    data_rows = data_shape[0]
    data_cols = data_shape[1]
    new_data=np.zeros(shape=(data_rows,data_cols))
    #origin = data_origin[0,:]
    for i in range(0, data_rows, 1):
        for j in range(0, data_cols, 1):
            data_col_min_values = min(data_value[:,j])
            data_col_max_values = max(data_value[:,j])
            new_data[i][j] = (data_value[i][j] - data_col_min_values) / (data_col_max_values - data_col_min_values)
    #new_data = np.vstack((origin,new_data))
    #np.savetxt("G:/Download/maps/wangyd_data/ice2_wyd_nor.csv", new_data, delimiter=',',fmt = '%s')#输出地址
    print('Normalization completed!')
    return new_data
if __name__ == '__main__':
    train_file_path='/media/liang/data4T/Onedrive/IoT-23/4_experiments/F17_S04_R_5_000_000//data//S04_R_5_000_000_clean.csv_train.csv'
    classification_col_name='detailed-label'
    x_train, y_train = load_data(train_file_path, classification_col_name)
    x_train=x_train.values
    y_train=y_train.values
    x_train1=max_min_normalization(x_train)
    train_data=np.column_stack((x_train1,y_train))
    train_file_path='/media/liang/data4T/Onedrive/IoT-23/4_experiments/F17_S04_R_5_000_000//data//normal_5_000_000_clean.csv_train.csv'
    x_train =pd.DataFrame(train_data)
    write_to_csv(x_train, train_file_path, mode='w')
    #print(x_train1)
    print('Training data Normalization completed!')

    test_file_path='/media/liang/data4T/Onedrive/IoT-23/4_experiments/F17_S04_R_5_000_000//data//S04_R_5_000_000_clean.csv_test.csv'
    classification_col_name='detailed-label'
    x_test, y_test= load_data(test_file_path, classification_col_name)
    x_test=x_test.values
    y_test=y_test.values
    x_test1=max_min_normalization(x_test)
    test_data=np.column_stack((x_test1,y_test))
    test_file_path='/media/liang/data4T/Onedrive/IoT-23/4_experiments/F17_S04_R_5_000_000//data//normal_5_000_000_clean.csv_test.csv'
    x_test =pd.DataFrame(test_data)
    write_to_csv(x_test, test_file_path, mode='w')
    print('Testing data Normalization completed!')