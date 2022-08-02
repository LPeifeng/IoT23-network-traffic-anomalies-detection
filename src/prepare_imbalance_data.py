import logging

from config import iot23_attacks_dir, iot23_imbalance_data_dir
from iot23 import iot23_metadata, data_cleanup, get_imbalance_data_sample
from helpers.log_helper import add_logger
from helpers.data_helper import run_imbalance_data_preprocessing

# Add Logger
add_logger(file_name='02_prepare_data.log')
logging.warning("!!! This step takes about 20 min to complete !!!")

# Prepare data
source_files_dir = iot23_attacks_dir
output_files_dir = iot23_imbalance_data_dir

'''
data_samples = [
    get_imbalance_data_sample(dataset_name='S02', rows_majority_dataset_file=30_000, rows_minority_dataset_file=2_000),  
    get_imbalance_data_sample(dataset_name='S12', rows_majority_dataset_file=20_000, rows_minority_dataset_file=1_000),  
    get_imbalance_data_sample(dataset_name='S22', rows_majority_dataset_file=30_000, rows_minority_dataset_file=3_000),  
    get_imbalance_data_sample(dataset_name='S32', rows_majority_dataset_file=30_000, rows_minority_dataset_file=2_000),  
    get_imbalance_data_sample(dataset_name='S42', rows_majority_dataset_file=20_000, rows_minority_dataset_file=1_000),  
    get_imbalance_data_sample(dataset_name='S52', rows_majority_dataset_file=30_000, rows_minority_dataset_file=3_000),  
    get_imbalance_data_sample(dataset_name='S62', rows_majority_dataset_file=40_000, rows_minority_dataset_file=2_000),  
    get_imbalance_data_sample(dataset_name='S72', rows_majority_dataset_file=50_000, rows_minority_dataset_file=2_000),  
]
'''
data_samples = [
    #get_imbalance_data_sample(dataset_name='7-1',  rows_majority_dataset_file=75_955, rows_minority_dataset_file=5_778), 
     
    #get_imbalance_data_sample(dataset_name='17-1', rows_majority_dataset_file=31_438, rows_minority_dataset_file=6_834),  
    #get_imbalance_data_sample(dataset_name='20-1', rows_majority_dataset_file=3_194, rows_minority_dataset_file=30),  
    #get_imbalance_data_sample(dataset_name='21-1', rows_majority_dataset_file=3_272, rows_minority_dataset_file=2_00),  
    #get_imbalance_data_sample(dataset_name='39-1', rows_majority_dataset_file=7_337, rows_minority_dataset_file=677),  
    #get_imbalance_data_sample(dataset_name='42-1', rows_majority_dataset_file=4_420, rows_minority_dataset_file=100),  
    #get_imbalance_data_sample(dataset_name='44-1', rows_majority_dataset_file=211, rows_minority_dataset_file=50),  
    #get_imbalance_data_sample(dataset_name='48-1', rows_majority_dataset_file=3_743, rows_minority_dataset_file=888),  
    #get_imbalance_data_sample(dataset_name='52-1', rows_majority_dataset_file=1_794, rows_minority_dataset_file=100),  
    get_imbalance_data_sample(dataset_name='60-1', rows_majority_dataset_file=2_476, rows_minority_dataset_file=150),  
    
]
run_imbalance_data_preprocessing(source_files_dir,
                       output_files_dir,
                       iot23_metadata["file_header"],
                       data_cleanup,
                       data_samples=data_samples,
                       overwrite=False)

print('Imbalance data preparation: The end.')
#quit()
