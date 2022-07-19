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

data_samples = [
    get_imbalance_data_sample(dataset_name='S02', rows_majority_dataset_file=20_000, rows_minority_dataset_file=1_000),  # ~ 10 min
    get_imbalance_data_sample(dataset_name='S22', rows_majority_dataset_file=20_000, rows_minority_dataset_file=2_000),  # ~ 10 min
]

run_imbalance_data_preprocessing(source_files_dir,
                       output_files_dir,
                       iot23_metadata["file_header"],
                       data_cleanup,
                       data_samples=data_samples,
                       overwrite=False)

print('Imbalance data preparation: The end.')
#quit()
