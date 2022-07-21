import warnings

import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from config import iot23_attacks_dir, iot23_data_dir, iot23_experiments_dir
from helpers.log_helper import add_logger
from helpers.process_helper import run_end_to_end_process
from iot23 import get_data_sample, iot23_metadata, feature_selections

# Add Logger
add_logger(file_name='demo.log')

# Setup warningsbs
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=sklearn.exceptions.ConvergenceWarning)
from src.helpers.data_helper import run_data_preprocessing
from src.helpers.data_stats_helper import explore_clean_data, explore_experiments_train_test_data
from src.helpers.experiments_helper import run_experiments
from src.helpers.file_helper import list_folder_names
from src.helpers.model_stats_helper import run_experiments_reports
from src.helpers.report_helper import combine_reports
from src.iot23 import iot23_metadata, data_cleanup

file_header = iot23_metadata["file_header"]
source_files_dir = iot23_attacks_dir
data_dir = iot23_data_dir
experiments_dir = iot23_experiments_dir
data_samples = [
    # get_data_sample(dataset_name='S04', rows_per_dataset_file=10_000),
    get_data_sample(dataset_name='S16', rows_per_dataset_file=10_000),
]

# Selected Features
features = [
    feature_selections['F14'],
    # feature_selections['F17'],
    # feature_selections['F18'],
    # feature_selections['F19'],
]

# Selected Algorithms
training_algorithms = dict([
    ('DecisionTree', Pipeline([('normalization', StandardScaler()), ('classifier', DecisionTreeClassifier())])),
    ('GaussianNB', Pipeline([('normalization', StandardScaler()), ('classifier', GaussianNB())])),
    ('LogisticRegression', Pipeline([('normalization', StandardScaler()), ('classifier', LogisticRegression())])),
    ('RandomForest', Pipeline([('normalization', StandardScaler()), ('classifier', RandomForestClassifier())])),
    ('SVC_linear', Pipeline([('normalization', MinMaxScaler()), ('classifier', LinearSVC())])),
])

# Prerequisites:
# 1. Run run_step00_configuration_check.py
# 2. Run run_step01_extract_data_from_scenarios.py
# 3. Run run_step01_shuffle_file_content.py


run_end_to_end_process(source_files_dir,
                       data_dir,
                       experiments_dir,
                       data_samples,
                       features,
                       training_algorithms,
                       overwrite=False,
                       enable_data_preprocessing=False,
                       enable_clean_data_charts=False,
                       enable_experiment_data_preparation=False,
                       enable_train_data_charts=False,
                       enable_model_training=False,
                       enable_score_tables=False,
                       enable_score_charts=True,
                       plot_corr=False,
                       plot_cls_dist=False,
                       plot_attr_dist=False,
                       enable_model_insights=False,
                       enable_final_report=False,
                       final_report_name='custom_experiment_scores.xlsx')
