
#from src.helpers.data_helper import run_data_preprocessing
#from src.helpers.data_stats_helper import explore_clean_data, explore_experiments_train_test_data
#from src.helpers.experiments_helper import run_experiments
#from src.helpers.file_helper import list_folder_names
#from src.helpers.model_stats_helper import run_experiments_reports
#from src.helpers.report_helper import combine_reports
#from src.iot23 import iot23_metadata, data_cleanup

from helpers.data_helper import run_data_preprocessing
from helpers.data_stats_helper import explore_clean_data, explore_experiments_train_test_data
from helpers.experiments_helper import run_experiments
from helpers.file_helper import list_folder_names
from helpers.model_stats_helper import run_experiments_reports
from helpers.report_helper import combine_reports
from iot23 import iot23_metadata, data_cleanup



def run_end_to_end_process(source_files_dir,
                           data_dir,
                           experiments_dir,
                           data_samples,
                           features,
                           training_algorithms,
                           overwrite=False,
                           enable_data_preprocessing=True,
                           enable_clean_data_charts=True,
                           enable_train_data_charts=True,
                           enable_experiment_data_preparation=True,
                           enable_model_training=True,
                           enable_score_tables=True,
                           enable_score_charts=True,
                           plot_corr=True,
                           plot_cls_dist=True,
                           plot_attr_dist=True,
                           enable_model_insights=True,
                           enable_final_report=True,
                           final_report_name='all_scores.xlsx'):
    file_header = iot23_metadata["file_header"]

    # Prepare Data
    if enable_data_preprocessing:
        run_data_preprocessing(source_files_dir,
                               data_dir,
                               file_header,
                               data_cleanup,
                               data_samples=data_samples,
                               overwrite=overwrite)

    # Explore Data
    if enable_clean_data_charts:
        explore_clean_data(data_dir,
                           data_samples=data_samples,
                           plot_corr=plot_corr,
                           plot_cls_dist=plot_cls_dist,
                           plot_attr_dist=plot_attr_dist)

    # Run Experiments
    if enable_experiment_data_preparation or enable_model_training:
        run_experiments(data_dir,
                        experiments_dir,
                        data_samples,
                        features,
                        training_algorithms,
                        overwrite=overwrite,
                        enable_exp_data_preparation=enable_experiment_data_preparation,
                        enable_model_training=enable_model_training)

    # Explore Train/ Test Data
    if enable_train_data_charts:
        explore_experiments_train_test_data(experiments_dir,
                                            data_samples,
                                            features,
                                            plot_corr=plot_corr,
                                            plot_cls_dist=plot_cls_dist,
                                            plot_attr_dist=plot_attr_dist)

    # Explore Experiments Output
    run_experiments_reports(experiments_dir,
                            data_samples,
                            features,
                            enable_score_tables=enable_score_tables,
                            enable_score_charts=enable_score_charts,
                            enable_model_insights=enable_model_insights)

    # Combine all stats into single XLS
    if enable_final_report:
        experiments_folder_list = list_folder_names(experiments_dir)
        combine_reports(experiments_dir, experiments_folder_list, final_report_name)
