from sklearn.model_selection import train_test_split
from commonutils import dataframeutils
from constants import commonconstants
from loaddata import local_load
import os


def get_train_test_dataframe(full_file_path,test_size=commonconstants.TEST_FRACTION,random_state=42):
    dataframe = dataframeutils.get_data_frame(full_file_path)
    df_train, df_test = train_test_split(dataframe, test_size=test_size, random_state=random_state)
    return df_train, df_test


def write_dataframe_to_csv(dataframe, file_to_save):
    dataframe.to_csv(file_to_save, index=False, encoding='utf-8')


def create_train_test_cateogy_folder_structure():
    for category_name in commonconstants.CATEGORY_LIST:
        category_path = os.path.join(commonconstants.SPLIT_FOLDER_NAME_OR_PATH,category_name)
        for split_type_name in ['train','test']:
            final_path = os.path.join(category_path,split_type_name)
            os.makedirs(final_path)


def generate_train_test_split_pipeline():
    create_train_test_cateogy_folder_structure()
    category_filepath_dict = local_load.get_file_paths_for_categories()
    for category_name, category_filepath_list in category_filepath_dict.items():
        for category_filename_path in category_filepath_list:
            df_train, df_test = get_train_test_dataframe(category_filename_path)
            split_frame_list = [df_train, df_test]
            dataset_file_name = str(category_filename_path).rsplit("\\", 1)[1]
            for split_type, df_split_frame in zip(commonconstants.DATA_SPLIT_TYPE_LIST,split_frame_list):
                full_path_to_save = os.path.join(commonconstants.SPLIT_FOLDER_NAME_OR_PATH,category_name,split_type,dataset_file_name)
                write_dataframe_to_csv(df_split_frame, full_path_to_save)

