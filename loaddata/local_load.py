import os
from constants import commonconstants


def get_file_paths_for_categories(category_name='all', data_folder_name_or_path=commonconstants.DATA_FOLDER_NAME_OR_PATH):
    category_file_dict = dict({})
    if category_name != 'all':
        category_list = os.listdir(data_folder_name_or_path)
        category_list = list(filter(lambda x: x == category_name, category_list))
    else:
        category_list = os.listdir(data_folder_name_or_path)
    for category in category_list:
        category_file_dict[category] = list([])
    for category in category_list:
        category_full_path = os.path.join(data_folder_name_or_path, category)
        file_name_list = os.listdir(category_full_path)
        for file_name in file_name_list:
            full_file_path = os.path.join(category_full_path, file_name)
            category_file_list = category_file_dict[category]
            category_file_list.append(full_file_path)
            category_file_dict[category] = category_file_list
    return category_file_dict
