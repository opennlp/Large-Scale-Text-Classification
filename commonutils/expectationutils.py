from constants import commonconstants


def check_num_row_in_dataframe(dataframe, data_size=commonconstants.SMALL_DATASET_SIZE):
    if len(dataframe) > data_size:
        return True
    return False
