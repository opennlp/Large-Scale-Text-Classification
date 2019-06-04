def get_test_data_size_and_index(dataframe_length,test_size_frac):
    test_size = int(dataframe_length*test_size_frac)
    test_start_index = dataframe_length - test_size
    return test_size, test_start_index


def get_number_of_chunks(dataframe_length,chunk_size, test_size):
    num_chunks = int((dataframe_length-test_size)/chunk_size)
    return num_chunks
