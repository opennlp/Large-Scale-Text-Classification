from sklearn.model_selection import train_test_split
from constants import commonconstants


def get_train_test_data(train_data,class_labels):
    return train_test_split(train_data, class_labels, test_size=commonconstants.TEST_FRACTION, random_state=42)
