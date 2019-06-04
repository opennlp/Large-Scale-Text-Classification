SMALL_DATASET_SIZE = 5000
EMBEDDING_DIMENSION = 300
DATA_FOLDER_NAME_OR_PATH = 'C:\\Users\\rupachak\\Desktop\\Papers-2019\\Large Scale Text Classification Comparison\\standard'
SPLIT_FOLDER_NAME_OR_PATH = 'C:\\Users\\rupachak\\Desktop\\Papers-2019\\Large Scale Text Classification Comparison\\standard-splits'
TEST_FRACTION = 0.2
CHUNK_SIZE = 40000
MAX_FEATURES_TO_SHOW = 10
MAX_INTERPRETATIONS_TO_GENERATE = 10
CATEGORY_LIST = ['reviews','spam-fake-hate-ironic','sentiment', 'emotion', 'news', 'general classification', 'medical',
                 'other']
DATA_SPLIT_TYPE_LIST = ['train','test']
VECTORIZER_LIST = ['featurehash', 'tfidf', 'glove', 'fasttext', 'word2vec', 'elmo', 'flair']
CLASSIFIER_LIST = ['Random Forest', 'Logistic Regression', 'Support Vector Machine (Linear Kernel)',
                   'GradientBoost', 'AdaBoost', 'Stochastic Gradient Descent']
METRIC_LIST = ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1']
METADATA_COLUMN_LIST = ['num_sentences', 'average_sentence_length', 'imbalance_measure', 'num_class_labels','num_tokens']
IMBALANCE_MEASURE_LOWER_QUANT = [0, 1.03]
IMBALANCE_MEASURE_MIDDLE_QUANT = [1.03, 4.4613]
IMBALANCE_MEASURE_UPPER_QUANT = [4.613, 78772241.68]

FEATURE_HASH = 'featurehash'
TF_IDF = 'tfidf'
NEURAL_EMBEDDING_LIST = ['word2vec','glove','fasttext','elmo','flair']
LOG_FILE_PATH = 'C:\\Users\\rupachak\\Documents\\Github\\Large Scale Text Classification\\logs\\app.log'
INTERPRETATION_FOLDER_PATH = 'C:\\Users\\rupachak\\Documents\\Github\\Large Scale Text Classification\\interpretationfiles\\'
ENSEMBLE_TYPE = 'ensemble'
DISCRIMINANT_TYPE = 'discriminant'
LINEAR_TYPE = 'linear'
SGD_TYPE = 'sgd'
RF = 'rf'
ADA_BOOST = 'ada'
GRAD_BOOST = 'grad'
WARM_START_TYPE = 'warm'
RESULT_FOLDER_PATH = 'C:\\Users\\rupachak\\Documents\\Github\\Large Scale Text Classification\\resultfiles\\'
PAPER_FIGURES_PATH = 'C:\\Users\\rupachak\\Desktop\\Papers-2019\\Large Scale Text Classification Comparison\\results\\'
