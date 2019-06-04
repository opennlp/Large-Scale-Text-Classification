import nltk
import numpy as np


def get_text_summary_statistics(sentence_list):
    summary_dict = dict({})
    summary_dict['num_sentences'] = len(sentence_list)
    token_count = 0.0
    for sentence in sentence_list:
        tokenized_text = nltk.word_tokenize(sentence)
        token_count = token_count + len(tokenized_text)
    summary_dict['num_tokens'] = token_count
    summary_dict['average_sentence_length'] = token_count/summary_dict['num_sentences']
    return summary_dict


def get_class_imbalance_score(class_iterable):
    unique_labels, counts = np.unique(class_iterable, return_counts=True)
    label_count_dict = dict(zip(unique_labels, counts))
    count_list = list(label_count_dict.values())
    imbalance_score = 0.0
    for i in range(len(count_list)):
        for j in range(i+1,len(count_list)):
            count_difference = abs(count_list[i] - count_list[j]) * 1.0
            count_sum = count_list[i] + count_list[j]
            imbalance_score = imbalance_score + (count_difference/count_sum)
    return imbalance_score
