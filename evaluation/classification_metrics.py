from sklearn import metrics


def get_accuracy_score(gold_labels, predicted_labels):
    return metrics.accuracy_score(gold_labels,predicted_labels)


def get_confusion_matrix(gold_labels, predicted_labels):
    return metrics.confusion_matrix(gold_labels, predicted_labels)


def get_precision_recall_f1_support(gold_labels, predicted_labels):
    return metrics.precision_recall_fscore_support(gold_labels, predicted_labels)


def get_roc_auc(gold_labels, predicted_labels):
    return metrics.roc_auc_score(gold_labels, predicted_labels)


def get_classification_report(gold_labels, predicted_labels):
    return metrics.classification_report(gold_labels, predicted_labels)
