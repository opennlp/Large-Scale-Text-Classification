from classifiers import non_neural_classifiers
from constants import commonconstants


def get_classifier_from_factory(classifier_type="all"):
    if classifier_type.lower() == commonconstants.ENSEMBLE_TYPE:
        return non_neural_classifiers.get_ensemble_tree_classifiers()
    elif classifier_type.lower() == commonconstants.DISCRIMINANT_TYPE:
        return non_neural_classifiers.get_discriminative_classifiers()
    elif classifier_type.lower() == commonconstants.LINEAR_TYPE:
        return non_neural_classifiers.get_linear_classifiers()
    elif classifier_type.lower() == commonconstants.SGD_TYPE:
        return non_neural_classifiers.get_sgd_classifier()
    elif classifier_type.lower() == commonconstants.WARM_START_TYPE:
        return non_neural_classifiers.get_warm_start_classifiers()
    elif classifier_type.lower() == commonconstants.RF:
        return non_neural_classifiers.get_random_forest_classifier()
    elif classifier_type.lower() == commonconstants.ADA_BOOST:
        return non_neural_classifiers.get_ada_boost_classifier()
    elif classifier_type.lower() == commonconstants.GRAD_BOOST:
        return non_neural_classifiers.get_grad_boost_classifier()
    return non_neural_classifiers.get_all_classifiers()
