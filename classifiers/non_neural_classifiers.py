from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier


def get_ensemble_tree_classifiers():
    rf = RandomForestClassifier(n_estimators=51,random_state=42)
    grad = GradientBoostingClassifier(random_state=42)
    ada = AdaBoostClassifier(random_state=42)
    classifier_list = [rf, grad, ada]
    classifier_name_list = ["Random Forest","GradientBoost", "AdaBoost"]
    return classifier_list, classifier_name_list


def get_discriminative_classifiers():
    support_vector = LinearSVC(random_state=42)
    return [support_vector], ["Support Vector Machine (Linear Kernel)"]


def get_linear_classifiers():
    logit_reg = LogisticRegression(random_state=42)
    return [logit_reg], ["Logistic Regression"]


def get_sgd_classifier():
    sgd = SGDClassifier(random_state=42)
    return [sgd], ["Stochastic Gradient Descent"]


def get_warm_start_classifiers():
    rf = RandomForestClassifier(n_estimators=51,random_state=42,warm_start=True)
    grad = GradientBoostingClassifier(random_state=42,warm_start=True)
    lr = LogisticRegression(solver='sag',warm_start=True,random_state=42)
    return [rf, lr, grad], ['Random Forest', 'Logistic Regression', 'GradientBoost']


def get_random_forest_classifier():
    rf = RandomForestClassifier(n_estimators=51, random_state=42)
    return [rf], ['Random Forest']


def get_ada_boost_classifier():
    ada = AdaBoostClassifier(random_state=42)
    return [ada],['AdaBoost']


def get_grad_boost_classifier():
    grad = GradientBoostingClassifier(random_state=42, warm_start=True)
    return [grad], ['GradientBoost']


def get_all_classifiers():
    classifier_list, classifier_name_list = get_ensemble_tree_classifiers()
    temp_classifier_list, temp_classifier_name_list = get_discriminative_classifiers()
    classifier_list.extend(temp_classifier_list)
    classifier_name_list.extend(temp_classifier_name_list)
    temp_classifier_list, temp_classifier_name_list = get_linear_classifiers()
    classifier_list.extend(temp_classifier_list)
    classifier_name_list.extend(temp_classifier_name_list)
    return classifier_list, classifier_name_list
