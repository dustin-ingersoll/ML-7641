from decision_trees import get_spam_dt, get_phishing_dt
from boosting import get_spam_boost, get_phishing_boost
from knn import get_spam_knn, get_phishing_knn
from neural_network import get_spam_neural, get_phishing_neural
from svm import get_spam_svm, get_phishing_svm


def run_spam_comparison():
    get_spam_dt()
    get_spam_boost()
    get_spam_knn()
    get_spam_neural()
    get_spam_svm()


def run_phishing_comparison():
    get_phishing_dt()
    get_phishing_boost()
    get_phishing_knn()
    get_phishing_neural()
    get_phishing_svm()

run_spam_comparison()
run_phishing_comparison()