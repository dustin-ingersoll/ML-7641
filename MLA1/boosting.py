from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from preprocessing import get_spam_xy, get_phishing_xy, test_attribute
import matplotlib.pyplot as plt


def get_spam_boost():
    data_x, data_y = get_spam_xy()

    print("Boosting Spam Score Optimum")
    test_attribute(data_x, data_y, AdaBoostClassifier, "n_estimators", range_dict=[1, 50, 1],
                   x_label="Estimators", y_label="score", title="Spam Score by Estimator Number",
                   file_name="Spam/Boosting/estimators")
    print("Boosting Spam Score by Learning Rate")
    test_attribute(data_x, data_y, AdaBoostClassifier, "learning_rate", range_dict=[1, 3, 1],
                   x_label="Learning Rate", y_label="score", title="Spam Score by Learning Rate",
                   file_name="Spam/Boosting/learning_rate")


def get_phishing_boost():
    data_x, data_y = get_phishing_xy()

    print("Boosting Phishing Score by Estimator Number")
    test_attribute(data_x, data_y, AdaBoostClassifier, "n_estimators", range_dict=[1, 50, 1],
                   x_label="Estimators", y_label="score", title="Phishing Score by Estimator Number",
                   file_name="Phishing/Boosting/estimators")
    print("Boosting Phishing Score by Learning Rate")
    test_attribute(data_x, data_y, AdaBoostClassifier, "learning_rate", range_dict=[1, 3, 1],
                   x_label="Learning Rate", y_label="score", title="Phishing Score by Learning Rate",
                   file_name="Phishing/Boosting/learning_rate")

