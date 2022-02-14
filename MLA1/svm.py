from sklearn import svm
from preprocessing import get_spam_xy, get_phishing_xy, test_attribute
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt



def get_spam_svm():
    data_x, data_y = get_spam_xy()

    print("SVM Spam Score by Kernel")
    test_attribute(data_x, data_y, svm.SVC, "kernel", dictionary=['linear', 'poly', 'rbf', 'sigmoid'],
                   x_label="Kernel", y_label="score", title="Spam Score by Kernel",
                   file_name="Spam/SVM/kernel")
    print("SVM Spam Score by C")
    test_attribute(data_x, data_y, svm.SVC, "C", range_dict=[1, 30, 1],
                   x_label="C", y_label="score", title="Spam Score by C",
                   file_name="Spam/SVM/c")
    print("SVM Spam Score by Gamma")


def get_phishing_svm():
    data_x, data_y = get_phishing_xy()

    print("SVM Phishing Score by Kernel")
    test_attribute(data_x, data_y, svm.SVC, "kernel", dictionary=['linear', 'poly', 'rbf', 'sigmoid'],
                   x_label="Kernel", y_label="score", title="Phishing Score by Kernel",
                   file_name="Phishing/SVM/kernel")
    print("SVM Phishing Score by C")
    test_attribute(data_x, data_y, svm.SVC, "C", range_dict=[1, 10, 1],
                   x_label="C", y_label="score", title="Phishing Score by C",
                   file_name="Phishing/SVM/c")


