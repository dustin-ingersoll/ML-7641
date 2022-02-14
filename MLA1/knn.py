from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from preprocessing import get_spam_xy, get_phishing_xy, test_attribute
import matplotlib.pyplot as plt


def get_spam_knn():
    data_x, data_y = get_spam_xy()

    print("KNN Spam Score by Neighbors")
    test_attribute(data_x, data_y, KNeighborsClassifier, "n_neighbors", range_dict=[1, 10, 1],
                   x_label="Neighbors", y_label="score", title="Spam Score by Neighbors",
                   file_name="Spam/KNN/neighbors")
    print("KNN Spam Score by Algorithm")
    test_attribute(data_x, data_y, KNeighborsClassifier, "algorithm", dictionary=['auto', 'ball_tree', 'kd_tree', 'brute'],
                   x_label="Algorithm", y_label="score", title="Spam Score by Algorithm",
                   file_name="Spam/KNN/algorithm")
    print("KNN Spam Score by P")
    test_attribute(data_x, data_y, KNeighborsClassifier, "p", range_dict=[1, 5, 1],
                   x_label="P", y_label="score", title="Spam Score by P",
                   file_name="Spam/KNN/p")


def get_phishing_knn():
    data_x, data_y = get_phishing_xy()

    print("KNN Phishing Score by Neighbors")
    test_attribute(data_x, data_y, KNeighborsClassifier, "n_neighbors", range_dict=[1, 10, 1],
                   x_label="Neighbors", y_label="score", title="Phishing Score by Neighbors",
                   file_name="Phishing/KNN/neighbors")
    print("KNN Phishing Score by Algorithm")
    test_attribute(data_x, data_y, KNeighborsClassifier, "algorithm", dictionary=['auto', 'ball_tree', 'kd_tree', 'brute'],
                   x_label="Neighbors", y_label="score", title="Phishing Score by Algorithm",
                   file_name="Phishing/KNN/algorithm")
    print("KNN Phishing Score by P")
    test_attribute(data_x, data_y, KNeighborsClassifier, "p", range_dict=[1, 5, 1],
                   x_label="Neighbors", y_label="score", title="Phishing Score by P",
                   file_name="Phishing/KNN/p")

