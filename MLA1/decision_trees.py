import numpy as np
from sklearn.tree import DecisionTreeClassifier
from preprocessing import get_spam_xy, get_phishing_xy, test_attribute


def get_phishing_dt():
    model = DecisionTreeClassifier()
    data_x, data_y = get_phishing_xy()
    path = model.cost_complexity_pruning_path(data_x, data_y)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    ccp_alphas = ccp_alphas[:-25]

    model.fit(data_x, data_y)
    importance = model.feature_importances_
    dictionary = {}
    for i, v in enumerate(importance):
        dictionary[i] = v
    sorted_features = {k: j for k, j in sorted(dictionary.items(), key=lambda item: item[1])}

    non_important = []
    for feature in sorted_features:
        this_feature = sorted_features[feature]
        if this_feature == 0:
            non_important.append(feature)
    data_x = np.delete(data_x, non_important, 1)

    print("DT Phishing Score by Max Depth")
    test_attribute(data_x, data_y, DecisionTreeClassifier, "max_depth", range_dict=[1, 100, 1],
                   x_label="Depth", y_label="score", title="Phishing Score by Max Depth",
                   file_name="Phishing/DT/max_depth")
    print("DT Phishing Score by Max Features")
    test_attribute(data_x, data_y, DecisionTreeClassifier, "max_features", range_dict=[1, data_x.shape[1], 1],
                   x_label="Features", y_label="score", title="Phishing Score by Max Features",
                   file_name="Phishing/DT/max_features")
    print("DT Phishing Score by Alpha")
    test_attribute(data_x, data_y, DecisionTreeClassifier, "ccp_alpha", dictionary=ccp_alphas,
                   x_label="Alphas", y_label="score", title="Phishing Score by Alpha",
                   file_name="Phishing/DT/alphas")


def get_spam_dt():
    model = DecisionTreeClassifier()
    data_x, data_y = get_spam_xy()
    path = model.cost_complexity_pruning_path(data_x, data_y)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    ccp_alphas = ccp_alphas[:-15]

    model = DecisionTreeClassifier(ccp_alpha=0.0006)
    model.fit(data_x, data_y)
    importance = model.feature_importances_
    dictionary = {}
    for i, v in enumerate(importance):
        dictionary[i] = v
    sorted_features = {k: j for k, j in sorted(dictionary.items(), key=lambda item: item[1])}

    non_important = []
    for feature in sorted_features:
        this_feature = sorted_features[feature]
        if this_feature == 0:
            non_important.append(feature)
    data_x = np.delete(data_x, non_important, 1)

    print("DT Spam Score by Max Depth")
    test_attribute(data_x, data_y, DecisionTreeClassifier, "max_depth", range_dict=[1, 100, 1],
                   x_label="Depth", y_label="score", title="Spam Score by Max Depth",
                   file_name="Spam/DT/max_depth")

    print("DT Spam Score by Max Features")
    test_attribute(data_x, data_y, DecisionTreeClassifier, "max_features", range_dict=[1, data_x.shape[1], 1],
                   x_label="Features", y_label="score", title="Spam Score by Max Features",
                   file_name="Spam/DT/max_features")

    print("DT Spam Score by Alpha")
    test_attribute(data_x, data_y, DecisionTreeClassifier, "ccp_alpha", dictionary=ccp_alphas,
                   x_label="Alpha", y_label="score", title="Spam Score by Alpha",
                   file_name="Spam/DT/alphas")
