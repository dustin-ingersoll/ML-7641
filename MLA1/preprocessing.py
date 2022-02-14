import re
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix


# Reads dataset_spam.csv and returns x (training) and y (labels) data.
def get_spam_xy(verbose=False, return_links=False):
    # read csv into Dataframe
    spam_ds = pd.read_csv("data/dataset_spam.csv", low_memory=False)
    # rip out data
    spam_x = spam_ds.values[0:, 1]
    links_x = extract_links(spam_x)
    # rip out labels
    spam_y = spam_ds.values[0:, 0]
    spam_y = spam_y.astype('int')
    if verbose:
        print("SPAM: ", spam_x.shape)

    if return_links:

        return links_x, spam_y

    # convert x data to sparse matrix for nlp
    spam_x = nlp_processing(spam_x)
    x_train, x_test, y_train, y_test = train_test_split(spam_x, spam_y)
    return spam_x, spam_y


def get_phishing_xy(verbose=False):
    # read csv into Dataframe
    phishing_ds = pd.read_csv("data/dataset_phishing.csv")
    # rip out data
    phishing_x = phishing_ds.values[0:, 1:51]
    # rip out labels
    phishing_y = phishing_ds.values[0:, -1]
    phishing_x = np.delete(phishing_x, -2, 1)
    # convert label strings into ints.
    # 0 = safe, 1 = phishing
    phishing_y[phishing_y == 'legitimate'] = 0
    phishing_y[phishing_y == 'phishing'] = 1
    phishing_y = phishing_y.astype('int')

    check_columns(phishing_x, verbose)

    if verbose:
        print("PHISHING: ", phishing_x.shape)

    return phishing_x, phishing_y


def check_columns(data, verbose=False):
    # convert to T/F cell is the same as first value
    col_check = data == data[0, :]
    # check if all cells down a column equate to true (all column is the same value)
    col_check = np.all(col_check == col_check[0, :], axis=0)
    # discard these columns as irrelevant, contain no differentiating information.
    del_cols = np.where(col_check)[0]
    data = np.delete(data, del_cols, 1)
    if verbose:
        print("$--- check_columns() ---$")
        print(len(del_cols), "columns were removed for irrelevance.")

    return data, del_cols


# returns a sparse matrix of word counts for the data.
def nlp_processing(data):
    vectorizer = CountVectorizer(stop_words='english', strip_accents='ascii', max_features=250)
    count_array = vectorizer.fit_transform(data).toarray()
    return count_array


def extract_links(data):
    new_data = data.copy()
    regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    for i, cell in enumerate(new_data):
        new_data[i] = np.array(re.findall(regex, cell))
        for j, link in enumerate(new_data[i]):
            new_data[i][j] = re.sub('<[^<]+?>|>[a-zA-z]*', '', link)
    return new_data


def test_attribute(data_x, data_y, model, attribute, rounds=10, range_dict=None, dictionary=None, add_attributes=None,
                y_label="Y", x_label="X", title="Title", file_name=None):
    if range_dict:
        range_len = int(range_dict[1]/range_dict[2])-1
        train_results = np.zeros((range_len, rounds))
        test_results = np.zeros((range_len, rounds))
        time_results = np.zeros((range_len, rounds))
    elif len(dictionary):
        train_results = np.zeros((len(dictionary), rounds))
        test_results = np.zeros((len(dictionary), rounds))
        time_results = np.zeros((len(dictionary), rounds))
    result_counts = []
    for round in range(rounds):
        print("ROUND: ", round)
        train_x, test_x, train_y, test_y = train_test_split(data_x, data_y)

        clfs = []
        counts = []
        times = []
        if range_dict:
            start = range_dict[0]
            stop = range_dict[1]
            step = range_dict[2]
            for i in range(start, stop, step):
                clf = model()
                setattr(clf, attribute, i)
                if add_attributes:
                    for attribute in add_attributes:
                        setattr(clf, attribute, add_attributes[attribute])
                start_time = time.time()
                clf.fit(train_x, train_y)
                end_time = time.time()
                times.append(end_time - start_time)
                clfs.append(clf)
                counts.append(i)
        elif len(dictionary):
            for i in dictionary:
                clf = model()
                setattr(clf, attribute, i)
                if add_attributes:
                    for attribute in add_attributes:
                        setattr(clf, attribute, add_attributes[attribute])
                start_time = time.time()
                clf.fit(train_x, train_y)
                end_time = time.time()
                times.append(end_time - start_time)
                clfs.append(clf)
                counts.append(i)
        result_counts = counts
        train_scores = np.array([clf.score(train_x, train_y) for clf in clfs])
        test_scores = np.array([clf.score(test_x, test_y) for clf in clfs])
        train_results[:, round] = train_scores
        test_results[:, round] = test_scores
        time_results[:, round] = times

    train_scores = train_results.mean(axis=1)
    test_scores = test_results.mean(axis=1)
    time_scores = time_results.mean(axis=1)
    # time_scores = np.around(time_scores, decimals=2)

    fig, ax = plt.subplots()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ln1 = ax.plot(result_counts, train_scores, label="train")
    ln2 = ax.plot(result_counts, test_scores, label="test")
    ax2 = ax.twinx()
    ax2.set_ylabel("Fit Time (s)")
    ln3 = ax2.plot(result_counts, time_scores, label="Fit Time", color="grey", linewidth=0.5, linestyle="-.")
    lns = ln1 + ln2 + ln3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs)
    if file_name:
        plt.savefig("graphs/" + file_name, bbox_inches='tight')
    else:
        plt.show()
    plt.cla()



