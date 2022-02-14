from sklearn.neural_network import MLPClassifier
from preprocessing import get_spam_xy, get_phishing_xy, test_attribute
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt



def get_spam_neural():
    data_x, data_y = get_spam_xy()

    test_attribute(data_x, data_y, MLPClassifier, "hidden_layer_sizes", range_dict=[20, 220, 20],
                   x_label="Hidden Layers", y_label="score", title="Spam Score by Hidden Layer Size",
                   file_name="Spam/NeuralNet/hidden_layers")
    print("NNET Spam Score by Activation")
    test_attribute(data_x, data_y, MLPClassifier, "activation", dictionary=["identity", 'logistic', 'tanh', 'relu'],
                   x_label="Activation", y_label="score", title="Spam Score by Activation",
                   file_name="Spam/NeuralNet/activation")
    print("NNET Spam Score by Learning Rate")
    test_attribute(data_x, data_y, MLPClassifier, "learning_rate", dictionary=["constant", 'invscaling', 'adaptive'],
                   x_label="Learning Rate", y_label="score", title="Spam Score by Learning Rate",
                   file_name="Spam/NeuralNet/learning_rate")


def get_phishing_neural():
    data_x, data_y = get_phishing_xy()

    print("NNET Phishing Score by Hidden Layers")
    test_attribute(data_x, data_y, MLPClassifier, "hidden_layer_sizes", range_dict=[20, 220, 20],
                   x_label="Hidden Layers", y_label="score", title="Phishing Score by Hidden Layers",
                   file_name="Phishing/NeuralNet/hidden_layers")
    print("NNET Phishing Score by Activation")
    test_attribute(data_x, data_y, MLPClassifier, "activation", dictionary=["identity", 'logistic', 'tanh', 'relu'],
                   x_label="Activation", y_label="score", title="Phishing Score by Activation",
                   file_name="Phishing/NeuralNet/activation")

