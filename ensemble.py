import numpy as np
from decision_tree import predict as predict_dt
from neural_network import get_predictions as predict_nn
from neural_network import load_data_for_nn, NeuralNetwork
from naive_bayes import make_model_and_predict as predict_nb
from logistic_regression import predict_logistic_regression as predict_lr, create_logistic_regression
from preprocessing import load_data_uoft_kaggle_separate_test
from typing import Any
import pickle

X_train, X_valid, X_test, y_train, y_valid, y_test, X_uoft, y_uoft, _ = load_data_uoft_kaggle_separate_test()

train_loader, valid_loader, uoft_test_loader = load_data_for_nn(X_train, X_valid, X_uoft, y_train, y_valid, y_uoft)
train_loader, valid_loader, test_loader = load_data_for_nn(X_train, X_valid, X_test, y_train, y_valid, y_test)

def predict_ensemble(X, X_for_nn) -> Any:
    # perform weighted voting ensemble

    # validation accuracies for each model in order of:
    # Decision Tree, Neural Network (SGD), Naive Bayes, Logistic Regression
    validation_accuracies = [0.8848, 0.9498, 0.9711, 0.9738]
    total = sum(validation_accuracies)
    weights = [accuracy / total for accuracy in validation_accuracies]

    predictions_dt = predict_dt(X)

    print("finished dt")

    with open('data/nn.pkl', 'rb') as f:
        nn = pickle.load(f)

    predictions_nn = predict_nn(nn, X_for_nn)

    print("finished nn")

    predictions_nb = predict_nb(X_train, y_train, X)

    print("finished nb")

    lr =create_logistic_regression(X_train, y_train)
    predictions_lr = predict_lr(lr, X)

    print("finished lr")

    # take the mode for each index of each prediction (i.e. a vote)
    # predictions = (predictions_dt + predictions_nn + predictions_nb) // 3
    # predictions = np.array([np.argmax(np.bincount([predictions_dt[i], predictions_nn[i], predictions_nb[i]])) for i in range(len(predictions_dt))])
    weighted_votes = np.array([np.argmax(np.bincount([predictions_dt[i], predictions_nn[i], predictions_nb[i], predictions_lr[i]], 
                               weights=weights)) for i in range(len(predictions_dt))])
    return weighted_votes

def report_accuracy(predictions, y) -> None:
    accuracy = (predictions == y).mean()
    print(f"Accuracy: {accuracy}")

if __name__ == "__main__":
    print("train")
    predictions = predict_ensemble(X_train, train_loader)
    report_accuracy(predictions, y_train)
    print("valid")
    predictions = predict_ensemble(X_valid, valid_loader)
    report_accuracy(predictions, y_valid)
    print("test")
    predictions = predict_ensemble(X_test, test_loader)
    report_accuracy(predictions, y_test)
    print("uoft")
    predictions = predict_ensemble(X_uoft, uoft_test_loader)
    report_accuracy(predictions, y_uoft)