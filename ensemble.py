import numpy as np
from decision_tree import predict as predict_dt
from neural_network import get_predictions as predict_nn
from neural_network import load_data_for_nn, NeuralNetwork
from naive_bayes import make_model_and_predict as predict_nb
from preprocessing import load_data_uoft_kaggle_separate_test
from typing import Any
import pickle

X_train, X_valid, X_test, y_train, y_valid, y_test, X_uoft, y_uoft, _ = load_data_uoft_kaggle_separate_test()

train_loader, valid_loader, uoft_test_loader = load_data_for_nn(X_train, X_valid, X_uoft, y_train, y_valid, y_uoft)
train_loader, valid_loader, test_loader = load_data_for_nn(X_train, X_valid, X_test, y_train, y_valid, y_test)

def predict_ensemble(X, X_for_nn) -> Any:
    predictions_dt = predict_dt(X)

    print("finished dt")

    with open('data/nn.pkl', 'rb') as f:
        nn = pickle.load(f)

    predictions_nn = predict_nn(nn, X_for_nn)

    print("finished nn")

    predictions_nb = predict_nb(X_train, y_train, X)

    print("finished nb")

    # take the mode for each index of each prediction (i.e. a vote)
    # predictions = (predictions_dt + predictions_nn + predictions_nb) // 3
    predictions = np.array([np.argmax(np.bincount([predictions_dt[i], predictions_nn[i], predictions_nb[i]])) for i in range(len(predictions_dt))])
    return predictions

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