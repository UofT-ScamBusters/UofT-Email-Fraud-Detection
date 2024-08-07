from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    classification_report, 
    accuracy_score, 
    f1_score, 
    classification_report, 
    ConfusionMatrixDisplay, 
    confusion_matrix,
    roc_curve,
    auc
)
from preprocessing import load_data, load_data_uoft_kaggle_merged_test, load_data_uoft_kaggle_separate_test
import matplotlib.pyplot as plt

def create_naive_bayes(X_train, y_train):
    """
    Creates a Naive Bayes classifier using the given training data.
    Parameters:
        X_train: The input features for training.
        y_train: The target values for training.
    Returns:
        MultinomialNB: The trained Naive Bayes classifier.
    """
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model

def predict_naive_bayes(model, X_test):
    """
    Predicts the target values using the Naive Bayes classifier.
    Args:
        model (MultinomialNB): The trained Naive Bayes model.
        X_test (array-like): The input features for testing.
    Returns:
       array-like: The predicted target values
    """
    test_pred = model.predict(X_test)
    return test_pred

def report_training_accuracy(model, X_train, y_train) -> float:
    """
    Calculate and print the training accuracy of the Naive Bayes classifier.
    Args:
        model (MultinomialNB): The trained Naive Bayes model.
        X_train: The input features for training.
        y_train (array-like): The target values for training.
    Returns:
       float: The training accuracy
    """
    train_pred = model.predict(X_train)
    accuracy = accuracy_score(y_train, train_pred)
    print(f"Training Accuracy: {accuracy}")
    return accuracy

def validate_naive_bayes(model, X_valid, y_valid) -> float:
    """
    Validates Naive Bayes classifier and prints the validation accuracy.
    Args:
        model (MultinomialNB): The trained Naive Bayes model.
        X_valid (array-like): The input features for validation.
        y_valid (array-like): The target values for validation.
    Returns:
       float: The validation accuracy
    """
    valid_pred = model.predict(X_valid)
    accuracy = accuracy_score(y_valid, valid_pred)
    print(f"Validation Accuracy: {accuracy}")
    return accuracy

def evaluate_naive_bayes(model, X_test, y_test):
    """
    Evaluates the Naive Bayes classifier and prints the accuracy.
    Args:
        model (MultinomialNB): The trained Naive Bayes model.
        X_test (array-like): The input features for testing.
        y_test (array-like): The target values for testing.
    Returns:
       float: The test accuracy
    """
    test_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, test_pred)
    print(f"Test Accuracy: {accuracy}")
    report = classification_report(y_test, test_pred)
    print('Classification:')
    print(report)
    return accuracy

def plot_confusion_matrix(model, X_test, y_test):
    """
    Plots the confusion matrix for the Naive Bayes classifier.
    Args:
        model (MultinomialNB): The trained Naive Bayes model.
        X_test (array-like): The input features for testing.
        y_test (array-like): The target values for testing.
    """
    test_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, test_pred)
    cx = ConfusionMatrixDisplay(cm, display_labels=['Phishing', 'Safe']).plot(cmap="YlGn")
    plt.title("Confusion Matrix for Naive Bayes")
    plt.savefig("visualizations/confusion_matrix_nb.png")
    plt.show()

def plot_roc_curve(model, X_test, y_test):
    """
    Plots the ROC curve for the Naive Bayes classifier.
    Args:
        model (MultinomialNB): The trained Naive Bayes model.
        X_test (array-like): The input features for testing.
        y_test (array-like): The target values for testing.
    """
    true_labels = y_test
    predictions = predict_naive_bayes(model, X_test)
    fpr, tpr, _ = roc_curve(true_labels, predictions)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig("visualizations/roc_curve_nb.png")
    plt.show()

def make_model_and_predict(X_train, y_train, X_prediction):
    """
    Creates and trains a Naive Bayes classifier and makes predictions.
    (I made this function just so I don't have to create the model in ensemble.py)
    Args:
        X_train (array-like): The input features for training.
        y_train (array-like): The target values for training.
        X_prediction (array-like): The input features for prediction.
    Returns:
        array-like: The predicted target values
    """
    model = create_naive_bayes(X_train, y_train)
    return predict_naive_bayes(model, X_prediction)

if __name__ == "__main__":
    # test data is just kaggle data
    # X_train, X_valid, X_test, y_train, y_valid, y_test = load_data()

    # test data is kaggle data and uoft data merged
    # X_train, X_valid, X_test, y_train, y_valid, y_test = load_data_uoft_kaggle_merged_test()

    # test data is kaggle data and uoft data separate
    X_train, X_valid, X_test, y_train, y_valid, y_test, X_uoft, y_uoft, _ = load_data_uoft_kaggle_separate_test()

    model = create_naive_bayes(X_train, y_train)
    report_training_accuracy(model, X_train, y_train)
    validate_naive_bayes(model, X_valid, y_valid)
    evaluate_naive_bayes(model, X_test, y_test)
    plot_confusion_matrix(model, X_test, y_test)
    plot_roc_curve(model, X_test, y_test)
