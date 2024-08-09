import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    roc_curve,
    auc,
    ConfusionMatrixDisplay,
)
import pickle as pickle

from preprocessing import load_data, load_data_uoft_kaggle_merged_test, load_data_uoft_kaggle_separate_test

def create_logistic_regression(X_train, y_train):
    """
    Creates a Logistic Regression classifier using the given training data.
    Parameters:
        X_train: The input features for training.
        y_train: The target values for training.
    Returns:
        LogisticRegression: The trained Logistic Regression classifier.
    """
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def predict_logistic_regression(model, X_test):
    """
    Predicts the target values using the Logistic Regression classifier.
    Args:
        model (LogisticRegression): The trained Logistic Regression model.
        X_test (array-like): The input features for testing.
    Returns:
       array-like: The predicted target values
    """
    test_pred = model.predict(X_test)
    return test_pred

def report_training_accuracy(model, X_train, y_train) -> float:
    """
    Calculate and print the training accuracy of the Logistic Regression classifier.
    Args:
        model (LogisticRegression): The trained Logistic Regression model.
        X_train: The input features for training.
        y_train (array-like): The target values for training.
    Returns:
       float: The training accuracy
    """
    train_pred = model.predict(X_train)
    accuracy = accuracy_score(y_train, train_pred)
    print(f"Training Accuracy: {accuracy}")
    return accuracy

def validate_logistic_regression(model, X_valid, y_valid) -> float:
    """
    Validates Logistic Regression classifier and prints the validation accuracy.
    Args:
        model (LogisticRegression): The trained Logistic Regression model.
        X_valid (array-like): The input features for validation.
        y_valid (array-like): The target values for validation.
    Returns:
       float: The validation accuracy
    """
    valid_pred = model.predict(X_valid)
    accuracy = accuracy_score(y_valid, valid_pred)
    print(f"Validation Accuracy: {accuracy}")
    return accuracy

def evaluate_logistic_regression(model, X_test, y_test):
    """
    Evaluates the Logistic Regression classifier and prints the accuracy.
    Args:
        model (LogisticRegression): The trained Logistic Regression model.
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
    Plots the confusion matrix for the Logistic Regression classifier.
    Args:
        model (LogisticRegression): The trained Logistic Regression model.
        X_test (array-like): The input features for testing.
        y_test (array-like): The target values for testing.
    """
    test_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, test_pred)
    ConfusionMatrixDisplay(cm, display_labels=['Phishing', 'Safe']).plot(cmap="OrRd")
    plt.title("Confusion Matrix for Logistic Regression")
    plt.savefig("visualizations/confusion_matrix_lr.png")
    plt.show()

def plot_roc_curve(model, X_test, y_test):
    """
    Plots the ROC curve for the Logistic Regression classifier.
    Args:
        model (LogisticRegression): The trained Logistic Regression model.
        X_test (array-like): The input features for testing.
        y_test (array-like): The target values for testing.
    """
    true_labels = y_test
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(true_labels, y_prob)
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
    plt.savefig("visualizations/roc_curve_lr.png")
    plt.show()

def save_model(model):
    with open('data/lr.pkl', 'wb') as file:
        pickle.dump(model, file)

if __name__ == "__main__":
    # test data is kaggle data and uoft data separate
    X_train, X_valid, X_test, y_train, y_valid, y_test, X_uoft, y_uoft, _ = load_data_uoft_kaggle_separate_test()

    model = create_logistic_regression(X_train, y_train)
    report_training_accuracy(model, X_train, y_train)
    validate_logistic_regression(model, X_valid, y_valid)
    evaluate_logistic_regression(model, X_test, y_test)
    plot_confusion_matrix(model, X_test, y_test)
    plot_roc_curve(model, X_test, y_test)
    save_model()
