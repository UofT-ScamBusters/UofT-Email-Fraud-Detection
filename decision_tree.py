from typing import Any
from sklearn.tree import DecisionTreeClassifier, plot_tree 
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix, 
    ConfusionMatrixDisplay, 
    f1_score,
    roc_curve, 
    auc
)
from preprocessing import load_data, load_data_uoft_kaggle_merged_test, load_data_uoft_kaggle_separate_test
import pickle
import matplotlib.pyplot as plt
import numpy as np

# Tree Hyperparameters
CRITERION = 'gini' # ['gini', 'entropy']
SPLITTER = 'best' # ['best', 'random']
MAX_TREE_DEPTH = 25 # any reasonable number
RANDOM_STATE = 30

def create_decision_tree(X_train, y_train) -> DecisionTreeClassifier:
    """
    Creates a decision tree classifier using the given training data.
    Parameters:
        X_train: The input features for training.
        y_train: The target values for training.
    Returns:
        DecisionTreeClassifier: The trained decision tree classifier.
    """

    tree = DecisionTreeClassifier(criterion=CRITERION, 
                                  splitter=SPLITTER, 
                                  max_depth=MAX_TREE_DEPTH, 
                                  random_state=RANDOM_STATE)
    tree.fit(X_train, y_train)
    return tree

def report_training_accuracy(tree, X_train, y_train) -> float:
    """
    Calculate and print the training accuracy of the decision tree.
    Args:
        tree (DecisionTreeClassifier): The trained decision tree model.
        X_train (array-like): The input features for training.
        y_train (array-like): The target values for training.
    Returns:
       float: The training accuracy
    """
    train_pred = tree.predict(X_train)
    accuracy = accuracy_score(y_train, train_pred)
    print(f"Training Accuracy: {accuracy}")
    return accuracy


def validate_decision_tree(tree, X_valid, y_valid) -> float:
    """
    Validates decision tree and prints the validation accuracy.
    Args:
        tree (DecisionTreeClassifier): The trained decision tree model.
        X_valid (array-like): The input features for validation.
        y_valid (array-like): The target values for validation.
    Returns:
       float: The validation accuracy
    """
    valid_pred = tree.predict(X_valid)
    accuracy = accuracy_score(y_valid, valid_pred)
    print(f"Validation Accuracy: {accuracy}")
    return accuracy


def test_decision_tree(tree, X_test, y_test) -> float:
    """
    Tests decision tree and prints the test accuracy.
    Args:
        tree (DecisionTreeClassifier): The trained decision tree model.
        X_test (array-like): The input features for testing.
        y_test (array-like): The target values for testing.
    Returns:
       float: The test accuracy
    """
    test_pred = tree.predict(X_test)
    accuracy = accuracy_score(y_test, test_pred)
    print(f"Test Accuracy: {accuracy}")
    return accuracy


def try_hyperparameters(X_train, y_train, X_valid, y_valid, criterions, splitters, max_depths) -> None:
    """
    Tries different hyperparameters for the decision tree and prints the validation accuracy.
    Reports the best hyperparameters.
    Args:
        X_train (array-like): The input features for training.
        y_train (array-like): The target values for training.
        X_valid (array-like): The input features for validation.
        y_valid (array-like): The target values for validation.
        criterions (list): The list of criteria to use for the decision tree.
        splitters (list): The list of splitters to use for the decision tree.
        max_depths (list): The list of maximum depths to use for the decision tree.
    Returns:
        None
    """
    max_so_far = {"accuracy": 0, "criterion": None, "splitter": None, "max_depth": None}

    for criterion in criterions:
        for splitter in splitters:
            for max_depth in max_depths:
                print(f"Trying criterion: {criterion}, splitter: {splitter}, max_depth: {max_depth}")
                tree = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth, random_state=RANDOM_STATE)
                tree.fit(X_train, y_train)
                report_training_accuracy(tree, X_train, y_train)
                valid_acc = validate_decision_tree(tree, X_valid, y_valid)
                if valid_acc > max_so_far["accuracy"]:
                    max_so_far["accuracy"] = valid_acc
                    max_so_far["criterion"] = criterion
                    max_so_far["splitter"] = splitter
                    max_so_far["max_depth"] = max_depth
    
    print(f"Best hyperparameters: {max_so_far}")

def run_train_valid(X_train, y_train, X_valid, y_valid) -> None:
    """
    Trains and validates the decision tree for a single run.

    Returns:
        DecisionTreeClassifier: The trained decision tree model.
    """
    tree = create_decision_tree(X_train, y_train)
    report_training_accuracy(tree, X_train, y_train)
    validate_decision_tree(tree, X_valid, y_valid)
    return tree

def save_model(X_train, y_train, filename: str) -> None:
    """
    Saves the decision tree model to a pickle file to reduce computation time later.

    Returns:
        None    
    """         
    tree = create_decision_tree(X_train, y_train)

    with open(filename, 'wb') as f:
        pickle.dump(tree, f)

def load_model(filename: str) -> DecisionTreeClassifier:
    """
    Loads the decision tree model from a pickle file.

    Returns:
        DecisionTreeClassifier: The loaded decision tree model.
    """
    with open(filename, 'rb') as f:
        tree = pickle.load(f)
    return tree

def predict(X) -> Any:
    """
    Predicts the target values for the input features.

    Returns:
        None
    """
    with open('data/decision_tree.pkl', 'rb') as f:
        tree = pickle.load(f)

    predictions = tree.predict(X)
    return predictions

def plot_confusion_matrix(y, predictions) -> None:
    """
    Creates a confusion matrix for the decision tree model.

    Returns:
        None
    """
    clf = confusion_matrix(y, predictions)
    cx = ConfusionMatrixDisplay(clf, display_labels=['Phishing', 'Safe']).plot(cmap="RdPu")
    plt.title("Confusion Matrix for Decision Tree Kaggle Data")
    plt.savefig("visualizations/confusion_matrix_dt.png")
    plt.show()

def plot_feature_importance(feature_names, tree, top_n=10) -> None:
    """
    Plots the feature importance of the decision tree model.

    Returns:
        None
    """
    importances = tree.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_indices = indices[:top_n]

    plt.figure()
    plt.title("Feature Importances")
    plt.bar(range(top_n), importances[top_indices], align="center")
    plt.xticks(range(top_n), np.array(feature_names)[top_indices], rotation=90)
    plt.tight_layout()
    plt.savefig("visualizations/feature_importance_dt.png")
    plt.show()


def plot_decision_tree(tree, feature_names) -> None:
    """
    Plots the decision tree model.

    Returns:
        None
    """
    plt.figure(figsize=(20, 20))
    plot_tree(tree, filled=True, feature_names=feature_names, class_names=['Phishing', 'Safe'])
    plt.title("Decision Tree")
    plt.savefig("visualizations/decision_tree.png")
    plt.show()

def plot_roc_curve(tree, X_test, y_test) -> None:
    """
    Plots the ROC curve for the decision tree model.

    Returns:
        None
    """
    true_labels = y_test
    predictions = predict(X_test)
    fpr, tpr, _ = roc_curve(true_labels, predictions)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve with Kaggle Data')
    plt.legend(loc="lower right")
    plt.savefig("visualizations/roc_curve_dt.png")
    plt.show()

if __name__ == "__main__":
    X_train, X_valid, X_test, y_train, y_valid, y_test, X_uoft, y_uoft, feature_names = load_data_uoft_kaggle_separate_test()

    tree = load_model('data/decision_tree.pkl')

    report_training_accuracy(tree, X_train, y_train)
    validate_decision_tree(tree, X_valid, y_valid)
    test_decision_tree(tree, X_test, y_test)
    print('uoft spam test')
    test_decision_tree(tree, X_uoft, y_uoft)

    # TODO: Train Decision Tree 
    tree = run_train_valid(X_train, y_train, X_valid, y_valid)
    plot_feature_importance(feature_names, tree)

    # TODO: NEED TO FIX THIS BECAUSE THE DECISION TREE IS WAY TOO BIG.. MAYBE JUST SHOW A PORTION OF IT?
    # plot_decision_tree(tree, feature_names)

    # TODO: if you want to batch test hyperparameters, uncomment the following line and input ur own parameters.
    # try_hyperparameters(X_train, y_train, X_valid, y_valid, ['gini', 'entropy'], ['best'], [16, 18, 25, 30, 35]) 

    # TODO: Test Decision Tree (when done tuning hyperparameters)
    # tree = create_decision_tree(X_test, y_test)
    # test_decision_tree(tree, X_test, y_test)
    # pred_dtr = tree.predict(X_test)
    # plot_confusion_matrix(y_test, pred_dtr)
    # plot_roc_curve(tree, X_test, y_test)
