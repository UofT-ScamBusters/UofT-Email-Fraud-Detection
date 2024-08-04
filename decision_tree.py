from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from preprocessing import load_data

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
        None
    """
    tree = create_decision_tree(X_train, y_train)
    report_training_accuracy(tree, X_train, y_train)
    validate_decision_tree(tree, X_valid, y_valid)

if __name__ == "__main__":
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_data()

    run_train_valid(X_train, y_train, X_valid, y_valid)

    # TODO: if you want to batch test hyperparameters, uncomment the following line and input ur own parameters.
    try_hyperparameters(X_train, y_train, X_valid, y_valid, ['gini', 'entropy'], ['best'], [16, 18, 25, 30, 35]) 

    # Test Decision Tree (when done tuning hyperparameters)
    # test_decision_tree(tree, X_test, y_test)

