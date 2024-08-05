import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from preprocessing import load_data, load_data_uoft_kaggle_merged_test, load_data_uoft_kaggle_separate_test
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import pickle

# Neural Network Hyperparameters
HIDDEN_SIZE = 50 # number of neurons in the hidden layer
NUM_EPOCHS = 20
LEARNING_RATE = 0.1
LAMBDA = 0.01 # regularization to prevent overfitting

def load_data_for_nn(X_train, X_valid, X_test, y_train, y_valid, y_test):
    """
    Load preprocessed data and convert it to PyTorch DataLoader objects.

    :param X_train: np.ndarray
    :param X_valid: np.ndarray
    :param X_test: np.ndarray
    :param y_train: np.ndarray
    :param y_valid: np.ndarray
    :param y_test: np.ndarray
    :return: (train_loader, valid_loader, test_loader)
        WHERE:
        train_loader: DataLoader for training data
        valid_loader: DataLoader for validation data
        test_loader: DataLoader for test data
    """
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    valid_dataset = TensorDataset(torch.FloatTensor(X_valid), torch.FloatTensor(y_valid))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
    
    return train_loader, valid_loader, test_loader

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize a class NeuralNetwork.
        
        :param input_size: int
        :param hidden_size: int
        :param output_size: int
        """
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU() # using ReLU as activator function
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid() # sigmoid to ensure output between 0 and 1
    
    def forward(self, x):
        """
        Return the forward pass of the neural network.
        
        :param x: torch.Tensor
        :return: torch.Tensor
        """
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

def train(model, criterion, optimizer, train_loader, valid_loader, num_epochs):
    """
    Train the neural network model. 

    :param model: NeuralNetwork
    :param criterion: loss function
    :param optimizer: optimization algorithm
    :param train_loader: DataLoader for training data
    :param valid_loader: DataLoader for validation data
    :param num_epochs: int
    :return: None
    """
    train_losses = []
    valid_accuracies = []
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for data, labels in train_loader:
            data = Variable(data)
            labels = Variable(labels).unsqueeze(1)  # Ensure labels have shape [batch_size, 1]
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        
        accuracy = evaluate(model, valid_loader)
        train_losses.append(train_loss/len(train_loader))
        valid_accuracies.append(accuracy)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss/len(train_loader):.4f}, Accuracy: {accuracy:.4f}')
    
    return train_losses, valid_accuracies

def evaluate(model, test_loader):
    """
    Evaluate the neural network model.

    :param model: NeuralNetwork
    :param test_loader: DataLoader for test data
    :return: accuracy
    """
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad(): 
        for data, labels in test_loader:
            labels = labels.unsqueeze(1)  # Ensure labels have shape [batch_size, 1]
            outputs = model(data)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    return accuracy

def get_predictions(model, data_loader):
    """
    Get the predicted labels from the neural network model.

    :param model: NeuralNetwork
    :param data_loader: DataLoader
    :return: np.ndarray of predictions (0 or 1)
    """
    model.eval()  

    predictions = []
    with torch.no_grad(): 
        for data, _ in data_loader:
            outputs = model(data)
            predicted = (outputs > 0.5).float() 
            predictions.extend(predicted.squeeze().cpu().numpy())  # Convert to numpy array

    int_array = np.array(predictions).astype(int)
    return np.array(int_array)

def plot_confusion_matrix(model, data_loader):
    """
    Plot the confusion matrix for the model's predictions on the data.

    :param model: NeuralNetwork
    :param data_loader: DataLoader
    :return: None
    """
    true_labels = []
    predictions = []

    model.eval()
    with torch.no_grad():
        for data, labels in data_loader:
            labels = labels.unsqueeze(1)
            outputs = model(data)
            predicted = (outputs > 0.5).float()
            true_labels.extend(labels.squeeze().cpu().numpy())
            predictions.extend(predicted.squeeze().cpu().numpy())
    cm = confusion_matrix(true_labels, predictions)        
    cm_display = ConfusionMatrixDisplay(cm, display_labels=['Phishing Email', 'Safe Email']).plot(cmap='Blues')
    plt.title("Confusion Matrix for Neural Network")
    plt.savefig("visualizations/confusion_matrix_nn.png")
    plt.show()

def plot_loss_accuracy(train_losses, valid_accuracies):
    """
    Plot the loss and accuracy over the epochs.

    :param train_losses: list of training losses
    :param valid_accuracies: list of validation accuracies
    :return: None
    """
    # Plot the training loss
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss over Epochs')
    plt.savefig("visualizations/training_loss_nn.png")
    plt.show()

    # Plot the validation accuracy
    plt.figure(figsize=(10,5))
    plt.plot(valid_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Validation Accuracy over Epochs')
    plt.savefig("visualizations/validation_accuracy_nn.png")
    plt.show()

def plot_roc_curve(model, data_loader):
    """
    Plot the ROC curve for the model's predictions on the data.

    :param model: NeuralNetwork
    :param data_loader: DataLoader
    :return: None
    """
    true_labels = []
    probabilities = []
    
    model.eval()
    with torch.no_grad():
        for data, labels in data_loader:
            labels = labels.unsqueeze(1)
            outputs = model(data)
            probabilities.extend(outputs.squeeze().cpu().numpy())
            true_labels.extend(labels.squeeze().cpu().numpy())
    
    fpr, tpr, _ = roc_curve(true_labels, probabilities)
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
    plt.savefig("visualizations/roc_curve_nn.png")
    plt.show()

def create_and_save_model(train_loader, valid_loader):
    model = NeuralNetwork(input_size=X_train.shape[1], hidden_size=HIDDEN_SIZE, output_size=1) # for binary classification, output_size should be 1
    criterion = nn.BCELoss() # binary cross entropy loss
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=LAMBDA) # stochastic gradient descent
    
    # train(model, criterion, optimizer, train_loader, valid_loader, NUM_EPOCHS)
    # Train the model and capture the loss and accuracy
    train_losses, valid_accuracies = train(model, criterion, optimizer, train_loader, valid_loader, NUM_EPOCHS)
    
    # Save model
    with open('data/nn.pkl', 'wb') as file:
        pickle.dump(model, file)

if __name__ == "__main__":
    # test data is just kaggle data
    # X_train, X_valid, X_test, y_train, y_valid, y_test = load_data()

    # test data is kaggle data and uoft data merged
    # X_train, X_valid, X_test, y_train, y_valid, y_test = load_data_uoft_kaggle_merged_test()

    # test data is kaggle data and uoft data separate
    X_train, X_valid, X_test, y_train, y_valid, y_test, X_uoft, y_uoft, _ = load_data_uoft_kaggle_separate_test()

    # below is using kaggle data as test data
    train_loader, valid_loader, test_loader = load_data_for_nn(X_train, X_valid, X_test, y_train, y_valid, y_test)
    
    # below is using uoft data as test data
    # train_loader, valid_loader, test_loader = load_data_for_nn(X_train, X_valid, X_uoft, y_train, y_valid, y_uoft)

    model = NeuralNetwork(input_size=X_train.shape[1], hidden_size=HIDDEN_SIZE, output_size=1) # for binary classification, output_size should be 1
    criterion = nn.BCELoss() # binary cross entropy loss
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=LAMBDA) # stochastic gradient descent
    
    # train(model, criterion, optimizer, train_loader, valid_loader, NUM_EPOCHS)
    # Train the model and capture the loss and accuracy
    train_losses, valid_accuracies = train(model, criterion, optimizer, train_loader, valid_loader, NUM_EPOCHS)

    # this is how you get predictions for data (ex. test data in this case)
    test_predictions = get_predictions(model, test_loader)
    
    # TODO: Test Decision Tree (when done tuning hyperparameters)
    test_accuracy = evaluate(model, test_loader)
    print(f'Final Test Accuracy: {test_accuracy:.4f}')

    plot_confusion_matrix(model, test_loader)
    plot_loss_accuracy(train_losses, valid_accuracies)
    plot_roc_curve(model, test_loader)