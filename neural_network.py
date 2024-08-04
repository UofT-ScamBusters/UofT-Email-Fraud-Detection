import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from preprocessing import load_data

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
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {accuracy:.4f}')

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
    for data, labels in test_loader:
        labels = labels.unsqueeze(1)  # Ensure labels have shape [batch_size, 1]
        outputs = model(data)
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    return accuracy

if __name__ == "__main__":
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_data()
    train_loader, valid_loader, test_loader = load_data_for_nn(X_train, X_valid, X_test, y_train, y_valid, y_test)
    
    input_size = X_train.shape[1]
    output_size = 1 # for binary classification, output_size should be 1
    
    model = NeuralNetwork(input_size, HIDDEN_SIZE, output_size)
    criterion = nn.BCELoss() # binary cross entropy loss
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=LAMBDA) # stochastic gradient descent
    
    train(model, criterion, optimizer, train_loader, valid_loader, NUM_EPOCHS)
    
    # TODO: Test Decision Tree (when done tuning hyperparameters)
    # final_accuracy = evaluate(model, test_loader)
    # print(f'Final Test Accuracy: {final_accuracy:.4f}')