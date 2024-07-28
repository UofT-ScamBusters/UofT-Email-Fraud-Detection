import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

from preprocessing import load_data as preprocess_load_data

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

def train(model, criterion, optimizer, train_loader, valid_loader, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        for data, labels in train_loader:
            data = Variable(data)
            labels = Variable(labels).unsqueeze(1)  # Ensure labels have shape [batch_size, 1]
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            valid_loss = 0
            correct = 0
            total = 0
            for data, labels in valid_loader:
                labels = labels.unsqueeze(1)  # Ensure labels have shape [batch_size, 1]
                outputs = model(data)
                valid_loss += criterion(outputs, labels).item()
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {valid_loss:.4f}, Accuracy: {accuracy:.4f}')

def evaluate(model, test_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for data, labels in test_loader:
            labels = labels.unsqueeze(1)  # Ensure labels have shape [batch_size, 1]
            outputs = model(data)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        print(f'Test Accuracy: {accuracy:.4f}')

def main():
    # Load preprocessed data
    X_train, X_valid, X_test, y_train, y_valid, y_test = preprocess_load_data()
    
    # Convert data to PyTorch tensors
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    valid_dataset = TensorDataset(torch.FloatTensor(X_valid), torch.FloatTensor(y_valid))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=64, shuffle=False)
    
    input_size = X_train.shape[1]
    hidden_size = 100
    output_size = 1
    num_epochs = 20
    learning_rate = 0.001
    
    model = NeuralNetwork(input_size, hidden_size, output_size)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train(model, criterion, optimizer, train_loader, valid_loader, num_epochs)
    
    evaluate(model, DataLoader(dataset=test_dataset, batch_size=64, shuffle=False))

if __name__ == "__main__":
    main()