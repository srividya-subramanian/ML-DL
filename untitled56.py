# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 05:41:43 2025

@author: srivi
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import DataLoader, TensorDataset

class SimpleNN(nn.Module):
    def __init__(self, input_size=10, hidden_size=32, output_size=1):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.sigmoid(x)

def generate_synthetic_data(num_clients=5, samples_per_client=100, input_size=10):
    clients_data = []
    
    for _ in range(num_clients):
        X = torch.randn(samples_per_client, input_size)  # Random input features
        y = torch.randint(0, 2, (samples_per_client, 1), dtype=torch.float)  # Binary labels
        clients_data.append((X, y))
    
    return clients_data

# Create synthetic data for 5 clients
clients_data = generate_synthetic_data()

def local_train(model, data, epochs=5, lr=0.01, batch_size=32):
    """
    Trains a model locally on client data.
    """
    X_train, y_train = data
    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    local_model = SimpleNN()
    local_model.load_state_dict(model.state_dict())  # Copy global model
    optimizer = optim.SGD(local_model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    local_model.train()
    for epoch in range(epochs):
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            output = local_model(X_batch)#.squeeze()
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

    return local_model.state_dict()


def federated_averaging(global_model, client_updates):
    """
    Aggregates model updates from all clients using Federated Averaging.
    """
    new_state_dict = {}
    for key in global_model.state_dict().keys():
        new_state_dict[key] = torch.stack([client_updates[i][key] for i in range(len(client_updates))]).mean(dim=0)

    global_model.load_state_dict(new_state_dict)


# Initialize global model
global_model = SimpleNN()
rounds = 10  # Number of federated learning rounds

for round in range(rounds):
    client_updates = []
    
    for client_id in range(len(clients_data)):
        local_model_state = local_train(global_model, clients_data[client_id])
        client_updates.append(local_model_state)
    
    # Aggregate updates using FedAvg
    federated_averaging(global_model, client_updates)

    print(f"Round {round+1} complete.")


def evaluate_model(model, test_data):
    model.eval()
    X_test, y_test = test_data
    with torch.no_grad():
        predictions = model(X_test)#.squeeze()
        predicted_labels = (predictions > 0.5).float()
        accuracy = (predicted_labels == y_test).float().mean()
    return accuracy.item()

# Generate new test data
test_data = generate_synthetic_data(num_clients=1, samples_per_client=200)[0]
test_accuracy = evaluate_model(global_model, test_data)

print(f"Final Model Accuracy on Test Data: {test_accuracy * 100:.2f}%")






























