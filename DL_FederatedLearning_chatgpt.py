# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 05:36:18 2025

@author: srivi


Federated Learning: An Overview 🚀
Federated Learning (FL) is a decentralized approach to machine learning where 
models are trained locally on edge devices (such as smartphones, IoT devices, 
or local servers) without sharing raw data with a central server. Instead, 
only model updates (gradients or weights) are sent to a central aggregator, 
ensuring privacy and security while still enabling collaborative learning.

🔹 Why Use Federated Learning?
Privacy-Preserving 🛡️ → No need to transmit raw user data to a central server.
Bandwidth-Efficient 📶 → Only model updates are shared, reducing data transfer.
Personalization 🎯 → Models can adapt to local user data while benefiting from 
global updates.
Scalability 📈 → Can train models across millions of devices without centralized 
data storage.

🛠️ How Federated Learning Works?
Model Initialization
A global model is sent to multiple clients (devices or edge servers).

Local Training
Each client trains the model on its own private dataset for a few epochs.

Model Update Sharing
Instead of sending data, clients send only model weight updates (gradients) 
back to the central server.

Aggregation
The central server aggregates updates from all clients (e.g., using Federated 
                                                        Averaging (FedAvg)).
Global Model Update
The new aggregated model is sent back to clients, and the cycle repeats.


🔑 Key Techniques in Federated Learning
Federated Averaging (FedAvg) 📊
The most common aggregation algorithm, where updates from multiple clients are 
averaged to update the global model.

Differential Privacy (DP) 🔒
Adds noise to model updates to protect user data from being reconstructed.

Secure Aggregation 🔐
Uses encryption techniques to ensure updates remain private even during aggregation.

Personalized FL 🎭
Allows local models to retain some unique characteristics while still benefiting 
from global knowledge.

📌 Real-World Applications of Federated Learning
Smartphones & AI Assistants 📱
Google’s Gboard keyboard improves text prediction without sending user data to the cloud.

Healthcare & Medical Research 🏥
Hospitals train AI models on patient data locally while preserving confidentiality.

Finance & Fraud Detection 💳
Banks share fraud detection models without exposing sensitive transaction details.

IoT & Edge Computing 🌍

Smart devices learn user preferences without sending raw data to manufacturers.
📌 Simple PyTorch Federated Learning Example

"""
import torch
import torch.nn as nn
import torch.optim as optim
import random

# Sample clients with local datasets
clients = 5
local_data = [torch.randn(100, 10) for _ in range(clients)]  # 5 clients, each with 100 samples of 10 features
local_labels = [torch.randint(0, 2, (100,)) for _ in range(clients)]  # Binary classification

# Simple neural network model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))

# Federated Training
global_model = SimpleModel()
optimizer = optim.SGD(global_model.parameters(), lr=0.01)

def local_train(model, data, labels):
    """Simulate local training on a client."""
    local_model = SimpleModel()
    local_model.load_state_dict(model.state_dict())  # Copy global model
    optimizer = optim.SGD(local_model.parameters(), lr=0.01)
    criterion = nn.BCELoss()

    for _ in range(5):  # Local epochs
        optimizer.zero_grad()
        output = local_model(data).squeeze()
        loss = criterion(output, labels.float())
        loss.backward()
        optimizer.step()

    return local_model.state_dict()

# Federated Averaging
for round in range(10):  # FL rounds
    client_updates = []
    
    for client in range(clients):
        local_model_state = local_train(global_model, local_data[client], local_labels[client])
        client_updates.append(local_model_state)
    
    # Aggregate updates
    new_state_dict = {}
    for key in global_model.state_dict().keys():
        new_state_dict[key] = torch.stack([client_updates[i][key] for i in range(clients)]).mean(dim=0)

    global_model.load_state_dict(new_state_dict)  # Update global model

    print(f"Round {round+1} complete.")


'''
💡 Key Takeaways
✅ Federated Learning = Local Training + Secure Aggregation
✅ Enables Privacy-Preserving AI without centralized data collection
✅ Used in real-world applications like mobile keyboards, healthcare, and IoT
✅ Scalable & Efficient, reducing bandwidth usage

Would you like a more detailed implementation or any modifications to the example? 🚀


The randomized response technique (RRT) is specifically designed to protect the 
privacy of survey participants, particularly when sensitive or personal information
 is being collected.

Here's how it works:

Participants are asked to answer a sensitive question in a way that provides 
privacy protection. Instead of answering directly, they are instructed to answer 
based on a randomization process (often involving a coin flip or some other 
                                  random mechanism).

For example, a survey might ask, "Have you ever committed a crime?" Instead of 
answering directly, the participant might flip a coin. If it lands heads, they 
answer truthfully; if tails, they are instructed to answer "yes" regardless of
 the truth. This way, the survey responses are mixed with random noise, making 
 it impossible to know for sure whether any given answer is truthful.

The key advantage of this method is that it reduces the risk of participants
 being identified based on their answers. Since the response could be the result 
 of the randomization and not the individual's true answer, their privacy is
 better protected, especially in cases where the subject matter is highly sensitive 
 (e.g., illegal behaviors, personal health issues).

In short, yes, the randomized response technique helps protect the privacy of 
survey participants by introducing randomness that masks individual answers.







'''




