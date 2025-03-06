# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 10:42:22 2025

@author: srivi
"""
'''
Backpropagation vs. Backpropagation Through Time (BPTT)
Both backpropagation (BP) and backpropagation through time (BPTT) are techniques 
for computing gradients in neural networks, but they differ in how they handle 
sequential data and time dependencies.

1. Backpropagation (BP)
Definition:
Backpropagation is the fundamental algorithm used to compute gradients in feedforward 
neural networks (FNNs), including deep neural networks (DNNs) and convolutional 
neural networks (CNNs). It uses the chain rule of differentiation to compute the 
gradients of the loss function with respect to network parameters (weights and biases).

How It Works:
    
Forward Pass: Compute activations from input to output.

Loss Calculation: Compute the loss (error) between the predicted and actual output.

Backward Pass (Backpropagation): Compute gradients of the loss function using the 
chain rule and propagate them layer by layer in reverse order.

Update Weights: Use an optimizer (e.g., SGD, Adam) to update the weights based 
on the computed gradients.

Key Features of BP:
- Used in feedforward networks (e.g., CNNs, DNNs).
- Works with non-sequential data.
- No temporal dependencies; each training sample is processed independently.

2. Backpropagation Through Time (BPTT)
Definition:
BPTT is an extension of backpropagation specifically designed for Recurrent Neural
Networks (RNNs), which process sequential data and have time dependencies 
(feedback loops). Since RNNs reuse the same parameters across multiple time steps, 
BPTT unrolls the network through time and applies backpropagation to compute 
gradients across all time steps.

How It Works:
    
Unrolling the RNN: Since RNNs operate over sequences, the network is unrolled 
across time steps (e.g., for a sequence of length T, the RNN is treated as a T-layer 
                   deep network with shared weights).

Forward Pass: Compute activations at each time step using shared weights.

Loss Calculation: Compute the total loss over all time steps.

Backward Pass (BPTT):
Gradients are computed across all time steps, starting from the last time step 
and moving backward.
Uses the chain rule to propagate gradients through time, updating weights at 
each time step.

Weight Update: Apply updates using an optimizer.

Key Features of BPTT:
- Used in Recurrent Neural Networks (RNNs).
- Handles sequential data where time dependencies matter.
- Computes gradients across multiple time steps.
- Can suffer from vanishing or exploding gradients when dealing with long sequences.

3. Differences Between BP and BPTT
Feature	Backpropagation (BP)	Backpropagation Through Time (BPTT)
Use Case	Feedforward networks (DNNs, CNNs)	Recurrent Neural Networks (RNNs)
Sequential Data	No	Yes
Time Dependencies	No	Yes (updates occur across multiple time steps)
Gradient Computation	Backpropagates through network layers	Backpropagates through both layers and time steps
Unrolling Required?	No	Yes, unrolls RNN across time
Common Issues	No major gradient issues	Vanishing/exploding gradients (especially for long sequences)
4. Truncated BPTT (TBPTT)
Since BPTT can be computationally expensive for long sequences, a variation called Truncated Backpropagation Through Time (TBPTT) is often used.

How TBPTT Works:
Instead of computing gradients over all time steps, the network truncates the 
sequence at a fixed length (e.g., 10–20 time steps).
The loss is backpropagated only for these shorter segments, making training more efficient.
Helps mitigate vanishing/exploding gradients for long sequences.
TBPTT is widely used in real-world applications of RNNs, such as language modeling, 
speech recognition, and time series forecasting.

5. Summary
Backpropagation (BP) is used in feedforward networks (e.g., CNNs, DNNs) and computes 
gradients layer-by-layer.
BPTT extends BP for RNNs, computing gradients across both time steps and layers.
BPTT is computationally expensive, and Truncated BPTT is used to optimize training efficiency.


'''