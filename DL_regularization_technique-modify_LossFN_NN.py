# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 13:37:41 2025

@author: srivi
"""
'''
# Modifying a Loss Function to Guide Weights Toward a Specific Pattern
# The loss function is the key driver of how a neural network learns. If we want 
# the network weights to follow a certain pattern, we can modify the loss 
# function by adding custom regularization terms or constraint-based penalties.

1️⃣   Regularizazion techniques

L2 - Ridge regression - encourages weights to be small but nonzero.
L_new = L_original + λ ∑w**2
L1 - Lasso regression - encourages sparsity - some weights become exactly zero
L_new = L_original + λ ∑|w|


2️⃣   Custom Weight Patterns (Encouraging Specific Shapes in Weights)
If we want weights to follow a specific pattern (e.g., symmetry, smooth transitions), 
we can define a custom regularization term.

🔹 Example: Encourage Symmetry in Weights
If we want the weight matrix W to be symmetric (𝑊=𝑊**𝑇), we can add a loss term:

L_new = L_original  + λ∣∣W−W**T∣∣**2

def custom_loss(y_true, y_pred, W, lambda_reg=0.01):
    base_loss = np.mean((y_true - y_pred) ** 2)  # MSE loss
    symmetry_loss = lambda_reg * np.sum((W - W.T) ** 2)  # Penalizes non-symmetric 
    weights
    return base_loss + symmetry_loss

✅ Use Case: If you want symmetric weights (useful in autoencoders and physics-
                                           inspired models).


3️⃣   Penalizing Certain Weight Directions

If we want weights to follow a certain distribution (e.g., favor some neurons 
over others), we can introduce directional constraints.

Example: Encourage Weights to be Close to a Target Pattern
If we want weights to stay close to a pre-defined pattern 𝑊_target, we can define:
L_new = L_original  + λ∣∣W−W_target∣∣**2

def custom_loss(y_true, y_pred, W, W_target, lambda_reg=0.01):
    base_loss = np.mean((y_true - y_pred) ** 2)  # MSE loss
    pattern_loss = lambda_reg * np.sum((W - W_target) ** 2)  
    return base_loss + pattern_loss

✅ Use Case: If you want weights to learn a predefined shape or structured patterns.

4️⃣ Constraint-Based Learning (Hard Constraints)
Instead of modifying the loss function, you can directly constrain the weight 
updates during training.

🔹 Example: Force Weights to be Positive
W = np.maximum(W, 0)  # Ensures all weights are non-negative after an update
✅ Use Case: Useful in biological models or certain economic models where weights 
must be non-negative.

Summary

Technique	            Effect on Weights	                  Use Case
L2 Regularization	    Encourages small weights	          Prevents overfitting
L1 Regularization	    Encourages sparse weights	          Feature selection, compression
Symmetry Regularization	Makes weight matrix symmetric	      Autoencoders, structured learning
Pattern Matching Loss	Forces weights to resemble a given pattern	 Custom neural networks, specific feature emphasis
Hard Constraints	    Forces specific weight properties	  Domain-specific applications
'''