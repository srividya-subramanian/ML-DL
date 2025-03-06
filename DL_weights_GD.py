# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 08:20:58 2025

@author: srivi
"""
'''
1. Standard (Batch) Gradient Descent
Update Rule:
- Uses all training examples per update. ***
- Slower but more stable.
- Used in convex optimization problems.    

Python    
W -= learning_rate * dW   
    
✅ Stable convergence
❌ Slow for large datasets   
    

2. Stochastic Gradient Descent (SGD)
Update Rule:   
- Updates weights after each training example. ***
- Faster but has higher variance.
- Helps escape local minima.

Python    
for i in range(num_samples):
    W -= learning_rate * dW[i]  # Update per sample
    
✅ Fast updates
❌ Noisy convergence    


3. Mini-Batch Gradient Descent
Update Rule:
- Uses small batches (e.g., 32, 64 samples). ***
- Balances speed & stability.

Python    
for batch in range(num_batches):
    W -= learning_rate * dW_batch

✅ Best of both worlds
❌ Requires tuning batch size


4. Momentum-Based Gradient Descent
Update Rule:

- Uses a velocity term  𝑣 to smooth weight updates. ***
- Helps reduce oscillations.

Python
v = beta * v - learning_rate * dW
W += v

✅ Faster convergence
❌ Needs momentum tuning

5. RMSprop (Root Mean Square Propagation)
Update Rule:
- Adapts learning rate per parameter. ***
- Helps in non-stationary problems.

Python
S = beta * S + (1 - beta) * (dW ** 2)
W -= (learning_rate / (np.sqrt(S) + epsilon)) * dW

✅ Adapts learning rate
❌ Requires parameter tuning


6. Adam (Adaptive Moment Estimation)
Combines Momentum & RMSprop
- Fast convergence & adaptive learning rates.

Python
m = beta1 * m + (1 - beta1) * dW
S = beta2 * S + (1 - beta2) * (dW ** 2)
W -= (learning_rate / (np.sqrt(S) + epsilon)) * m

✅ Works well in deep learning
❌ Higher memory usage


Conclusion

 Adam is the most commonly used optimizer for deep learning.
 RMSprop is great for non-stationary problems like reinforcement learning.
 Momentum helps accelerate training in deep networks.







    
 '''   
    
    
    
    
    
    
    
    
    
    
    
    
    