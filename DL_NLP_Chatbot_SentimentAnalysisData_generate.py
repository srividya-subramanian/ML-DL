# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 06:27:03 2025

@author: srivi
"""

import random
import pandas as pd

# Define sample sentence structures for each sentiment
positive_templates = [
    "I absolutely love {}!", "This {} is amazing.", "I'm so happy with {}.", 
    "What a fantastic {}!", "{} makes my day better.", "I feel great about {}.", 
    "I highly recommend {}.", "The best {} ever!"
]

negative_templates = [
    "I really hate {}.", "This {} is terrible.", "I'm so frustrated with {}.", "What an awful {}!", 
    "{} is so disappointing.", "I regret using {}.", "This {} ruined my day.", "I wouldn't recommend {}."
]

neutral_templates = [
    "{} is just okay.", "I'm neutral about {}.", "This {} is neither good nor bad.", "I have no strong feelings about {}.", 
    "{} is fine, I guess.", "I don't care much about {}.", "It's an average {}.", "Nothing special about {}."
]

# Define some random subjects
subjects = ["this movie", "the food", "the service", "this product", "the weather", "my experience", "this book", "this place"]

# Generate data
data = []
for _ in range(333):  # Generate approximately equal number of sentences for each sentiment
    data.append((random.choice(positive_templates).format(random.choice(subjects)), "positive"))
    data.append((random.choice(negative_templates).format(random.choice(subjects)), "negative"))
    data.append((random.choice(neutral_templates).format(random.choice(subjects)), "neutral"))

# Add an extra one to make it 1000
data.append((random.choice(positive_templates).format(random.choice(subjects)), "positive"))

# Convert to DataFrame
df = pd.DataFrame(data, columns=["Sentence", "Sentiment"])

# Save to CSV (optional)
df.to_csv("sentiment_dataset.csv", index=False)

# Display some samples
print(df.head(10))
