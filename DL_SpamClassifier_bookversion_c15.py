# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 04:36:15 2025

@author: srivi

Federated learning is about providing a secure environment for the model and 
learning how to solve a task without transmitting the data anywhere. 


"""

import codecs

# Open file safely and read line-by-line
with codecs.open('C:/Users/srivi/Documents/velptec_K4.0016_2.0_DL_Buchmaterial/spam.txt',
                 "r", encoding='utf-8', errors='ignore') as f:
    raw = f.readlines()

# Initialize sets and lists
vocab, spam, ham = set(["<unk>"]), [], []

for row in raw:
    cleaned_row = row.strip()  # Removes extra spaces and newline characters
    if cleaned_row:  # Avoid processing empty lines
        words = set(cleaned_row.split())  # Convert to a set of unique words
        spam.append(words)  # Add words to spam list
        vocab.update(words)  # Add words to vocabulary


with codecs.open('C:/Users/srivi/Documents/velptec_K4.0016_2.0_DL_Buchmaterial/ham.txt',
                 "r",encoding='utf-8',errors='ignore') as f:
    raw = f.readlines()

for row in raw:
    cleaned_row = row.strip()
    if cleaned_row:
        words = set(cleaned_row.split())  # Convert to a set of unique words
        ham.append(words)  # Add words to spam list
        vocab.update(words)  # Add words to vocabulary  
    
vocab, w2i = (list(vocab), {})

for i,w in enumerate(vocab):
    w2i[w] = i

def to_indices(input, l=500):
    indices = list()
    for line in input:
        if(len(line) < l):
            line = list(line) + ["<unk>"] * (l - len(line))
            idxs = list()

            for word in line:
                idxs.append(w2i[word])
            indices.append(idxs)
    return indices    
    
spam_idx = to_indices(spam) 
ham_idx = to_indices(ham) 
train_spam_idx = spam_idx[0:-1000] 
train_ham_idx = ham_idx[0:-1000] 
test_spam_idx = spam_idx[-1000:] 
test_ham_idx = ham_idx[-1000:] 
train_data = list()
train_target = list()
test_data = list()    
test_target = list()

for i in range(max(len(train_spam_idx),len(train_ham_idx))): 
    train_data.append(train_spam_idx[i%len(train_spam_idx)]) 
    train_target.append([1]) 
    train_data.append(train_ham_idx[i%len(train_ham_idx)]) 
    train_target.append([0])

for i in range(max(len(test_spam_idx),len(test_ham_idx))): 
    test_data.append(test_spam_idx[i%len(test_spam_idx)]) 
    test_target.append([1]) 
    test_data.append(test_ham_idx[i%len(test_ham_idx)]) 
    test_target.append([0])

def train(model, input_data, target_data, batch_size=500,iterations=5): 
    n_batches = int(len(input_data) / batch_size)
    for iter in range(iterations): 
        iter_loss = 0
        for b_i in range(n_batches):
            # Padding token should stay at 0 
            model.weight.data[w2i['<unk>']] *= 0
            input = Tensor(input_data[b_i*bs:(b_i+1)*bs],autograd=True) 
            target = Tensor(target_data[b_i*bs:(b_i+1)*bs],autograd=True) 
            pred = model.forward(input).sum(1).sigmoid()
            loss = criterion.forward(pred,target) 
            loss.backward()
            optim.step()
            iter_loss += loss.data[0] / bs
            sys.stdout.write("\r\tLoss:" + str(iter_loss /(b_i+1)))
        print() 
    return model

def test(model, test_input, test_output): 
    model.weight.data[w2i['<unk>']] *= 0
    input = Tensor(test_input, autograd=True) 
    target = Tensor(test_output, autograd=True) 
    pred = model.forward(input).sum(1).sigmoid()
    return((pred.data > 0.5) == target.data).mean()
model = Embedding(vocab_size=len(vocab), dim=1) 
model.weight.data *= 0
criterion = MSELoss()
optim = SGD(parameters=model.get_parameters(), alpha=0.01)

for i in range(3):
    model = train(model, train_data, train_target,iterations=1)
    print("% correct for test data: "+str(test(model, test_data, test_target)*100))