# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 13:40:02 2025

@author: srivi
"""

def w_sum(a,b):
    assert(len(a) == len(b))
    output = 0
    for i in range(len(a)):
        output += (a[i] * b[i])
    return output

def neural_network(input, weights):
    pred = w_sum(input,weights)
    return pred

weights = [0.1, 0.2, 0]
toes = [8.5, 9.5, 9.9, 9.0]
wlrec = [0.65, 0.8, 0.8, 0.9]
nfans = [1.2, 1.3, 0.5, 1.0]
input = [toes[0],wlrec[0],nfans[0]]


pred = neural_network(input,weights)
print(pred)



toes = [8.5, 9.5, 9.9, 9.0]
wlrec = [0.65, 0.8, 0.8, 0.9]
nfans = [1.2, 1.3, 0.5, 1.0]


weight = 0.5
input = 0.5
goal_prediction = 0.8
step_amount = 0.015 
for iteration in range(100): 
    prediction = input * weight
    error = (prediction - goal_prediction) ** 2
#    print("Fehler:" + str(error) + " Vorhersage:" +str(prediction),iteration)
    up_prediction = input * (weight + step_amount) 
    up_error = (goal_prediction - up_prediction) ** 2
    down_prediction = input * (weight - step_amount) 
    down_error = (goal_prediction - down_prediction) ** 2
    if(down_error < up_error):
        weight = weight - step_amount 
    if(down_error > up_error):
        weight = weight + step_amount


weight = 0.5
goal_pred = 0.8
input = 0.5
for iteration in range(20):
    pred = input * weight
    error = (pred - goal_pred)** 2
    direction_and_amount = (pred - goal_pred) * input 
    weight = weight - direction_and_amount
    print("Fehler:" + str(error) +" Vorhersage:" + str(pred))
    break




