# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 11:17:50 2025

@author: srivi
"""

import numpy as np
streetlights = np.array([ 
    [ 1, 0, 1 ],
    [ 0, 1, 1 ],
    [ 0, 0, 1 ],
    [ 1, 1, 1 ],
    [ 0, 1, 1 ],
    [ 1, 0, 1 ] ])

walk_vs_stop = np.array([ 
    [ 0 ],
    [ 1 ],
    [ 0 ],
    [ 1 ],
    [ 1 ],
    [ 0 ] ])

weights = np.array([0.5,0.48,-0.7])
alpha = 0.1


#''for single training example light'''
#input = streetlights[1] 
#goal_prediction = walk_vs_stop[1]
#for iteration in range(20):
#    prediction = input.dot(weights)
#    error = (goal_prediction - prediction) ** 2
#    delta = prediction - goal_prediction
#    weights = weights - (alpha * (input * delta))
#    print("Fehler:" + str(error) + " Vorhersage:" + str(prediction))

'''train it with all combinations of traffic light lights'''
for iteration in range(40):
    error_for_all_lights = 0
    for row_index in range(len(walk_vs_stop)):
        input = streetlights[row_index]
        goal_prediction = walk_vs_stop[row_index]
        prediction = input.dot(weights)
        error = (goal_prediction - prediction) ** 2
        error_for_all_lights += error
        delta = prediction - goal_prediction
        weights = weights - (alpha * (input * delta))
        print("Vorhersage:" + str(prediction))
    print("Fehler:" + str(error_for_all_lights) + "\n")











