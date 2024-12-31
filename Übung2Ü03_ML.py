# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 12:46:03 2024

@author: srivi
"""
import numpy as np
import numpy.random as rand
import pandas as pd

def random_normal_distribution(mu,sig,n): # n is the length of the random array
    return rand.RandomState(1).normal(mu, sig, n)


f1 = random_normal_distribution(20, 5, 100)
f2 = random_normal_distribution(30, 5, 100)

df = pd.DataFrame(zip(f1,f2))
df.columns = ['Feature 1', 'Feature 2']

mean = df.mean()
std = df.std()

std_df = (df - mean)/std

    