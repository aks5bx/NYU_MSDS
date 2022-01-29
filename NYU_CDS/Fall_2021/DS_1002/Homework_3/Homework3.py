#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 13:48:42 2021

@author: adisrikanth
"""

import numpy as np
import math 
import matplotlib.pyplot as plt

## PART A ###
samples = np.load('samples.npy')

def F(x):
    expTerm = math.exp(-x)
    return (-expTerm)

results = np.vectorize(F)(samples)
plt.hist(results, bins = 25)

## PART B ##
