#!/usr/bin/env python3
import numpy as np
import pandas as pd
from . import utils

class NeighborController:

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def getControl(self, initialState, finalState):

        relativeInitial, relativeFinal = utils.absoluteToRelative(initialState, finalState)

        newState = np.array([
            relativeInitial[0],
            relativeFinal[0],
            relativeFinal[1],
            relativeFinal[2],
            relativeFinal[3]
        ])
        
        cost = np.sum((self.X[:,:4] - newState[:4])**2, axis=1)
        cost += (np.pi/2 - np.abs(np.abs(self.X[:,4] - newState[4]) - np.pi/2))**2


        index = np.argmin(cost, axis=0)

        return self.y[index,:].reshape((2))
