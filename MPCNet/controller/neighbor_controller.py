#!/usr/bin/env python3

import numpy as np
import pandas as pd

class NeighborController:

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def getControl(self, initialState, finalState):
        relativeFinalState = finalState
        relativeFinalState[0] -= initialState[0]
        relativeFinalState[1] -= initialState[1]

        newState = np.array([
            initialState[2],
            initialState[3],
            relativeFinalState[0],
            relativeFinalState[1],
            relativeFinalState[2],
            relativeFinalState[3]
        ])

        index = np.argmin(np.sum((self.X - newState)**2, axis=1), axis=0)

        return self.y[index,:].reshape((2))
