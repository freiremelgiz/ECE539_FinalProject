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

        index = np.argmin(np.sum((self.X - newState)**2, axis=1), axis=0)

        return self.y[index,:].reshape((2))
