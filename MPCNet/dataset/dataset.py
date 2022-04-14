#!/bin/env python3

"""
This script creates a dataset from random features
x_0 as the initial state (x y v psi)
x_f as the desired final state (x y v psi)
The script chooses random features and uses a
controller(x_0, x_f) function to generate labels.
"""

import numpy as np
from numpy.random import seed
from numpy.random import rand
import pandas as pd
from tqdm import tqdm

# seed random number generator (repeatability)
#seed(1)

class Dataset():
    def __init__(self, K=10):
        self.K = K # Number of samples
        # State bounds
        self.x_lb = np.array((0, -200, 0, -np.pi/2)).reshape((4,1))
        self.x_ub = np.array((200, 200, 32, np.pi/2)).reshape((4,1))
        # Initialize dataset matrices
        self.X = np.zeros((K,5)) # Feature matrix
        self.y = np.zeros((K,2)) # Label matrix

    # Get an initial state with speed within bounds:
    # v \in [v_lb, v_ub]
    def _get_rand_x0(self):
        x0 = np.zeros((4,1))
        x0[2] = (self.x_ub[2]-self.x_lb[2])*rand(1,1) + self.x_lb[2]
        return x0

    # Get a random final state given initial state x0
    def _get_rand_xf(self, x0): # NOTE: x0 not used currently
        return (self.x_ub - self.x_lb)*rand(4,1) + self.x_lb

    # Generate dataset and fill X and y
    # Pass a controller(x0, xf) object to this
    def generate(self, controller):
        x = np.zeros(5) # the i-th feature
        for i in tqdm(range(self.K)):
            x0 = self._get_rand_x0()
            xf = self._get_rand_xf(x0)
            x[0] = x0[2]
            x[1:] = xf.T
            self.X[i,:] = x
            self.y[i] = controller(x0.flatten(), xf.flatten()).T

    # Save the dataset [X, y] as a .csv file
    def save(self, fileName='data.csv'):
        np.savetxt(fileName, np.hstack((self.X, self.y)),
                delimiter=',')

    @classmethod
    def load(cls, fileName):
        raw = pd.read_csv(fileName, header=None).to_numpy()
        X = raw[:,0:5]
        y = raw[:,5:]
        return (X,y)

# Test the class
if __name__=="__main__":
    # Create dummy controller
    def controller(x0, xf):
        return np.ones((2,1))

    # Create class
    dset = Dataset(K=5)
    assert(not dset.y.any())
    # Generate data
    dset.generate(controller)
    assert(dset.y.all())
    # Save as csv file
    dset.save()
