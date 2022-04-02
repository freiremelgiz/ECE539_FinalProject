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
        self.x_lb = np.array((-200, -200, 0, -np.pi)).reshape((4,1))
        self.x_ub = np.array((200, 200, 32, np.pi)).reshape((4,1))
        # Initialize dataset matrices
        self.X = np.zeros((K,8)) # Feature matrix
        self.y = np.zeros((K,2)) # Label matrix

    # Get a random initial state within the following bounds:
    # x \in [x_lb ,x_ub]
    def _get_rand_x0(self):
        return (self.x_ub - self.x_lb)*rand(4,1) + self.x_lb

    # Get a random final state given initial state x0
    def _get_rand_xf(self, x0): # TODO: Make feasible
        return (self.x_ub - self.x_lb)*rand(4,1) + self.x_lb

    # Generate dataset and fill X and y
    # Pass a controller(x0, xf) object to this
    def generate(self, controller):
        for i in tqdm(range(self.K)):
            x0 = self._get_rand_x0()
            xf = self._get_rand_xf(x0)
            self.X[i,:] = np.hstack((x0.T,xf.T))
            self.y[i] = controller(x0.flatten(), xf.flatten()).T

    # Save the dataset [X, y] as a .csv file
    def save(self):
        np.savetxt('data.csv', np.hstack((self.X, self.y)),
                delimiter=',')

    @classmethod
    def load(cls, fileName):
        raw = pd.read_csv(fileName, header=None).to_numpy()
        X = raw[:,0:8]
        y = raw[:,8:-1]
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
