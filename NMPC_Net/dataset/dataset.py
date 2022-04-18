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
from ..controller import utils

# seed random number generator (repeatability)
#seed(1)

class Dataset():
    def __init__(self, numRuns=10, samplesPerRun=10):
        self.numRuns = numRuns # Number of runs
        self.numSamplesPerRun = samplesPerRun # Number of samples per run
        self.K = numRuns * samplesPerRun
        # State bounds
        self.x_lb = np.array([-200, -200, 0, -np.pi/2], dtype=np.double).reshape((4,1))
        self.x_ub = np.array([200, 200, 32, np.pi/2], dtype=np.double).reshape((4,1))
        # Initialize dataset matrices
        self.X = [] # Feature matrix
        self.y = [] # Label matrix

    # Get an initial state with speed within bounds:
    # v \in [v_lb, v_ub]
    def _get_rand_x0(self):
        x0 = np.zeros((4,1), dtype=np.double)
        x0[2] = (self.x_ub[2]-self.x_lb[2])*rand(1,1) + self.x_lb[2]
        return x0

    # Get a random final state given initial state x0
    def _get_rand_xf(self, x0): # NOTE: x0 not used currently
        return (self.x_ub - self.x_lb)*rand(4,1) + self.x_lb

    # Generate dataset and fill X and y
    # Pass a controller(x0, xf) object to this
    def generate(self, controller):
        for i in tqdm(range(self.numRuns)):
            x0 = self._get_rand_x0()
            xf = self._get_rand_xf(x0)
            
            traj, control = controller(x0.flatten(), xf.flatten(), fullTrajectory=True)
            
            indices = np.random.choice(traj.shape[0], self.numSamplesPerRun, replace=False)
            initiaStates = traj[indices, :]
            controls = control[indices, :]
            
            for j in range(self.numSamplesPerRun):
                relativeX0, relativeXf = utils.absoluteToRelative(initiaStates[j,:], xf)
                control = controls[j,:]
                inputVector = np.array([
                    relativeX0[0],
                    relativeXf[0],
                    relativeXf[1],
                    relativeXf[2],
                    relativeXf[3]
                ], dtype=np.double)
                self.X.append(inputVector.reshape((1, 5)))
                self.y.append(control.reshape((1,2)))


    # Save the dataset [X, y] as a .csv file
    def save(self, fileName='data.csv'):
        X = np.vstack(self.X).reshape((self.K, 5))
        y = np.vstack(self.y).reshape((self.K, 2))
        np.savetxt(fileName, np.hstack((X, y)),
                delimiter=',')

    @classmethod
    def load(cls, fileName):
        raw = pd.read_csv(fileName, header=None).to_numpy(dtype=np.double)
        X = raw[:,0:5]
        y = raw[:,5:]
        return (X,y)

# Test the class
if __name__=="__main__":
    # Create dummy controller
    def controller(x0, xf):
        return np.ones((2,1))

    # Create class
    dset = Dataset(numRuns=3, samplesPerRun=10)
    assert(not dset.y.any())
    # Generate data
    dset.generate(controller)
    assert(dset.y.all())
    # Save as csv file
    dset.save()
