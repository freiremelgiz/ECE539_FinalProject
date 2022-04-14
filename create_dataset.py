#!/bin/env python3

"""
Create a dataset and save as data.csv

"""

from MPCNet.dataset.dataset import Dataset
from MPCNet.controller.pyomo_controller import mpcController
from functools import partial

mpc = partial(mpcController,
            plot=False)

# Create Dataset class
dset = Dataset(numRuns=10000, samplesPerRun=100)
dset.generate(mpc)
dset.save('data5-1000000.csv')

