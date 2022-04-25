#/bin/env python3

"""
Create a dataset and save as data.csv

"""

from NMPC_Net.dataset.dataset import Dataset
from NMPC_Net.controller.MPC import MPC, MPCParams
from functools import partial

params = MPCParams()
controller = MPC(params=params)

# Create Dataset class
dset = Dataset(numRuns=200, samplesPerRun=100)
dset.generate(controller)
dset.save('data5-200000.csv')

