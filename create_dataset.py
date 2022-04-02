#!/bin/env python3

"""
Create a dataset and save as data.csv

"""

from MPCNet.dataset.dataset import Dataset
from MPCNet.controller.pyomo_controller import mpcController as mpc

# Create Dataset class
dset = Dataset(K=100)
dset.generate(mpc)
dset.save()

