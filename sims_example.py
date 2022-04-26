#!/bin/env python3

import numpy as np
from NMPC_Net.simulation import simulation
from NMPC_Net.controller.MPC import MPC, MPCParams
from NMPC_Net.dataset import dataset

# Number of simulations to run
N_sims = 15
u_freq = 10.0 # [Hz]
tsim = 20.0   # [sec]
data = dataset.Dataset()
data.populate_features(N_sims)

# Setup controller
controller = MPC()
sims = []

# Iterate over each simulation
for i in range(N_sims):
    # Extract data
    x = data.X[i]
    x0 = x[0,0:4]
    xf = x[0,4:]
    # Init simulation
    sims.append(simulation.Simulation(x0, xf, u_freq, controller, stop_r = 0.5))
    sims[i].run_simulation(tsim)

# Plot all paths
simulation.plot_paths(sims, f"paths_sim.png")

