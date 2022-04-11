#!/bin/env python3

from MPCNet.simulation import simulation
import numpy as np
from MPCNet.controller import pyomo_controller
from functools import partial
from MPCNet.controller.MPC import MPC


### SIM EXAMPLE
initial = np.array([0.0, 0.0, 0.0, 0.0])
final = np.array([20.0, 20.0, 0.0, np.pi/2])
controller = MPC(final, plot=True)
sim = simulation.Simulation(initial, final, 50.0, controller)
sim.runSimulation(20.0)
simulation.plot_path(sim, f"path_sim.png")
simulation.plot_input(sim, f"input_sim.png")


### PYOMO EXAMPLE
#initial = [70.0, 3.0, 10.0, 0.0]
#final = [100.0, 5.0, 0.0, 0.0]
#pyomo_controller.mpcController(initial, final, plot=True)


### PYOMO CLASS EXAMPLE
#initial = [70.0, 3.0, 10.0, 0.0]
#final = [100.0, 5.0, 0.0, 0.0]
#controller = MPC(final, plot=True)
#controller(initial)
