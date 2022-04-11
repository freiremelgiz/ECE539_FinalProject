#!/bin/env python3

from MPCNet.simulation import simulation
import numpy as np
from MPCNet.controller import pyomo_controller
from functools import partial
from MPCNet.controller.MPC import MPC


### SIM EXAMPLE
#goal = np.array([100.0, 5.0, 0.0, 0.0])
#controller = partial(pyomo_controller.mpcController,
#                     finalState=goal,
#                     plot=True)
#
#sim = simulation.Simulation(
#        np.array([0.0, 0.0, 10.0, 0.0]),
#        100.0,
#        controller)
#
#sim.runSimulation(20.0, goal)
#simulation.plotSimulation(sim, f"run.png")


### PYOMO EXAMPLE
#initial = [70.0, 3.0, 10.0, 0.0]
#final = [100.0, 5.0, 0.0, 0.0]
#pyomo_controller.mpcController(initial, final, plot=True)


### PYOMO CLASS EXAMPLE
initial = [70.0, 3.0, 10.0, 0.0]
final = [100.0, 5.0, 0.0, 0.0]
controller = MPC(final, plot=True)
controller(initial)
