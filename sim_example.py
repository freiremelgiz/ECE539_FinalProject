#!/bin/env python3

from MPCNet.simulation import simulation
import numpy as np
from MPCNet.controller import pyomo_controller
from functools import partial

#controller = partial(pyomo_controller.mpcController,
#                     finalState=np.array([100.0, 5.0, 0.0, 0.0]),
#                     plot=True)
#
#sim = simulation.Simulation(
#        np.array([0.0, 0.0, 10.0, 0.0]),
#        100.0,
#        controller)
#
#sim.runSimulation(10.0)
#simulation.plotSimulation(sim, f"run.png")

initial = [0.0, 0.0, 10.0, 0.0]
final = [100.0, 5.0, 0.0, 0.0]
#for i in range(20):
#    initial = [0.0, 0.0, 0.0, (i/20)*2*np.pi]
pyomo_controller.mpcController(initial, final, plot=True)
